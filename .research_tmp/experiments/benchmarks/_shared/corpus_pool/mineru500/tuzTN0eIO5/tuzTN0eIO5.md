# ZERO BUBBLE (ALMOST) PIPELINE PARALLELISM

Penghui Qi\*, Xinyi Wan\*, Guangxing Huang & Min Lin

Sea AI Lab

{qiph, wanxy, huanggx, linmin}@sea.com

# ABSTRACT

Pipeline parallelism is one of the key components for large-scale distributed training, yet its efficiency suffers from pipeline bubbles which were deemed inevitable. In this work, we introduce a scheduling strategy that, to our knowledge, is the first to successfully achieve zero pipeline bubbles under synchronous training semantics. The key idea behind this improvement is to split the backward computation into two parts, one that computes gradient for the input and another that computes for the parameters. Based on this idea, we handcraft novel pipeline schedules that significantly outperform the baseline methods. We further develop an algorithm that automatically finds an optimal schedule based on specific model configuration and memory limit. Additionally, to truly achieve zero bubble, we introduce a novel technique to bypass synchronizations during the optimizer step. Experimental evaluations show that our method outperforms the 1F1B schedule up to 15% in throughput under a similar memory limit. This number can be further pushed to 30% when the memory constraint is relaxed. We believe our results mark a major step forward in harnessing the true potential of pipeline parallelism. The source code based on Megatron-LM is publicly available at https://github.com/sail-sg/zero-bubble-pipeline-parallelism.

# 1 INTRODUCTION

The realm of distributed model training has become a focal point in the deep learning community, especially with the advent of increasingly large and intricate models. Training these behemoths often requires a vast amount of GPUs interconnected with various topologies. Various parallelism techniques have been proposed for training DNN in the past years. Data parallelism (DP) (Goyal et al., 2017; Li et al., 2020) is the default strategy for models of small to moderate sizes due to its simplicity. Beyond a model size, it is no longer possible to fit the model parameters in one single GPU. This is when model parallelism comes to the rescue (Harlap et al., 2018; Huang et al., 2019; Fan et al., 2021; Zheng et al., 2022). There are two main model parallel schemes, tensor parallelism (TP) and pipeline parallelism (PP). TP splits the matrix multiplication in one layer to several devices, while PP segments the entire model into different stages which can be processed across different devices. Notably, ZeRO (Rajbhandari et al., 2020) provides a strong alternative to model parallelism by sharding parameters across devices, while keeping the simplicity of DP.

Recent research indicates that achieving optimal performance in large-scale training scenarios requires a non-trivial interaction of DP, TP and PP strategies. In the abundance of interconnection resources, e.g. NVLink between GPUs within one compute node, a hybrid of DP, TP and ZeRO strategies works efficiently. Whereas there are numerous empirical evidences Fan et al. (2021); Zheng et al. (2022); Narayanan et al. (2021) showing PP is particularly advantageous for utilizing cross-server connections, especially at the scale of thousands of GPUs. This highlights the primary aim of our work: enhancing the efficiency of PP.

Going deeper into the intricacies of PP, the efficiency of its implementation relies heavily on the amount of device idle time referred to as pipeline bubbles. Due to the dependency between layers, bubbles seem inevitable. A prominent early work to address this issue is GPipe (Huang et al., 2019), which attempts to reduce the bubble ratio by increasing the number of concurrent batches in the pipeline. However, a direct consequence of this is an increase in peak memory demands.

To mitigate this, GPipe discards part of the intermediate activations while recomputing them during the backward pass. Yet, this approach introduced a computation overhead of around 20% (Fan et al., 2021). One line of work that improves over GPipe focuses on asynchronous PP, including PipeDream (Harlap et al., 2018), PipeMare (Yang et al., 2021). Asynchronous PP is theoretically bubble free, they greatly improve pipeline efficiency, however, at the sacrifice of exact optimization semantics. On the other hand, improvements are also made under synchronous settings. A notable scheduling strategy to address the limitation of GPipe is called one-forward-one-backward (1F1B). It was first proposed in PipeDream (Harlap et al., 2018) under the asynchronous setting, and later introduced under synchronous settings (Fan et al., 2021; Narayanan et al., 2021). 1F1B offers faster memory clearance by early scheduling the backward passes. With the same number of microbatches, it yields similar bubble ratios but with a distinct advantage in peak memory. Based on 1F1B, Narayanan et al. (2021) introduced the 1F1B interleaved strategy. By assigning multiple stages to the same device, it further reduces the bubble size at the cost of more communication and higher peak memory.

Despite various efforts, to this date the remaining bubbles still pose the largest issue for PP under synchronous training semantics. In this work, we spotted the opportunity that PP can be further optimized by representing and scheduling the computation graph at a finer granularity. Classical deep learning frameworks are designed at the granularity of layers, whereas modern deep learning compilers use different intermediate representations for optimizations at various levels. (Chen et al., 2018; Roesch et al., 2018; Sabne, 2020; Tillet et al., 2019; Lattner et al., 2020). Although a finer granularity always means a larger space for searching, it is often impeded by the lack of optimization tools to navigate the space. Therefore, choosing a suitable granularity is crucial.

![](images/fa4fe4cb59b5b12994deda59bddb7e3b805e3ae4c28ab66cf136e1def455f43e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Forward x"] --> B["Wx"]
    B --> C["z"]
    C --> D["σ(z)"]
    D --> E["y"]
    E --> F["F"]
    G["Backward ∇xL"] --> H["W^T∇zL"]
    H --> I["dσ(z)/dz∇yL"]
    I --> J["∇zL"]
    J --> K["∇zLx^T"]
    K --> L["∇wL"]
    M["N×"] --> B
    N["N×"] --> C
    O["X"] --> B
    P["X"] --> D
    Q["X"] --> H
    R["X"] --> I
    S["X"] --> K
```
</details>

Figure 1: Computation Graph for MLP.

Traditionally, neural networks are granularized as stacked layers. There are two functions associated with each layer, forward and backward. In the forward pass, the input x is transformed into the output y with the parameterized mapping $f(\boldsymbol{x}, \boldsymbol{W})$ . The backward pass, crucial for training, involves two computations: $\nabla_{\boldsymbol{x}} f(\boldsymbol{x}, \boldsymbol{W})^{\top} \frac{d\ell}{dy}$ and $\nabla_{\boldsymbol{W}} f(\boldsymbol{x}, \boldsymbol{W})^{\top} \frac{d\ell}{dy}$ . Correspondingly, they compute the gradient with respect to the input x and the layer's parameters W. For convenience, we use single letters B and W to denote these two computations respectively, and F to denote forward pass (Figure 1). Traditionally, B and W are grouped and provided as a single backward function. This design is conceptually friendly to the user, and it happens to work well for DP, because the communication of the weights' gradient at layer i can be overlapped with the backward computation at layer i - 1. However, in PP, this design unnecessarily increases the sequentially dependent computations, i.e. B at the layer i - 1 depends on W at the layer i, which is usually detrimental for the efficiency of the pipeline.

Based on split B and W, we present new pipeline schedules that greatly improve pipeline efficiency. The remainder of this paper is organized as follows: In Section 2, we introduce handcrafted schedules based on an ideal assumption that the execution times of F, B and W are identical. Subsequently,

in Section 3, we remove this assumption and propose an automatic scheduling algorithm that works under more realistic conditions. To achieve zero bubble, Section 4 details a method that sidesteps the need for synchronization during the optimizer step, yet preserves synchronous training semantics. We conclude with empirical evaluations of our methods against baseline methods under diverse settings.

We should note that we do not aim to explore general mixed strategies for large scale distributed training. Instead, we specifically target to improve the efficiency of pipeline scheduling, supported with apple to apple comparisons with baselines. Our method is orthogonal to DP, TP and ZeRO strategies, and it can be used as a parallel replacement for the PP part in large scale training.

# 2 HANDCRAFTED PIPELINE SCHEDULES

Based on the key observation that splitting B and W could reduce sequential dependency and thus improve efficiency, we redesign the pipeline starting from the commonly utilized 1F1B schedule. As depicted in Figure 2, 1F1B initiates with a warm-up phase. In this phase, workers conduct varying numbers of forward passes, with each stage typically performing one more forward pass than its immediately subsequent stage. Following the warm-up phase, each worker transits to a steady state where they alternately execute one forward pass and one backward pass, ensuring an even workload distribution among stages. In the final phase, each worker processes the backward passes for the outstanding in-flight microbatches, completing the batch.

In our improved version we split the backward pass into B and W passes, it is imperative that F and B from the same microbatch must still remain sequentially dependent across pipeline stages. However, W can be flexibly scheduled anywhere after the corresponding B of the same stage. This allows for strategic placement of W to fill the pipeline bubbles. There are many possible schedules that improve over 1F1B, trading off differently on the bubble size and the memory footprint. We introduce two particularly interesting handcrafted schedules in this section to show the great potential of finer granularity at reducing pipeline bubbles (see Figure 3). For the sake of clarity in our initial design, we assume that the time costs for F, B, and W are identical, an assumption shared by earlier studies (Narayanan et al., 2021; Huang et al., 2019). However, in Section 3, we re-evaluate this assumption to optimize scheduling efficiency in real-world scenarios.

![](images/0008471e6a0ae067e3fb1b78068faaa51c43bc62c30a3bf8126df9032cd7fd83.jpg)  
Figure 2: 1F1B pipeline schedule.

![](images/5b0f336be6f6b2f1d8c8e4e739312b847c41b9cfcbe0cf4c2204d46a597f7efc.jpg)  
Figure 3: Handcrafted pipeline schedules, top: ZB-H1; bottom: ZB-H2

# 2.1 MEMORY EFFICIENT SCHEDULE

Our first handcrafted schedule, named ZB-H1, ensures that the maximum peak memory usage over all workers doesn't exceed that of 1F1B. ZB-H1 generally follows the 1F1B schedule, but it adjusts

the starting points of W depending on the number of warm-up microbatches. This ensures all workers maintain the same number of in-flight microbatches. As a result, as seen in Figure 3 (top), the bubble size is reduced to a third of 1F1B's size. This reduction is because B is initiated earlier across all workers compared to 1F1B, and the tail-end bubbles are filled by the later-starting W passes. As W typically uses less memory than B (Table 1), the first worker has the maximum peak memory usage which is consistent with 1F1B.

# 2.2 ZERO BUBBLE SCHEDULE

When we permit a larger memory footprint than 1F1B and have a sufficient number of microbatches, it's possible to achieve a zero bubble schedule, which we label as ZB-H2. As illustrated in Figure 3 (bottom), we introduce more $F$ passes during the warm-up phase to fill the bubble preceding the initial $B$ . We also reorder the $W$ passes at the tail, which changes the layout from trapezoid into a parallelogram, eliminating all the bubbles in the pipeline. It is important to highlight that the synchronization between the optimizer steps is removed here, we discuss how this is safely done in Section 4.

# 2.3 QUANTITATIVE ANALYSES

We use p to denote the number of stages and b to denote the size of each microbatch. For transformer architecture, we denote the number of attention heads as a, the sequence length as s and the hidden dimension size as h. We use the notations $M_{B}/M_{W}$ to represent the memory required to store activations for one B/W pass, and $T_{F}/T_{B}/T_{W}$ to represent the running time for one F/B/W pass. For simplicity, we only do quantitative analyses on transformer architecture (Vaswani et al., 2017), using a typical setting similar to GPT-3 (Brown et al., 2020) where the hidden dimension size inside feedforward is 4h and the dimension size for each attention head is h/a.

As in Narayanan et al. (2021), we only consider matmul operations when calculating FLOPs because they contribute most of the computations in a transformer layer. For each matmul operation in the forward pass, there are two matmul operations with the same FLOPs in corresponding backward pass (see Figure 1), each of which belongs to either B or W. The approximate formula for calculating the FLOPs of a transformer layer is in Table 1. We can see that $T_{W} < T_{F} < T_{B}$ and $T_{B} + T_{W} = 2T_{F}$ . We use the same method in Korthikanti et al. (2023) to estimate activations memory required for B. After B completes, it releases some activations not used anymore but keeps some extra gradients ( $\nabla_{z}L$ in Figure 1) for W. The total memory required by W, as in Table 1, is less than B.

Table 1: FLOPs and activations memory required per transformer layer for each pass 

<table><tr><td>Pass</td><td>FLOPs</td><td>Activations Memory Required</td></tr><tr><td>F</td><td>sbh(24h + 4s)</td><td>0</td></tr><tr><td>B</td><td>sbh(24h + 8s)</td><td>sb(34h + 5as)</td></tr><tr><td>W</td><td>sbh(24h)</td><td>32sbh</td></tr></table>

Without the assumption of $T_{F} = T_{B} = T_{W}$ , the peak activations memory and bubble size of ZB-H1 and ZB-H2 are quantified in Table 2. Notably, the activations memory of worker i is $(p - i + 1)M_{B} + (i - 1)M_{W}$ for ZB-H1 and $(2p - 2i + 1)M_{B} + (2i - 2)M_{W}$ for ZB-H2. As in Table 1, the activations memory required for W is smaller than that for B. Therefore, the peak activations memory is $pM_{B}$ and $(2p - 1)M_{B}$ , for ZB-H1 and ZB-H2 respectively.

Table 2: Comparison between 1F1B and our handcrafted schedules. 

<table><tr><td>Schedule</td><td>Bubble size</td><td>Peak activations memory</td></tr><tr><td>1F1B</td><td> $(p - 1)(T_F + T_B + T_W)$ </td><td> $pM_B$ </td></tr><tr><td>ZB-H1</td><td> $(p - 1)(T_F + T_B - T_W)$ </td><td> $pM_B$ </td></tr><tr><td>ZB-H2</td><td> $(p - 1)(T_F + T_B - 2T_W)$ </td><td> $(2p - 1)M_B$ </td></tr></table>

# 3 AUTOMATIC PIPELINE SCHEDULING

While handcrafted schedules offer simplicity and better comprehensibility, they face several issues in practical applications. For one, scheduling under the assumption that $T_{F} = T_{B} = T_{W}$ introduces unwanted bubbles, especially for models where these values differ significantly. Moreover, communication time (denoted as $T_{comm}$ ) required to transfer activation/gradient between stages is often ignored in handcrafted schedules, leading to noticeable latencies in the pipeline stream. Finally, striking a balance between minimizing bubble size and adhering to memory limit becomes particularly challenging when the available memory is insufficient to accommodate enough microbatches for a bubble-free schedule.

To address these challenges and ensure generalization to practical scenarios, we propose algorithms to automatically search the optimal schedule given the number of pipeline stages p, the number of microbatches m, the activations memory limit $M_{limit}$ , and the running time estimations $T_{F}$ , $T_{B}$ , $T_{W}$ and $T_{comm}$ . We design a heuristic strategy, which always generates an optimal or near optimal solution especially when m is large enough. We also systematically formulate the problem as Integer Linear Programming (for more details see Appendix G), which can be solved by an off-the-shelf ILP solver (Forrest & Lougee-Heimer, 2005) when the problem is under a certain scale. These two approaches can be combined: first, use the heuristic solution as initialization, and then optimize it further with ILP.

# 3.1 THE HEURISTIC ALGORITHM

We present our heuristic algorithm in the following steps:

- In the warm-up phase, within the memory limit, we schedule as many $F$ passes as possible to minimize the bubble before the first $B$ . The resulting schedule may still have a small bubble (less than $T_F$ ) before the first $B$ if not reaching memory limit, where scheduling another $F$ may delay the following $B$ . We use a binary hyperparameter to control whether to do it or not.   
- After the warm-up phase, we adhere to the pattern where one $F$ and one $B$ are scheduled iteratively. We insert $W$ to fill the bubble when there is a gap larger than $T_W$ . When a bubble occurs but the size is less than $T_W$ , we still insert a $W$ if the current bubble makes the largest cumulative bubble size among all stages become larger. We also insert $W$ to recycle some memory when the memory limit is hit. Typically, our heuristic strategy enters a steady state that follows $1F - 1B - 1W$ pattern.   
- Throughout this process, pipeline stage $i$ is always guaranteed to schedule at least one more $F$ than stage $i + 1$ anytime before $F$ is used up. When this difference exceeds one, we use another binary hyperparameter to decide whether to skip one $F$ in pipeline stage $i$ if it doesn't cause more bubbles. We perform a grid search to find the best combination of hyperparameters.   
- In each stage, when $F$ and $B$ passes run out, we schedule all the left $W$ passes one by one.

# 4 BYPASSING OPTIMIZER SYNCHRONIZATIONS

In most practices of PP, synchronizations over pipeline stages are usually performed in optimizer step for the sake of numerical robustness. For example, a global gradient norm needs to be computed for gradient norm clipping (Pascanu et al., 2013); a global check for NAN and INF values are performed in the mixed precision settings (Micikevicius et al., 2017); both of them require an all-reduce communication across all stages. However, synchronization at the optimizer step destroys the parallelogram (Figure 3) and makes zero bubble impossible. In this section, we propose an alternative mechanism to bypass these synchronizations, while still maintaining a synchronous optimization semantics.

In existing implementations, an all-reduce communication is first launched to collect the global states, followed by the optimizer steps which are conditioned on the global states. However, we noticed that most of the time the global states have no effects, e.g., the global check for NAN and INF rarely trigger because in a robust setting most iterations shouldn't have numerical issues; the gradient clipping rate is also quite low empirically to justify a synchronization of global gradient norm at every iteration.

Based on these observations, we propose to replace the before-hand synchronizations with a post update validation. The idea is illustrated in Figure 4, at each stage before the optimizer step, a partially reduced global state is received from the previous stage, combined with the current stage's local state, and passed on to the next stage. The optimizer step of each stage is controlled by the partially reduced state, e.g. skip the update when a NAN is spotted or the partially reduced gradient norm exceeds the clipping threshold. During the warm-up phase of the next iteration, the fully reduced global state is then propagated back from the last stage to first stage. Upon receiving the global state, each stage performs a validation to decide whether the previous optimizer step is legitimate. If an amendment to the gradient is required, a rollback will be issued (for more details see Appendix C) and then we redo the optimizer step based on the fully reduced global state.

![](images/f3ef2f81233976d0f4da8e8b9a5199fbd084f7ee7bfe38b9fca5db2485798879.jpg)

<details>
<summary>text_image</summary>

1
2
3
4
5
6
7
8
</details>

![](images/82dcc52854e08fd5ea5a25f435656c97d497b9add65f310bf84f0c1614891aa1.jpg)

Reduce local values by propagating from 1 to 4

![](images/31f0f39c144ab1f718f2250ec0df243ef0579194ff986b2453de1ebd87afc251.jpg)

Optimizer step

![](images/ceb74a1e2fc8acae3b04d90ca557c66ce675ceeb207bd3a9189c0baebf94dee3.jpg)

Propagate globally reduced value to each stage

![](images/2b14cfb5ef3128045003517dd92296bc55f6b56630b998ce8e307c5fc4a22fd9.jpg)

Rollback if validation fails

Figure 4: The post-validation strategy to replace optimizer synchronization.

# 5 EXPERIMENTS

# 5.1 SETUP

We base our implementation on the open-source Megatron-LM project (Narayanan et al., 2021) and assess its performance using models analogous to GPT-3 (Brown et al., 2020), as detailed in Table 3. During our experiments, we first conducted a specific number of iterations for profiling, collecting empirical measurements for $T_F$ , $T_B$ , $T_W$ , and $T_{\mathrm{comm}}$ . After obtaining these values, we fed them into our automatic pipeline scheduling algorithm to determine the optimal schedule. It's worth noting that both the initial and final pipeline stages possess one fewer transformer layer compared to the intermediate stages. This design is to compensate for the extra embedding lookup and loss computations in the initial and final stages so that they won't become the bottleneck and cause bubbles to other stages.

Table 3: Models and fixed settings used in experiments 

<table><tr><td>Model</td><td>Layers</td><td>Attention Heads</td><td>Hidden Size</td><td>Sequence Length</td><td>Pipelines (GPUs)</td><td>Microbatch Size</td><td>Number of Microbatches</td></tr><tr><td>1.5B</td><td>22</td><td>24</td><td>2304</td><td>1024</td><td>8</td><td>6</td><td>24 / 32 / 64</td></tr><tr><td>6.2B</td><td>30</td><td>32</td><td>4096</td><td>1024</td><td>8</td><td>3</td><td>24 / 32 / 64</td></tr><tr><td>14.6B</td><td>46</td><td>40</td><td>5120</td><td>1024</td><td>16</td><td>1</td><td>48 / 64 / 128</td></tr><tr><td>28.3B</td><td>62</td><td>48</td><td>6144</td><td>1024</td><td>32</td><td>1</td><td>96 / 128 / 256</td></tr></table>

# Compared methods:

- ZB-1p: Automatically searched schedule with the activation memory limited to $pM_B$ , which theoretically has the same peak memory as 1F1B.   
- ZB-2p: Automatically searched schedule with the activation memory limited to $2pM_B$ , which is the least amount of memory to empirically achieve close to zero bubble (see Figure 7).   
- 1F1B and 1F1B-I: 1F1B and interleaved 1F1B methods introduced by Harlap et al. (2018) and Narayanan et al. (2021) with implementation from Megatron-LM. For interleaved 1F1B, the entire model is divided into a sequence of chunks, which are cyclically taken by each stage, forming an interleaved pipeline. In our interleaved experiments, we always use the maximum number of chunks to ensure least bubble, i.e. each transformer layer serves as a chunk.

Our experiments utilize up to 32 NVIDIA A100 SXM 80G GPUs distributed across 4 nodes interconnected by a RoCE RDMA network. The running time of each iteration is recorded after several warm-up iterations. Thanks to the reproducibility provided by Megatron-LM implementation, we can verify the correctness of ZB-1p and ZB-2p without running models until convergence. We use a fixed random seed to initialize the model, record the loss after every iteration for ZB-1p, ZB-2p, and 1F1B, and then verify that they're bit-to-bit identical.

# 5.2 MAIN RESULTS

![](images/62b6067bd07a32f2a56168c8e34db1d3ab5bee7289c65345d2ec61df52b0ef3e.jpg)

Figure 5: Comparison of throughput across different pipeline schedules.   
Table 4: Experiment result details 

<table><tr><td rowspan="3">Setup</td><td>Model</td><td colspan="3">1.5B</td><td colspan="3">6.2B</td><td colspan="3">14.6B</td><td colspan="3">28.3B</td></tr><tr><td>#GPU</td><td colspan="3">8</td><td colspan="3">8</td><td colspan="3">16</td><td colspan="3">32</td></tr><tr><td>#Microbatch</td><td>24</td><td>32</td><td>64</td><td>24</td><td>32</td><td>64</td><td>48</td><td>64</td><td>128</td><td>96</td><td>128</td><td>256</td></tr><tr><td rowspan="4">Samples per GPU per second</td><td>ZB-2p</td><td>14.5</td><td>14.8</td><td>14.9</td><td>4.32</td><td>4.35</td><td>4.39</td><td>1.81</td><td>1.83</td><td>1.85</td><td>0.99</td><td>1.00</td><td>1.00</td></tr><tr><td>ZB-1p</td><td>12.9</td><td>13.4</td><td>14.2</td><td>3.88</td><td>4.00</td><td>4.20</td><td>1.61</td><td>1.67</td><td>1.76</td><td>0.87</td><td>0.90</td><td>0.96</td></tr><tr><td>1F1B</td><td>11.8</td><td>12.5</td><td>13.6</td><td>3.50</td><td>3.70</td><td>4.03</td><td>1.40</td><td>1.49</td><td>1.64</td><td>0.76</td><td>0.80</td><td>0.88</td></tr><tr><td>1F1B-I</td><td>13.1</td><td>13.4</td><td>13.9</td><td>4.01</td><td>4.08</td><td>4.19</td><td>1.54</td><td>1.59</td><td>1.66</td><td>0.82</td><td>0.85</td><td>0.90</td></tr><tr><td rowspan="4">Memory (GB)</td><td>ZB-2p</td><td>59</td><td>59</td><td>59</td><td>70</td><td>70</td><td>70</td><td>51</td><td>51</td><td>51</td><td>74</td><td>74</td><td>74</td></tr><tr><td>ZB-1p</td><td>32</td><td>32</td><td>32</td><td>42</td><td>42</td><td>42</td><td>33</td><td>33</td><td>33</td><td>44</td><td>44</td><td>44</td></tr><tr><td>1F1B</td><td>30</td><td>30</td><td>30</td><td>39</td><td>39</td><td>39</td><td>32</td><td>32</td><td>32</td><td>43</td><td>43</td><td>43</td></tr><tr><td>1F1B-I</td><td>40</td><td>40</td><td>40</td><td>48</td><td>48</td><td>48</td><td>39</td><td>39</td><td>39</td><td>58</td><td>58</td><td>58</td></tr></table>

We present the throughput of all methods in Figure 5, and leave the additional details for each setup in Table 4. Our experiments demonstrate that ZB-2p consistently outperforms all other methods across various settings. Notably, the throughput of 1F1B, 1F1B-I and ZB-1p show a strong positive correlation with the number of microbatches. In contrast, ZB-2p maintains the efficiency even with fewer microbatches. This is because the bubble rate in ZB-2p has almost reached zero (Table 5), and its throughput is already close to the upper bound. Here the upper bound is roughly estimated by multiplying the throughput of 1F1B and $\frac{1}{1 - \text{bubble rate of } 1\text{F1B}}$ (for more details see Section 5.3). As mentioned before, the improved efficiency of ZB-2p comes at the cost of a higher memory consumption compared to the 1F1B baseline. We also compare ZB-2p with 1F1B under the same memory consumption in Appendix F, and the experimental results also show that ZB-2p achieves a higher throughput even with half microbatch size compared to 1F1B.

In contrast, ZB-1p is designed to have a peak memory cost similar to the 1F1B baseline. It shows a comparable throughput to 1F1B-I in the 8 GPUs setups. In multi-node setups where communication bandwidth is more of a bottleneck, ZB-1p clearly outperforms 1F1B-I, highlighting its advantage in reducing pipeline bubbles without incurring extra communication cost.

In most of our settings we set number of microbatches $m$ larger than number of stages $p$ because they're more common use cases of pipeline parallelism. However we conducted experiments listed in Appendix H for $m \leq p$ cases which shows $20\%$ to $30\%$ improvements with a similar memory consumption.

# 5.3 EFFICIENCY OF AUTOMATIC SCHEDULING

Table 5: Bubble rates of 1F1B, 1F1B-I, ZB-H1, ZB-H2, ZB-1p, ZB-2p under different settings. 

<table><tr><td>Model</td><td>#Stage (p)</td><td>#Microbatch (m)</td><td>1F1B</td><td>1F1B-I</td><td>ZB-H1</td><td>ZB-H2</td><td>ZB-1p</td><td>ZB-2p</td></tr><tr><td rowspan="3">1.5B</td><td rowspan="3">8</td><td>24</td><td>0.2431</td><td>0.1055</td><td>0.1585</td><td>0.1083</td><td>0.1585</td><td>0.0433</td></tr><tr><td>32</td><td>0.1985</td><td>0.0818</td><td>0.1242</td><td>0.0837</td><td>0.1242</td><td>0.0039</td></tr><tr><td>64</td><td>0.1240</td><td>0.0443</td><td>0.0674</td><td>0.0444</td><td>0.0674</td><td>0.0026</td></tr><tr><td rowspan="3">6.2B</td><td rowspan="3">8</td><td>24</td><td>0.2347</td><td>0.0808</td><td>0.1323</td><td>0.0698</td><td>0.1323</td><td>0.0029</td></tr><tr><td>32</td><td>0.1898</td><td>0.0628</td><td>0.1045</td><td>0.0559</td><td>0.1045</td><td>0.0022</td></tr><tr><td>64</td><td>0.1091</td><td>0.0320</td><td>0.0554</td><td>0.0294</td><td>0.0554</td><td>0.0010</td></tr><tr><td rowspan="3">14.6B</td><td rowspan="3">16</td><td>48</td><td>0.2552</td><td>0.1104</td><td>0.1397</td><td>0.0672</td><td>0.1397</td><td>0.0066</td></tr><tr><td>64</td><td>0.2082</td><td>0.0852</td><td>0.1088</td><td>0.0516</td><td>0.1088</td><td>0.0054</td></tr><tr><td>128</td><td>0.1251</td><td>0.0445</td><td>0.0576</td><td>0.0266</td><td>0.0576</td><td>0.0028</td></tr><tr><td rowspan="3">28.3B</td><td rowspan="3">32</td><td>96</td><td>0.2646</td><td>0.1493</td><td>0.1421</td><td>0.0641</td><td>0.1421</td><td>0.0038</td></tr><tr><td>128</td><td>0.2168</td><td>0.1164</td><td>0.1106</td><td>0.0490</td><td>0.1106</td><td>0.0029</td></tr><tr><td>256</td><td>0.1352</td><td>0.0624</td><td>0.0594</td><td>0.0257</td><td>0.0594</td><td>0.0018</td></tr></table>

![](images/6d54e26c21eb9591c35ca1d45639b2584f609870417289cf0f8412f3665cd225.jpg)  
Figure 6: A schedule produced by ZB-2p (top) and its profiled execution process (bottom).

We study the efficiency of the schedules generated from our automatic scheduling algorithm. The same setups as our main experiments are used, however, since our purpose is to study the efficiency of the automatic scheduling algorithm, the numbers here are based on theoretical calculations instead of real experiments. To quantify the efficiency of a pipeline schedule, we introduce the concept of bubble rate, which is calculated as $\left(\text{cost}-m(T_{F}+T_{B}+T_{W})\right)/\text{cost}$ . The cost here is defined as

the largest execution time of all stages, calculated for each schedule using profiled $T_{F}$ , $T_{B}$ , $T_{W}$ and $T_{comm}$ values. The $m(T_{F} + T_{B} + T_{W})$ is the optimal execution time when all communications are overlapped with computations and hence no bubbles in the pipeline.

The bubble rates for different schedules are presented in Table 5. We include the handcrafted schedules ZB-H1 and ZB-H2 as baselines to the automatically searched schedules. In most of the settings, ZB-2p produces a bubble rate of less than 1%, which is the best among all schedules. In contrast, ZB-H2 consistently performs worse than ZB-2p. This provides a strong evidence that our automatic scheduling algorithm adapts better to realistic scenarios by using more accurate estimates of $T_{F}$ , $T_{B}$ , $T_{W}$ and $T_{comm}$ . On the contrary, this improvement is not observed in ZB-1p vs ZB-H1, hypothetically because the memory limit becomes the dominate factor. Notably, all of our methods significantly outperform 1F1B.

We also plot ZB-2p and its profiled real execution on 16 GPUs to provide a direct visual evidence that it is truly a zero bubble schedule. As shown in Figure 6, the automatically generated ZB-2p schedule has almost no bubble. The profiled execution has slightly more bubbles but retains a good overall alignment.

![](images/a5e2728a355fb8f04e79bda8b5c7a32b9bbf1d5d02b2d40bc1f35c33eacdfa3e.jpg)

<details>
<summary>line</summary>

| Model Size | Microbatches | M_limit | Bubble Rate |
|------------|--------------|----------|-------------|
| 1.5B       | 24           | 1.0      | 0.15        |
| 1.5B       | 24           | 2.0      | 0.05        |
| 1.5B       | 24           | 3.0      | 0.05        |
| 1.5B       | 32           | 1.0      | 0.12        |
| 1.5B       | 32           | 2.0      | 0.05        |
| 1.5B       | 32           | 3.0      | 0.05        |
| 1.5B       | 64           | 1.0      | 0.07        |
| 1.5B       | 64           | 2.0      | 0.05        |
| 1.5B       | 64           | 3.0      | 0.05        |
| 6.2B       | 24           | 1.0      | 0.13        |
| 6.2B       | 24           | 2.0      | 0.05        |
| 6.2B       | 24           | 3.0      | 0.05        |
| 6.2B       | 32           | 1.0      | 0.11        |
| 6.2B       | 32           | 2.0      | 0.05        |
| 6.2B       | 32           | 3.0      | 0.05        |
| 6.2B       | 64           | 1.0      | 0.11        |
| 6.2B       | 64           | 2.0      | 0.05        |
| 6.2B       | 64           | 3.0      | 0.05        |
| 6.2B       | 96           | 1.0      | 0.14        |
| 6.2B       | 96           | 2.0      | 0.05        |
| 6.2B       | 96           | 3.0      | 0.05        |
| 14.6B      | 24           | 1.0      | 0.13        |
| 14.6B      | 24           | 2.0      | 0.05        |
| 14.6B      | 24           | 3.0      | 0.05        |
| 14.6B      | 32           | 1.0      | 0.11        |
| 14.6B      | 32           | 2.0      | 0.05        |
| 14.6B      | 32           | 3.0      | 0.05        |
| 14.6B      | 64           | 1.0      | 0.11        |
| 14.6B      | 64           | 2.0      | 0.05        |
| 14.6B      | 64           | 3.0      | 0.05        |
| 14.6B      | 96           | 1.0      | 0.14        |
| 14.6B      | 96           | 2.0      | 0.05        |
| 14.6B      | 96           | 3.0      | 0.05        |
| 28.3B      | 24           | -        | -           |
| 28.3B      | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -           |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            |-        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          | -            | -        | -             |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -          (-)   | -            | -        | -           |
| -3         (-)   (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (-)    (p=8)     |
| (P-value)   (P-value)   (P-value)   (P-value)   (P-value)   (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)   (P=8)     |
| (P-value)   (P-value)         (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)       (P=8)     |
| (P-value)   (P-value)         (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (p=8)     |
| (P-value)   (P-value)         (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (P-value)     (p=8)     |
| (P-value)   (P-value)         (P-value)     (P-value)     (P-value)     (N/A)|
</details>

Figure 7: The relation between memory limit and bubble rate using our heuristic algorithm.

# 5.4 MEMORY LIMIT

To better understand the effect of memory limit, we study the relationship of the bubble rate to $M_{limit}$ . We run our heuristic algorithm with a series of $M_{limit}$ and plot them in Figure 7. Initially, the bubble rate shows a close-to-linear decreasing trend as we increase the value of $M_{limit}$ . Theoretically, the curve should plateau around $\frac{(p-1)(T_{B}+2T_{\mathrm{comm}})+pT_{F}}{T_{F}}M_{B}$ . Empirically, we find $2pM_{B}$ a good threshold for achieving close to zero bubble rate when $T_{F} \approx T_{B}$ and $T_{comm}$ is relatively small. Beyond the inflection point, although a sufficiently large memory limit does result in a theoretically zero bubble rate, in general the cost outweighs the gain. For more details see Appendix B.

# 6 CONCLUSION AND DISCUSSION

In this work, we introduced a novel strategy to improve the efficiency of pipeline parallelism by splitting the activation gradient and parameter gradient in backward computation, and we design an automatic pipeline scheduling algorithm that can minimize the pipeline bubble rate under different memory budgets. The schedules produced by this algorithm consistently outperform 1F1B and even achieve close to zero bubble rate. Empirically, achieving zero bubble requires approximately twice the activation memory compared to 1F1B, which raises concerns about out of memory issues. According to Appendix F, we believe it is worth trading some memory for a zero bubble pipeline schedule in the training of large models. Strategies like ZeRO, tensor parallelism can be used to accommodate the increased memory need. Another advantage of zero bubble schedule is that it can achieve optimal efficiency with a smaller number of microbatches (typically 3p is enough), which means more microbatches can be partitioned over data parallelism dimension. This brings a better scalability for the training of large models.

# REFERENCES

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, et al. {TVM}: An automated {End-to-End} optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18), pp. 578–594, 2018.   
Shiqing Fan, Yi Rong, Chen Meng, Zongyan Cao, Siyu Wang, Zhen Zheng, Chuan Wu, Guoping Long, Jun Yang, Lixue Xia, et al. Dapple: A pipelined data parallel approach for training large models. In Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, pp. 431–445, 2021.   
John Forrest and Robin Lougee-Heimer. Cbc user guide. In Emerging theory, methods, and applications, pp. 257–277. INFORMS, 2005.   
Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.   
Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil Devanur, Greg Ganger, and Phil Gibbons. Pipedream: Fast and efficient pipeline parallel dnn training. arXiv preprint arXiv:1806.03377, 2018.   
Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Advances in neural information processing systems, 32, 2019.   
Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5, 2023.   
Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, and Oleksandr Zinenko. Mlir: A compiler infrastructure for the end of moore's law. arXiv preprint arXiv:2002.11054, 2020.   
Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. arXiv preprint arXiv:2006.15704, 2020.   
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017.   
Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–15, 2021.   
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In International conference on machine learning, pp. 1310–1318. Pmlr, 2013.   
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.   
Jared Roesch, Steven Lyubomirsky, Logan Weber, Josh Pollock, Marisa Kirisame, Tianqi Chen, and Zachary Tatlock. Relay: A new ir for machine learning frameworks. In Proceedings of the 2nd ACM SIGPLAN international workshop on machine learning and programming languages, pp. 58–68, 2018.

Amit Sabne. Xla : Compiling machine learning for peak performance, 2020.   
Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Bowen Yang, Jian Zhang, Jonathan Li, Christopher Ré, Christopher Aberger, and Christopher De Sa. Pipemare: Asynchronous pipeline parallel dnn training. Proceedings of Machine Learning and Systems, 3:269–296, 2021.   
Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P Xing, et al. Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pp. 559–578, 2022.

# A OVERLAP COMMUNICATION IN DATA PARALLELISM

When data parallelism is taken into consideration, an all-reduce communication is launched to collect gradients before optimizer step. Generally, such communication is poorly overlapped with computation pass, resulting in a latency especially when the communication bandwidth is limited. As in Figure 3, usually a number of W passes are scheduled at the tail of an iteration. For each W pass, it consists of several independent computations calculating gradients for different parameters. As in Figure 8, We can reorder all of these computations to cluster those calculating the gradients for the same parameter, thus achieving the optimal overlapping between computation and communication.

![](images/219701cf3e4d1350efb9e89deb9598bb96f63e57ef313945a47c229555083374.jpg)  
(a) The schedule grouped by W

![](images/7767e7b728c8ccd56eccf198a25e21a6f545d4e5959110443159a7f6fd597dc2.jpg)  
(b) The schedule grouped by parameter   
Figure 8: Comparison between the original schedule grouped by W with poor overlapping (top) and the reordered schedule grouped by parameters with optimal overlapping (bottom). The number i represents the computation belongs to i-th W, and different colors represent computations for different parameters.

# B THE MEMORY LIMIT FOR AUTOMATIC SCHEDULING ALGORITHM

The relation between memory limit and bubble rate is highly affected by the bubbles preceding the first B in the initial stage. For the first microbatch, the forward pass needs to go through from the initial stage to final stage, and the backward pass reverses this process until it eventually goes back to the initial stage. The total time for the first microbatch from start to complete takes at least $p(T_{F} + T_{B}) + 2(p - 1)T_{\mathrm{comm}}$ and it cannot be squeezed due to the dependency chains. We denote the number of F passes as $k(\geq 1)$ and the bubble size as $\beta(\geq 0)$ , preceding the first B pass in the initial stage. Then we have:

$$
M _ {\text { limit }} \geq k M _ {B} \tag {1}
$$

$$
\beta \geq p (T _ {F} + T _ {B}) + 2 (p - 1) T _ {\mathrm{comm}} - k T _ {F} - T _ {B} = (p - 1) (T _ {B} + 2 T _ {\mathrm{comm}}) + (p - k) T _ {F} \tag {2}
$$

The lower bound of $M_{limit}$ is in proportion to k (see Formula 1), and $\beta$ is inversely proportional to k (see Formula 2). When increasing k and keeping $k < \left\lfloor \frac{(p-1)(T_B + 2T_{\text{comm}}) + pT_F}{T_F} \right\rfloor$ , $\beta$ decreases linearly, meanwhile the lower bound of $M_{limit}$ increases linearly. When $k = \left\lfloor \frac{(p-1)(T_B + 2T_{\text{comm}}) + pT_F}{T_F} \right\rfloor$ , $\beta$ reaches its minimum value without delaying B and its value is less than $T_F$ , with a peak activation memory at least $\left\lfloor \frac{(p-1)(T_B + 2T_{\text{comm}}) + pT_F}{T_F} \right\rfloor M_B$ . Beyond this point, further reducing pipeline bubbles to zero is not easy. This is because there is a small bubble less than $T_F$ in each stage (see Figure 6), and scheduling another F will delay the starting time of B thus causing more requirements on F in previous stages. Theoretically, another p - 1 F passes are required in the initial stage to fully eliminate bubbles preceding the first B for all stages (see Figure 9), which also means a total activation memory usage at least $\left\lfloor \frac{(p-1)(T_B + 2T_{\text{comm}}) + (2p-1)T_F}{T_F} \right\rfloor M_B$ .

![](images/0f8278e600b60cdc5d9ba723761c4e29a6a352b0418e9ecb67619ac993a9df57.jpg)

<details>
<summary>bar_stacked</summary>

| Time | F    | B    | W    |
|------|------|------|------|
| Device 0 | 0    | 0    | 0    |
| Device 1 | 0    | 0    | 0    |
| Device 2 | 0    | 0    | 0    |
| Device 3 | 0    | 0    | 0    |
| Device 4 | 0    | 0    | 0    |
| Device 5 | 0    | 0    | 0    |
| Device 6 | 0    | 0    | 0    |
| Device 7 | 0    | 0    | 0    |
</details>

Figure 9: Zero bubble schedule for 1.5B model with 32 microbatches.

# C IN-PLACE OPTIMIZER ROLLBACK

When we need to rollback an optimizer step, a typical method is to store a historic version of parameters and optimizer states, and revert to this historic version when needed. However, this method is memory inefficient and lots of copy operations are needed, which definitely hurts the training performance. For most optimizers, we notice that the step function is arithmetically reversible. Under this observation, we propose a novel technique to perform in-place optimizer rollback, which avoids allocating extra memory and requires extra computations only when the rollback is performed. As in Algorithm 1, we show how to rollback the step function for AdamW optimizer (Loshchilov & Hutter, 2017).

Algorithm 1 In-place rollback for AdamW   
1: Optimizer States:
2: $\gamma(\mathrm{lr})$ , $\beta_{1}$ , $\beta_{2}(\mathrm{betas})$ , $\epsilon$ (epsilon), $\lambda(\mathrm{weight~decay})$ ,
3: m (first moment), v (second moment), $\theta$ (parameters),
4: t(time stamp).
5: function STEP(g) ▷ In-place step
6: $t = t + 1$ 7: $m = \beta_{1}m + (1 - \beta_{1})g$ 8: $v = \beta_{2}v + (1 - \beta_{2})g^{2}$ 9: $m' = m / (1 - \beta_{1}^{t})$ 10: $v' = v / (1 - \beta_{2}^{t})$ 11: $\theta = \theta - \gamma\lambda\theta - \gamma m' / (\sqrt{v'} + \epsilon)$ 12: end function
13: function ROLLBACK(g) ▷ In-place rollback
14: $m' = m / (1 - \beta_{1}^{t})$ 15: $v' = v / (1 - \beta_{2}^{t})$ 16: $\theta = (\theta + \gamma m' / (\sqrt{v'} + \epsilon)) / (1 - \gamma\lambda)$ 17: $m = (m - (1 - \beta_{1})g) / \beta_{1}$ 18: $v = (v - (1 - \beta_{2})g^{2}) / \beta_{2}$ 19: t = t - 1
20: end function

# D PROFILED TIME IN EXPERIMENTS

In our experiments, we record the profiled time of $T_{F}$ , $T_{B}$ , $T_{W}$ , and $T_{comm}$ in ZB-2p across different settings. These values are then used to calculate bubble rates for all the methods considered in Section 5.3 and 5.4. These values can be found in Table 6.

Table 6: Profiled time of $T_{F}$ , $T_{B}$ , $T_{W}$ , and $T_{comm}$ . 

<table><tr><td>Model</td><td>#Stage (p)</td><td>#Microbatch (m)</td><td> $T_F$ </td><td> $T_B$ </td><td> $T_W$ </td><td> $T_{comm}$ </td></tr><tr><td rowspan="3">1.5B</td><td rowspan="3">8</td><td>24</td><td>18.522</td><td>18.086</td><td>9.337</td><td>0.601</td></tr><tr><td>32</td><td>18.513</td><td>18.086</td><td>9.331</td><td>0.626</td></tr><tr><td>64</td><td>18.546</td><td>18.097</td><td>9.321</td><td>0.762</td></tr><tr><td rowspan="3">6.2B</td><td rowspan="3">8</td><td>24</td><td>29.718</td><td>29.444</td><td>19.927</td><td>0.527</td></tr><tr><td>32</td><td>29.802</td><td>29.428</td><td>19.530</td><td>0.577</td></tr><tr><td>64</td><td>29.935</td><td>29.621</td><td>19.388</td><td>0.535</td></tr><tr><td rowspan="3">14.6B</td><td rowspan="3">16</td><td>48</td><td>11.347</td><td>11.248</td><td>8.132</td><td>0.377</td></tr><tr><td>64</td><td>11.307</td><td>11.254</td><td>8.101</td><td>0.379</td></tr><tr><td>128</td><td>11.325</td><td>11.308</td><td>8.109</td><td>0.378</td></tr><tr><td rowspan="3">28.3B</td><td rowspan="3">32</td><td>96</td><td>10.419</td><td>10.207</td><td>7.715</td><td>0.408</td></tr><tr><td>128</td><td>10.408</td><td>10.204</td><td>7.703</td><td>0.408</td></tr><tr><td>256</td><td>10.402</td><td>10.248</td><td>7.698</td><td>0.460</td></tr></table>

# E ABLATION STUDY ON OPTIMIZER POST-VALIDATION STRATEGY

In this section, we provide an ablation study on the effectiveness of the optimizer post-validation strategy. The study compares the throughput of ZB-2p under two conditions: with post-validation and with all-reduce synchronization. According to the experimental results in Table 7, the synchronized version of ZB-2p demonstrates a performance decrease of approximately 8% compared to ZB-2p with optimizer post-validation.

Table 7: Throughput (Samples per GPU per second) comparison between ZB-2p and synchronized ZB-2p 

<table><tr><td>Model</td><td>#Stage (p)</td><td>#Microbatch (m)</td><td>Post-validation</td><td>All-reduce synchronization</td></tr><tr><td>1.5B</td><td>8</td><td>24</td><td>14.5</td><td>13.11</td></tr><tr><td>6.2B</td><td>8</td><td>24</td><td>4.32</td><td>4.00</td></tr><tr><td>14.6B</td><td>16</td><td>48</td><td>1.81</td><td>1.68</td></tr><tr><td>28.3B</td><td>32</td><td>96</td><td>0.99</td><td>0.91</td></tr></table>

# F COMPARE ZB-2P WITH 1F1B UNDER THE SAME MEMORY CONSUMPTION

Under the same memory consumption, we double the size of each microbatch for 1F1B and ZB-1p and compare their throughput with ZB-2p in Table 8. The experimental results show that ZB-2p also holds a better performance even with a half microbatch size compared to 1F1B. Empirically, a larger batch size increases the utilization rate of GPU and thus improves the efficiency. However, it is less of a concern for large models because the hidden dimension is large enough to saturate device utilization. Based on this consideration and our experimental results, we believe ZB-2p is more preferred than increasing the batch size for 1F1B. In some experiments where the device utilization is less saturated and m/p is relatively large, ZB-1p with a doubled microbatch size may slightly outperform than ZB-2p.

# G ILP FORMULATION

Any pass in the pipeline can be uniquely indexed by $(i,j,c)$ , where $i \in \{1,2,\ldots,p\}$ indexes the stage, $j \in \{1,2,\ldots,m\}$ indexes the microbatch, and $c \in \{F,B,W\}$ denotes the specific pass of the microbatch. We define the variable $T_{(i,j,c)}$ as the time cost and $E_{(i,j,c)}$ as the ending time of a pass. We introduce $\Delta M_{(i,j,c)}$ to denote the memory increment incurred by the pass $(i,j,c)$ . For example, $\Delta M_{(\cdot,\cdot,F)} = M_B$ because the forward pass leads to a net increase of $M_B$ of activation stored for the backward pass. $\Delta M_{(\cdot,\cdot,B)} = M_W - M_B$ which removes the memory stored for B while adding those required by W, and $\Delta M_{(\cdot,\cdot,W)} = -M_W$ . Finally, the variable that we want to search is the ordering of the passes in the schedule, for which we introduce the variable $O_{(i,j,c) \to (i,j',c')} \in \{0,1\}$ , which is an indicator whether the pass index by $(i,j,c)$ is scheduled before $(i,j',c')$ .

$$
\min _ {O, E} \quad \max _ {i} E _ {(i, m, W)} - E _ {(i, 1, F)} + T _ {(i, 1, F)} \tag {3}
$$

$$
s. t. \quad E _ {(i, j, F)} \geq E _ {(i - 1, j, F)} + T _ {\text { comm }} + T _ {(i, j, F)} \tag {4}
$$

$$
E _ {(i, j, B)} \geq E _ {(i + 1, j, B)} + T _ {\mathrm{comm}} + T _ {(i, j, B)} \tag {5}
$$

$$
E _ {(i, j, c)} \geq E _ {(i, j ^ {\prime}, c ^ {\prime})} + T _ {(i, j, c)} - O _ {(i, j, c) \rightarrow (i, j ^ {\prime}, c ^ {\prime})} \infty \tag {6}
$$

$$
M _ {\text { limit }} \geq \Delta M _ {(i, j ^ {\prime}, c ^ {\prime})} + \sum_ {j, c} \Delta M _ {(i, j, c)} O _ {(i, j, c) \rightarrow (i, j ^ {\prime}, c ^ {\prime})} \tag {7}
$$

Overall, the optimization target (3) is to minimize the time spent by the longest stage. Constraints (4) and (5) add the sequential dependency requirements on the F and B passes of the same microbatch in adjacent stages. Additionally, (6) adds the dependency constraint imposed by our decision of the scheduling order. Finally, (7) limits the peak activations memory to be below $M_{limit}$ .

Table 8: Comparison between 1F1B, ZB-1p and ZB-2p under the same memory consumption. 

<table><tr><td>Model</td><td>p</td><td>m</td><td>b</td><td>Samples per GPU per second</td><td>Memory(GB)</td><td>Schedule</td></tr><tr><td rowspan="9">1.5B</td><td rowspan="9">8</td><td rowspan="3">24</td><td>12</td><td>12.0</td><td>57</td><td>1F1B</td></tr><tr><td>12</td><td>13.0</td><td>61</td><td>ZB-1p</td></tr><tr><td>6</td><td>14.5</td><td>59</td><td>ZB-2p</td></tr><tr><td rowspan="3">32</td><td>12</td><td>12.6</td><td>57</td><td>1F1B</td></tr><tr><td>12</td><td>13.6</td><td>61</td><td>ZB-1p</td></tr><tr><td>6</td><td>14.8</td><td>59</td><td>ZB-2p</td></tr><tr><td rowspan="3">64</td><td>12</td><td>13.8</td><td>57</td><td>1F1B</td></tr><tr><td>12</td><td>14.4</td><td>61</td><td>ZB-1p</td></tr><tr><td>6</td><td>14.9</td><td>59</td><td>ZB-2p</td></tr><tr><td rowspan="9">6.2B</td><td rowspan="9">8</td><td rowspan="3">24</td><td>6</td><td>3.56</td><td>66</td><td>1F1B</td></tr><tr><td>6</td><td>3.95</td><td>71</td><td>ZB-1p</td></tr><tr><td>3</td><td>4.32</td><td>70</td><td>ZB-2p</td></tr><tr><td rowspan="3">32</td><td>6</td><td>3.76</td><td>66</td><td>1F1B</td></tr><tr><td>6</td><td>4.05</td><td>71</td><td>ZB-1p</td></tr><tr><td>3</td><td>4.35</td><td>70</td><td>ZB-2p</td></tr><tr><td rowspan="3">64</td><td>6</td><td>4.09</td><td>66</td><td>1F1B</td></tr><tr><td>6</td><td>4.24</td><td>71</td><td>ZB-1p</td></tr><tr><td>3</td><td>4.39</td><td>70</td><td>ZB-2p</td></tr><tr><td rowspan="9">14.6B</td><td rowspan="9">16</td><td rowspan="3">48</td><td>2</td><td>1.53</td><td>50</td><td>1F1B</td></tr><tr><td>2</td><td>1.73</td><td>51</td><td>ZB-1p</td></tr><tr><td>1</td><td>1.81</td><td>51</td><td>ZB-2p</td></tr><tr><td rowspan="3">32</td><td>2</td><td>1.62</td><td>50</td><td>1F1B</td></tr><tr><td>2</td><td>1.79</td><td>51</td><td>ZB-1p</td></tr><tr><td>1</td><td>1.83</td><td>51</td><td>ZB-2p</td></tr><tr><td rowspan="3">128</td><td>2</td><td>1.78</td><td>50</td><td>1F1B</td></tr><tr><td>2</td><td>1.89</td><td>51</td><td>ZB-1p</td></tr><tr><td>1</td><td>1.85</td><td>51</td><td>ZB-2p</td></tr><tr><td rowspan="9">28.3B</td><td rowspan="9">32</td><td rowspan="3">96</td><td>2</td><td>0.81</td><td>72</td><td>1F1B</td></tr><tr><td>2</td><td>0.93</td><td>74</td><td>ZB-1p</td></tr><tr><td>1</td><td>0.99</td><td>74</td><td>ZB-2p</td></tr><tr><td rowspan="3">32</td><td>2</td><td>0.85</td><td>72</td><td>1F1B</td></tr><tr><td>2</td><td>0.96</td><td>74</td><td>ZB-1p</td></tr><tr><td>1</td><td>1.00</td><td>74</td><td>ZB-2p</td></tr><tr><td rowspan="3">256</td><td>2</td><td>0.94</td><td>72</td><td>1F1B</td></tr><tr><td>2</td><td>1.02</td><td>74</td><td>ZB-1p</td></tr><tr><td>1</td><td>1.00</td><td>74</td><td>ZB-2p</td></tr></table>

# H COMPARE ZB METHODS WITH 1F1B ON SMALL NUMBER OF MICROBATCHES

By nature of PP when the number of microbatches $m$ is less than number of stages $p$ , there'll be a large bubble rate. However zerobubble methods can still boost performance under these rare settings by approximately 20% to 30%. In a rough analysis ignoring communication and assuming $m <= p$ and $T_W < T_B$ , an 1F1B iteration takes $(m + p - 1) * (T_F + T_B + T_W)$ to complete, while a ZB iteration takes $(m + p - 1) * (T_F + T_B) + T_W$ . The experiment result is shown in Table 9. Noticeably when $m <= p$ ZB-1p and ZB-2p are essentially the same and consumes similar memory as 1F1B.

Table 9: Comparison between 1F1B and ZB-2p on small number of microbatches. 

<table><tr><td>Model</td><td>p</td><td>m</td><td>b</td><td>Samples per GPU per second</td><td>Memory(GB)</td><td>Schedule</td></tr><tr><td rowspan="6">1.5B</td><td rowspan="6">8</td><td rowspan="2">2</td><td>6</td><td>3.56</td><td>11</td><td>1F1B</td></tr><tr><td>6</td><td>4.25</td><td>12</td><td>ZB-2p</td></tr><tr><td rowspan="2">4</td><td>6</td><td>5.74</td><td>18</td><td>1F1B</td></tr><tr><td>6</td><td>6.92</td><td>19</td><td>ZB-2p</td></tr><tr><td rowspan="2">8</td><td>6</td><td>8.26</td><td>29</td><td>1F1B</td></tr><tr><td>6</td><td>9.90</td><td>34</td><td>ZB-2p</td></tr><tr><td rowspan="6">6.2B</td><td rowspan="6">8</td><td rowspan="2">2</td><td>3</td><td>1.04</td><td>21</td><td>1F1B</td></tr><tr><td>3</td><td>1.33</td><td>21</td><td>ZB-2p</td></tr><tr><td rowspan="2">4</td><td>3</td><td>1.69</td><td>28</td><td>1F1B</td></tr><tr><td>3</td><td>2.16</td><td>29</td><td>ZB-2p</td></tr><tr><td rowspan="2">8</td><td>3</td><td>2.45</td><td>39</td><td>1F1B</td></tr><tr><td>3</td><td>3.07</td><td>44</td><td>ZB-2p</td></tr><tr><td rowspan="6">14.6B</td><td rowspan="6">16</td><td rowspan="2">4</td><td>1</td><td>0.39</td><td>19</td><td>1F1B</td></tr><tr><td>1</td><td>0.52</td><td>20</td><td>ZB-2p</td></tr><tr><td rowspan="2">8</td><td>1</td><td>0.65</td><td>24</td><td>1F1B</td></tr><tr><td>1</td><td>0.85</td><td>24</td><td>ZB-2p</td></tr><tr><td rowspan="2">16</td><td>1</td><td>0.95</td><td>32</td><td>1F1B</td></tr><tr><td>1</td><td>1.25</td><td>33</td><td>ZB-2p</td></tr></table>