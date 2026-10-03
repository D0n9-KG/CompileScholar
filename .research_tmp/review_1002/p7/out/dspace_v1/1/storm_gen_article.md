## Related Work

**Workload Characterization and Sources**
Prior research on LLM serving has predominantly relied on synthetic benchmarks or specific application scenarios rather than comprehensive production data. Several studies utilize standard academic benchmarks or representative models to evaluate system performance, such as [1] and [10], while others focus on specific multi-tenant scenarios with shared system prompts [2] or prototype implementations for document-based QA and recommendations [4]. Other works target specialized workloads, including retrieval-augmented generation (RAG) [6] and streaming applications like multi-round dialogue [8]. In contrast, this paper presents the first systematic characterization of KV$ workload patterns derived from real-world production traces at a large cloud provider. While [24] also characterizes production workloads, it focuses on serverless FaaS resource management rather than LLM-specific KV cache behavior, leaving a gap in understanding the unique reuse patterns of LLM serving in production environments.

**Analysis Scope: Cache Behavior vs. Inference Optimization**
The majority of prior work focuses on optimizing inference algorithms or general serving throughput without a deep empirical analysis of cache reuse dynamics. Techniques such as quantization [12], dynamic KV cache compression [11], and speculative prefetching [10] aim to reduce memory footprint or latency, while others like [2, 4, 6, 8, 9] propose architectural or algorithmic modifications to improve efficiency. General LLM serving frameworks [1] and multi-turn conversation optimizations [23] also prioritize throughput or cost-efficiency over cache-specific characterization. Conversely, this paper’s scope is a systematic characterization of KV$ reuse patterns, including reuse skew, time, and probability, as well as the determination of ideal cache size requirements. This empirical focus distinguishes our work from algorithmic innovations, providing the foundational insights necessary for workload-aware system design.

**Eviction Policy Design**
Existing eviction policies for KV caches and general buffer management typically rely on generic heuristics or model-agnostic metrics rather than workload-specific patterns. Classic algorithms such as LRU [13], 2Q [14], and adaptive variants like EELRU [15] and LRFU [16] are widely used but do not account for LLM-specific semantics. More recent approaches include attention-sink based windowing [8], heavy hitter retention based on attention scores [9], and layer-wise dynamic allocation [11]. In the broader cache replacement literature, policies such as LIRS [17], ARC [18], CAR [19], CLOCK-Pro [20], LHD [21], and learning-based methods like CacheUS [22] have been proposed to optimize hit rates. However, none of these prior works derive their eviction strategies from observed real-world LLM request patterns. This paper introduces a workload-aware eviction policy that leverages the predictable reuse characteristics of specific request categories, addressing the suboptimality of generic policies in production LLM serving.

**Request Category Granularity**
Most prior studies treat LLM traffic as a homogeneous stream or focus exclusively on specific interaction types, failing to capture the diversity of real-world request categories. Works such as [1, 4, 11, 12] analyze performance without distinguishing between different request structures, while others focus narrowly on multi-turn conversations [3, 8, 10, 23] or specific input formats like shared system prompts [2] and RAG text chunks [6]. This lack of granularity obscures the fact that reuse patterns are diverse overall but predictable within specific categories. In contrast, this paper distinguishes between single-turn and multi-turn requests, demonstrating that reuses between single-turn requests are equally important as those in multi-turn contexts. By analyzing category-specific predictability, we challenge the assumption of a monolithic workload and provide insights that enable more precise cache management strategies.

## References

[1] Efficient Memory Management for Large Language Model Serving with
  PagedAttention
[2] ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and
  Two-Phase Partition
[3] Cost-Efficient Large Language Model Serving for Multi-turn Conversations
               with CachedAttention
[4] Prompt Cache: Modular Attention Reuse for Low-Latency Inference
[5] Efficiently Programming Large Language Models using SGLang
[6] CacheBlend: Fast Large Language Model Serving for RAG with Cached
  Knowledge Fusion
[7] EPIC: Efficient Position-Independent Context Caching for Serving Large Language Models
[8] Efficient Streaming Language Models with Attention Sinks
[9] {H2O:} Heavy-Hitter Oracle for Efficient Generative Inference of Large
               Language Models
[10] InfiniGen: Efficient Generative Inference of Large Language Models with
  Dynamic KV Cache Management
[11] PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information
  Funneling
[12] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
  Quantization
[13] The {LRU-K} Page Replacement Algorithm For Database Disk Buffering
[14] 2Q: {A} Low Overhead High Performance Buffer Management Replacement
                  Algorithm
[15] {EELRU:} Simple and Effective Adaptive Page Replacement
[16] {LRFU:} {A} Spectrum of Policies that Subsumes the Least Recently
                  Used and Least Frequently Used Policies
[17] {LIRS:} an efficient low inter-reference recency set replacement policy
               to improve buffer cache performance
[18] {ARC:} {A} Self-Tuning, Low Overhead Replacement Cache
[19] {CAR:} Clock with Adaptive Replacement
[20] CLOCK-Pro: An Effective Improvement of the {CLOCK} Replacement
[21] {LHD:} Improving Cache Hit Rate by Maximizing Hit Density
[22] Learning Cache Replacement with {CACHEUS}
[23] {Cost-Efficient} Large Language Model Serving for Multi-turn Conversations with {CachedAttention}
[24] Serverless in the Wild: Characterizing and Optimizing the Serverless
  Workload at a Large Cloud Provider