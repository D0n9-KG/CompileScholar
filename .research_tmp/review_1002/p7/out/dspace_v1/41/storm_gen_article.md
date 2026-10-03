## Related Work

**Optimization Signal Source**
Prior research on serverless cold-start mitigation has employed diverse signals to identify optimization opportunities, ranging from static code inspection to runtime workload analysis. A subset of approaches relies on static analysis to identify unreachable or optional code without executing the application; for instance, FaaSLight uses function-level call graphs to separate indispensable from optional code [17]. In contrast, other methods leverage runtime workload characteristics, such as invocation frequency and function-chains, to guide resource management policies [6, 8]. Specifically, work characterizing large-scale cloud workloads utilizes invocation frequency analysis to propose resource management strategies that reduce cold starts [8]. While these runtime-based approaches capture some dynamic behavior, they often focus on resource allocation rather than the specific library loading overhead. SLIMSTART distinguishes itself by employing profile-guided optimization through statistical sampling and call-path profiling, a mechanism that directly targets workload-dependent library usage patterns rather than relying solely on static reachability or general invocation metrics [8].

**Workload Dependency Handling**
The handling of workload variability presents a key distinction among existing solutions, with some methods offering adaptive strategies while others remain workload-agnostic. FaaSLight, for example, applies a one-size-fits-all optimization by statically separating code based on call graphs, which does not adapt to changing workload distributions [17]. Conversely, adaptive frameworks like Fifer utilize function-aware scaling and proactive container spawning to address underutilization and avoid cold starts dynamically [6]. Similarly, resource management policies derived from characterizing real-world workloads, such as those in Azure Functions, adapt to invocation patterns to mitigate cold starts [8]. SLIMSTART aligns with this adaptive paradigm by integrating into CI/CD pipelines to enable continuous monitoring and optimization. Unlike prior adaptive methods that focus on resource provisioning, SLIMSTART tailors its code transformations to evolving workloads, ensuring that library loading overhead is minimized in response to specific, changing usage patterns [6, 8].

**Optimization Target Scope**
Existing literature addresses cold-start latency through various optimization targets, including container image size, memory efficiency, and general resource scheduling. Some works focus on reducing container image size and download latency through layer-wise caching and sharing, as seen in RainbowCake [15], or by optimizing function compression and warmup location [3]. Others target memory efficiency via deduplication [2] or address cold-start and resource scheduling for dynamic workflows [4, 5]. A significant body of work specifically targets library loading and initialization latency, employing techniques such as opportunistic pre-loading [9], pre-baking function environments [12], provisioned concurrency [13], greedy-dual caching [14], function fusion [16], and initialization-less booting mechanisms [11]. While these approaches share the goal of reducing initialization overhead, they often operate at the infrastructure or container level. SLIMSTART specifically targets the library loading overhead within the application code itself, aiming to reduce initialization latency by eliminating the loading of rarely used libraries, a granularity distinct from broader container provisioning or image optimization strategies [9, 11, 12, 13, 14, 16, 17].

**Implementation Mechanism**
The mechanisms used to implement cold-start optimizations vary significantly, with most prior work relying on infrastructure-level adjustments or runtime interception rather than source code modification. Many solutions employ container configuration changes, such as pre-warming [9, 12, 13, 14], pre-baking [12], or caching [14, 15]. Other approaches utilize container sharing strategies [1], resource scheduling [4], adaptive container provisioning [5], or heterogeneity-aware warming [7]. Some works implement snapshot-based VM state restoration [10], initialization-less booting [11], function fusion [16], or code separation with on-demand loading [17]. Notably, no cited prior work employs automated code transformations as the primary mechanism for mitigation. SLIMSTART introduces a novel implementation mechanism by applying automated code transformations to the application source. This approach fundamentally alters the application's initialization behavior to remove inefficient library loads, distinguishing it from prior methods that rely on external infrastructure management or runtime interception [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17].

## References

[1] Help rather than recycle: Alleviating cold startup in serverless computing through $\{$Inter-Function$\}$ container sharing
[2] Memory deduplication for serverless computing with medes
[3] CodeCrunch: Improving Serverless Performance via Function Compression and Cost-Aware Warmup Location Optimization
[4] Sustainable serverless computing with cold-start optimization and automatic workflow resource scheduling
[5] Kraken: Adaptive container provisioning for deploying dynamic dags in serverless platforms
[6] Fifer: Tackling Underutilization in the Serverless Era
[7] Icebreaker: Warming serverless functions better with heterogeneity
[8] Serverless in the Wild: Characterizing and Optimizing the Serverless
  Workload at a Large Cloud Provider
[9] Pre-Warming is Not Enough: Accelerating Serverless Inference With Opportunistic Pre-Loading
[10] Faasnap: Faas made fast using snapshot-based vms
[11] Catalyzer: Sub-millisecond startup for serverless computing with initialization-less booting
[12] Prebaking functions to warm the serverless cold start
[13] Provisioned concurrency for lambda functions
[14] FaasCache: keeping serverless computing alive with greedy-dual caching
[15] RainbowCake: Mitigating Cold-starts in Serverless with Layer-wise Container Caching and Sharing
[16] Mitigating cold start problem in serverless computing with function fusion
[17] FaaSLight: General Application-Level Cold-Start Latency Optimization for
  Function-as-a-Service in Serverless Computing
[18] The GraalVM native image