## Related Work

**Optimization Formulation**
Prior research on resource allocation for real-time systems has predominantly relied on Mixed-Integer Linear Programming (MILP) or heuristic search methods to address the complexity of task and resource co-allocation. While MILP formulations provide exact solutions, they often suffer from high computational costs, particularly as the number of tasks and resources scales. In contrast, this paper introduces a 0-1 Linear Program (LP) formulation for task-resource co-allocation. Unlike the cited prior works, which do not employ a 0-1 LP structure, our approach leverages the specific constraints of the problem to achieve higher computational efficiency. This distinction allows our method to find more optimal solutions within the same time limit compared to existing MILP-based approaches, addressing a key gap in the efficiency of exact optimization methods for this domain.

**Heuristic Search Strategy**
To manage the combinatorial explosion inherent in multi-resource allocation, several studies have employed heuristic or hybrid search strategies. For instance, [22] proposes a hybrid multi-layer design space exploration technique to co-optimize cache partitioning and multi-core task scheduling, exploiting cache sensitivity to guide the search. However, this approach does not incorporate multi-objective Pareto pruning or dynamic programming for the inner optimization layer. In contrast, this paper presents a novel multi-layer framework where an outer layer performs Pareto-pruned search to explore resource allocations, while an inner layer optimizes task allocation by solving a knapsack problem using dynamic programming. This specific combination of Pareto-pruned outer search and DP-based inner knapsack solving is unique among the cited works, enabling superior performance in schedulability and resource usage compared to state-of-the-art co-allocation algorithms.

**Resource Co-allocation Scope**
The scope of resource co-allocation in prior work varies significantly, with many studies focusing on isolated resource types. A substantial body of literature addresses cache partitioning only, including works on real-time cache management [4], the effectiveness of cache partitioning in hard real-time systems [13, 14], its impact on embedded systems [15], minimizing cache usage [16], and tradeoffs in mixed-criticality systems [19]. Similarly, other studies focus exclusively on CPU core allocation [1, 2, 3] or memory bandwidth regulation [7, 8, 17]. Some approaches address hardware management and mixed-criticality provisioning [5], or specific hardware resource partitioning mechanisms such as Intel RDT [9, 11] and Arm MPAM [12]. While [10] proposes a holistic resource allocation approach and [18] coordinates last-level cache and memory bandwidth partitioning, the latter targets fairness in workload consolidation on commodity servers rather than real-time schedulability guarantees. This paper distinguishes itself by performing the joint optimization of memory bandwidth and cache partitioning specifically for multicore real-time systems, a scope shared only by [18] but applied to a different objective and system context.

**Evaluation Platform & Mechanism**
Experimental validation in prior work has largely relied on general-purpose server hardware or specific hardware partitioning features. Studies such as [9, 11] evaluate Intel Resource Director Technology (RDT), while [12] utilizes the Arm MPAM specification. Other works employ different hypervisors like Xen without specific bandwidth regulation mechanisms [6], or focus on GPU-based SoC platforms [17]. The work by [18] evaluates its coordination scheme on general-purpose x86 servers. In contrast, this paper evaluates its proposed optimization algorithm on an embedded AMD UltraScale+ ZCU102 platform. This setup is distinguished by the use of the Jailhouse hypervisor to enable fine-grained resource partitioning, specifically leveraging cache set partitioning and MemGuard for memory bandwidth regulation. No cited prior work utilizes this specific embedded platform combined with the Jailhouse hypervisor and MemGuard mechanism, providing a unique real-world validation context for the proposed methods.

## References

[1] Partitioned Scheduling of Recurrent Real-Time Tasks
[2] New strategies for assigning real-time tasks to multiprocessor systems
[3] The partitioned multiprocessor scheduling of sporadic task systems
[4] Real-time cache management for multi-core virtualization
[5] Attacking the One-Out-Of-m Multicore Problem by Combining Hardware Management with Mixed-Criticality Provisioning
[6] {Xilinx Xen Support with Cache-Coloring}
[7] {MemGuard}: Memory bandwidth reservation system for efficient performance isolation in multi-core platforms
[8] {MemPol}: Policing Core Memory Bandwidth from Outside of the Cores
[9] {Resource Director Technology}
[10] Holistic resource allocation for multicore real-time systems
[11] {A Closer Look at Intel Resource Director Technology (RDT)}
[12] {Arm Memory System Resource Partitioning and Monitoring (MPAM) System Component Specification}
[13] OUTSTANDING PAPER: Evaluation of Cache Partitioning for Hard Real-Time Systems
[14] On the effectiveness of cache partitioning in hard real-time systems
[15] Impact of cache partitioning on multi-tasking real time embedded systems
[16] Minimizing cache usage for real-time systems
[17] Dynamic memory bandwidth allocation for real-time GPU-based SoC platforms
[18] Copart: Coordinated partitioning of last-level cache and memory bandwidth for fairness-aware workload consolidation on commodity servers
[19] Cache sharing and isolation tradeoffs in multicore mixed-criticality systems
[20] {PDPA}: Period Driven Task and Cache Partitioning Algorithm for Multi-Core Systems
[21] {$IA^3$}: An Interference Aware Allocation Algorithm for Multicore Hard Real-Time Systems
[22] Co-Optimizing Cache Partitioning and Multi-Core Task Scheduling: Exploit
  Cache Sensitivity or Not?
[23] Holistic multi-resource allocation for multicore real-time virtualization
[24] Holistic Resource Allocation Under Federated Scheduling for Parallel Real-time Tasks
[25] {DNA}: Dynamic resource allocation for soft real-time multicore systems