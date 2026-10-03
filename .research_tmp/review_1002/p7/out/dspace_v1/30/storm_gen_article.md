## Related Work

**Algorithmic Complexity and Optimality**
The computational complexity of isolation testing has emerged as a critical bottleneck for scaling verification to large-scale workloads. Early work on strong isolation levels, such as serializability, established that checking these properties is NP-complete, leading to exponential-time algorithms in the worst case [8, 11]. Subsequent research extended this analysis to weak isolation levels and memory models, demonstrating that even bounded weak-memory testing remains NP-complete [16], and that verifying causal consistency for a single execution is NP-complete [15]. While recent studies have shown that certain weak isolation levels are testable in polynomial time, existing tools often suffer from high polynomial complexities, such as $O(n^3)$ or $O(n^4)$, which restricts their applicability to moderate-sized histories [6]. In contrast, AWDIT achieves provably optimal sub-cubic complexity, running in $O(n^{3/2})$ for Read Committed and Read Atomic, and $O(n \cdot k)$ for Causal Consistency. Furthermore, while prior work on C11 memory model variants has achieved nearly-linear-time complexity with fine-grained optimality results [17], AWDIT provides a matching conditional lower bound for its target database isolation levels, establishing that its performance is essentially optimal and cannot be significantly improved asymptotically.

**Target Isolation Levels**
Prior research has targeted a diverse range of consistency models, reflecting the heterogeneity of modern distributed systems. Some works focus on strong isolation levels, such as serializability, which are computationally expensive to verify [8, 11]. Others address specific intermediate levels like Snapshot Isolation, developing specialized checkers such as Viper [9] and PolySI [10]. A significant body of work, including our own, targets weak isolation levels prevalent in modern databases, such as Read Committed, Read Atomic, and Causal Consistency [1, 7, 15]. Additionally, research has explored generalized isolation level definitions [2, 3] and client-centric specifications [4], as well as frameworks for transactional consistency with atomic visibility [5]. While tools like Plume [7] and theoretical analyses [1, 15] address similar weak isolation levels, AWDIT distinguishes itself by providing a unified, highly efficient tester for these specific levels that outperforms existing state-of-the-art tools in both speed and scalability.

**Testing Methodology**
The standard paradigm for detecting isolation bugs is black-box history validation, which checks whether observed transaction histories adhere to a prescribed isolation level without requiring internal database access. This approach is employed by several prominent tools, including Plume [7], PolySI [10], and general complexity analyses of transactional consistency [6]. In contrast, other methodologies focus on stateless model checking or consistency checking of executions, particularly in the context of C11-style memory models [17]. While AWDIT adheres to the black-box history validation methodology, its contribution lies not in changing the fundamental testing strategy but in optimizing the underlying algorithms to achieve optimal complexity, thereby enabling the testing of realistic, large-scale workloads that are intractable for prior black-box tools.

## References

[1] Session Guarantees for Weakly Consistent Replicated Data
[2] A Critique of ANSI SQL Isolation Levels
[3] Generalized Isolation Level Definitions
[4] Seeing Is {{Believing}}: {{A Client-Centric Specification}} of {{Database Isolation}}
[5] A {{Framework}} for {{Transactional Consistency Models}} with {{Atomic Visibility}}
[6] On the Complexity of Checking Transactional Consistency
[7] Plume: Efficient and Complete Black-Box Checking of Weak Isolation
                  Levels
[8] Cobra: Making Transactional Key-Value Stores Verifiably Serializable
[9] Viper: {{A Fast Snapshot Isolation Checker}}
[10] Efficient Black-box Checking of Snapshot Isolation in Databases
[11] The Serializability of Concurrent Database Updates
[12] Memory-Model-Aware Testing: A Unified Complexity Analysis
[13] Testing {{Shared Memories}}
[14] Mathematizing C++ concurrency
[15] On Verifying Causal Consistency
[16] How Hard is Weak-Memory Testing?
[17] Optimal Reads-From Consistency Checking for C11-Style Memory Models