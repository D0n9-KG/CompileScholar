## Related Work

**Processing Architecture Paradigm**
Prior research on accelerating de novo genome assembly has predominantly relied on general-purpose parallel hardware, specifically GPU-accelerated distributed systems [2, 5, 6, 7, 8, 9]. These approaches leverage the high computational throughput of GPUs to handle large-scale metagenomic and genomic data, yet they remain constrained by the memory wall inherent in general-purpose architectures. In contrast, a distinct line of work explores memory-centric computing paradigms, such as near-memory processing (NMP) and processing-in-memory (PIM), to mitigate data movement overheads [1, 3]. While storage-accelerated systems [4] and general-purpose CPU clusters [10] represent alternative architectural choices, they do not address the specific memory-bound nature of assembly algorithms as directly as NMP. NMP-PaK aligns with the NMP paradigm [1, 3] but distinguishes itself through channel-level integration, providing a more granular and efficient hardware substrate for assembly tasks compared to prior PIM platforms.

**Memory Hierarchy & Footprint Strategy**
The management of memory footprint is a critical challenge in de novo assembly, where state-of-the-art assemblers often exhibit large memory footprints due to complex data structures. Existing solutions typically address this by relying on large-capacity DRAM [2] or by offloading data to external storage [4], both of which introduce significant latency or resource constraints. Other approaches focus on standard cache-hierarchy optimizations within general-purpose CPU environments [10], which are insufficient for the irregular memory access patterns of assembly algorithms. A notable gap exists in the literature, as no cited prior work achieves a minimal memory footprint through the specific combination of NMP scratchpads and customized batch processing. NMP-PaK fills this gap by leveraging near-memory scratchpad space to drastically reduce the overall memory footprint, achieving a 14X reduction compared to state-of-the-art methods that rely on DRAM-centric or storage-offloading strategies.

**Handling of Data Irregularity**
De novo assembly involves complex, interdependent data structures that lead to irregular data patterns and hardware underutilization. Most prior work addresses this through pure software optimization on CPUs [2, 10], which limits the potential for hardware acceleration. A specific subset of research attempts to exploit algorithmic regularities, such as leveraging quasi-sequential characteristics to accelerate sequence-to-graph mapping [11]. However, no cited prior work employs a hybrid CPU-NMP processing strategy to manage irregular data patterns within a hardware-accelerated context. NMP-PaK introduces this hybrid approach, dynamically offloading irregular tasks to the CPU to prevent NMP hardware underutilization, thereby ensuring efficient processing of both regular and irregular data segments.

**System Integration Scope**
The integration of hardware and software optimizations is a key differentiator in high-performance genome assembly. The majority of prior work focuses on software-only algorithmic improvements, such as succinct de Bruijn graph representations [2] or optimized pairwise alignment algorithms [10], which are independent of the underlying hardware architecture. While some studies propose hardware platforms [1, 3, 4], they often treat software optimizations as separate layers rather than co-designing them with the hardware. No cited prior work adopts a holistic hardware-software co-design approach where software strategies like customized batch processing and hybrid scheduling are tightly coupled with the NMP hardware. NMP-PaK stands out by integrating these software optimizations directly into the NMP architecture, ensuring that the hardware capabilities are fully utilized to address the specific bottlenecks of de novo assembly.

## References

[1] Ultra efficient acceleration for de novo genome assembly via near-memory computing
[2] MEGAHIT: An ultra-fast single-node solution for large and complex
  metagenomics assembly via succinct de Bruijn graph
[3] Pim-assembler: A processing-in-memory platform for genome assembly
[4] Abakus: Accelerating k-mer Counting with Storage Technology
[5] Accelerating large scale de novo metagenome assembly using GPUs
[6] Gpu-accelerated large-scale genome assembly
[7] Gpu-euler: Sequence assembly using gpgpu
[8] GAGM: Genome assembly on GPU using mate pairs
[9] GRASShopPER—An algorithm for de novo assembly based on GPU alignments
[10] Minimap2: pairwise alignment for nucleotide sequences
[11] Harp: Leveraging Quasi-Sequential Characteristics to Accelerate Sequence-to-Graph Mapping of Long Reads