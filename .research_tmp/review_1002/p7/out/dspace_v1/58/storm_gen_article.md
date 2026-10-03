## Related Work

**Query Modality**
Existing table discovery methods primarily rely on either structural similarity or purely semantic inputs to retrieve relevant tables. A significant body of work focuses on structural similarity, utilizing query tables to identify unionable or joinable candidates through value-based indexing, semantic relationships, or contrastive learning of column encoders [5, 6, 7, 12, 13, 14, 15, 17]. In contrast, other approaches employ natural language queries for semantic search, leveraging graph-based representations or self-supervised learning to match free-text questions with table content [1, 3]. Some systems adopt a hybrid approach combining keywords with content-based structural matching [2], while others rely solely on lexical matching for ad hoc retrieval [16]. However, no prior work combines a query table with natural language conditions to refine search results, a hybrid modality that enables more precise filtering than either structural or semantic inputs alone.

**Condition Complexity**
Prior benchmarks and methods typically handle simple filtering criteria, lacking the capacity to process complex logical operations within natural language. Many structural discovery methods are limited to single-attribute filtering, where the query specifies a single column or value constraint to find matching tables [5, 6, 7, 12, 13, 14, 15]. Similarly, keyword-based and structural matching systems generally perform simple keyword matching, treating the query as a set of terms rather than a logical expression [2, 16, 17]. While natural language retrieval systems accept unstructured free-text questions, they do not explicitly structure or evaluate complex relational logic such as unions, joins, or fuzzy conditions [1, 3]. Our work introduces structured natural language conditions that explicitly incorporate union, join, and fuzzy logic, addressing a gap in prior benchmarks that typically lack such complex relational specifications.

**Evaluation Scope**
The evaluation of table discovery systems varies significantly in scope, ranging from comprehensive benchmarks to single-method case studies. Several recent works have established comprehensive benchmarks with relevance annotations to evaluate multiple state-of-the-art methods, providing rigorous baselines for table retrieval and understanding tasks [1, 2, 3, 5, 6, 7, 9, 17]. In contrast, a number of studies focus on single-method case studies, proposing specific algorithms for joinable table discovery, data augmentation, or table enrichment without extensive comparative benchmarking against a wide range of competitors [12, 13, 14, 15, 16]. Our work aligns with the comprehensive benchmark approach, providing a large-scale dataset with extensive relevance annotations to evaluate six state-of-the-art methods, thereby revealing performance gaps that single-method studies may overlook.

**User Interaction Model**
Most existing table discovery systems are designed for direct top-K retrieval, aiming to return a final ranked list of tables in a single step. This paradigm is prevalent across structural similarity methods, natural language retrieval systems, and keyword-based approaches, all of which seek to minimize the number of user interactions by providing immediate results [1, 2, 3, 5, 6, 7, 12, 13, 14, 15, 16, 17]. However, this direct retrieval model often leaves users with large result sets that require manual filtering to identify the most relevant tables. Our work positions table discovery as a refinement step, where users combine a query table with natural language conditions to refine large result sets, addressing the bottleneck of manual filtering that direct top-K retrieval systems do not explicitly target.

## References

[1] Retrieving Complex Tables with Multi-Granular Graph Representation
  Learning
[2] StruBERT: Structure-aware BERT for Table Search and Matching
[3] Solo: Data Discovery Using Natural Language Questions Via A
  Self-Supervised Approach
[4] Table union search on open data
[5] Dataset Discovery in Data Lakes
[6] SANTOS: Relationship-based Semantic Table Union Search
[7] Semantics-aware Dataset Discovery from Data Lakes with Contextualized
  Column-based Representation Learning
[8] Automatic {Table} {Union} {Search} with {Tabular} {Representation} {Learning}
[9] TURL: Table Understanding through Representation Learning
[10] {InfoGather}: entity augmentation and attribute discovery by holistic matching with web tables
[11] {JOSIE}: {Overlap} {Set} {Similarity} {Search} for {Finding} {Joinable} {Tables} in {Data} {Lakes}
[12] Efficient Joinable Table Discovery in Data Lakes: A High-Dimensional
  Similarity-Based Approach
[13] DeepJoin: Joinable Table Discovery with Pre-trained Language Models
[14] ARDA: Automatic Relational Data Augmentation for Machine Learning
[15] Table Enrichment System for Machine Learning
[16] Ad Hoc Table Retrieval using Semantic Similarity
[17] LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes
[18] Recovering semantics of tables on the web
[19] Data integration for the relational web