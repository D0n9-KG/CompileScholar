## Related Work

### Knowledge Representation & Retrieval Strategy
A significant portion of prior work in claim verification relies on direct retrieval of unstructured text snippets from corpora such as Wikipedia to support fact-checking processes [1, 3, 4]. In contrast, other approaches have explored structured or programmatic methods to bridge the gap between natural language claims and evidence; for instance, some methods employ program-guided decomposition of claims into sub-tasks [7], while others utilize First-Order-Logic (FOL)-guided reasoning over knowledge-grounded question-and-answer pairs [8]. Additionally, several studies focus on leveraging Knowledge Graphs (KGs) directly, either by retrieving raw KG triples without semantic mapping [9] or by employing planning-retrieval-reasoning frameworks that generate relation paths [12]. However, no prior cited work adopts the specific strategy of using a specialized LLM to generate pseudo-subgraphs that guide the retrieval of structured KG subgraphs, a mechanism that ClaimPKG introduces to more effectively leverage KG structure for verification.

### LLM Architecture & Specialization
Most existing frameworks for claim verification and reasoning over structured data rely on a single general-purpose LLM to handle all stages of the pipeline, including decomposition, retrieval, and final reasoning [7, 8, 10, 11, 12]. This monolithic approach often struggles with the complexity of multi-step modular pipelines and specific KG-structuring tasks. In contrast, ClaimPKG proposes a hybrid architecture that offloads the specific task of representing claims as pseudo-subgraphs to a lightweight, specialized LLM, while reserving the final verdict and justification to a general-purpose LLM. This division of labor addresses the limitations of single-model pipelines by leveraging the efficiency of specialized models for structural mapping and the broad reasoning capabilities of general models for final inference.

### Input Data Modality
The choice of evidence source significantly impacts verification performance. A dominant line of research utilizes unstructured text corpora, such as Wikipedia, as the primary evidence source for fact verification [1, 3, 4]. Other methods have explored different modalities, including pre-computed fact tables [6], knowledge-grounded question-and-answer pairs [8], and general structured data [11]. More recently, several works have shifted toward using Structured Knowledge Graphs (KGs) as the primary evidence source to provide richer semantic representations [9, 10, 12]. ClaimPKG aligns with this latter group by utilizing KGs as the primary evidence source, arguing that their structured nature is better suited for complex reasoning than unstructured text, while addressing the specific challenge of mapping unstructured claims to these structured representations.

### Generalization Scope
The applicability of verification frameworks is often constrained by the domain and structure of the target data. Many existing methods that leverage KGs are limited to single-domain KG verification, where the structure of the knowledge base is tightly coupled with the verification task [9, 12]. In contrast, ClaimPKG demonstrates zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS. By effectively combining structured knowledge from KGs with LLM reasoning, the proposed framework maintains high performance even when the target dataset lacks explicit KG structure, a capability not observed in the cited prior works that remain confined to specific KG domains.

## References

[1] FEVER: a large-scale dataset for Fact Extraction and VERification
[2] {F}a{VIQ}: {FA}ct Verification from Information-seeking Questions
[3] Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence
[4] HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification
[5] Graph Neural Networks: A Review of Methods and Applications
[6] TAPAS: Weakly Supervised Table Parsing via Pre-training
[7] Fact-Checking Complex Claims with Program-Guided Reasoning
[8] Explainable Claim Verification via Knowledge-Grounded Reasoning with
  Large Language Models
[9] FactKG: Fact Verification via Reasoning on Knowledge Graphs
[10] {KG-GPT:} {A} General Framework for Reasoning on Knowledge Graphs
                  Using Large Language Models
[11] {S}truct{GPT}: A General Framework for Large Language Model to Reason over Structured Data
[12] Reasoning on Graphs: Faithful and Interpretable Large Language Model
  Reasoning