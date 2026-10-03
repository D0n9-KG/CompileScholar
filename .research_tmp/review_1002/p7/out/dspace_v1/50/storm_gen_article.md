## Related Work

**Data Representation**
Prior research has increasingly adopted structured representations to facilitate complex queries and relational analyses in software and machine learning ecosystems. Specifically, [1, 2] have constructed knowledge graphs to model relationships within the machine learning domain, leveraging these structures to enhance recommendation capabilities. In contrast, other approaches have focused on different graph topologies or platforms; for instance, [3] developed a cross-platform graph linking academic papers to code repositories, while [4] utilized graph-based methods specifically for GitHub repositories. While these works demonstrate the utility of graph structures, they do not construct a unified knowledge graph specifically for the Hugging Face ecosystem. This paper distinguishes itself by introducing HuggingKG, a structured knowledge graph that captures domain-specific relations and rich textual attributes unique to open-source ML resources.

**Task Scope**
Existing benchmarks and systems in this area are predominantly limited to single-objective tasks. A significant portion of prior work focuses exclusively on recommendation, as seen in [1, 2, 3], which aim to match users or entities with relevant resources. Similarly, [4] addresses a single classification task, specifically tag assignment for repositories. There is a notable gap in the literature regarding comprehensive benchmarks that integrate multiple distinct information retrieval objectives. Unlike these single-task approaches, this paper presents HuggingBench, a multi-task benchmark that simultaneously evaluates recommendation, classification, and a novel tracing task for model evolution, thereby providing a more holistic assessment of IR capabilities in this domain.

**Domain Specificity**
The domain of application varies significantly across prior studies, with most focusing on general software engineering or broader ML contexts rather than the specific infrastructure of open-source model sharing. For example, [1] targets general machine learning method recommendation, while [2] focuses on the recommendation of ML/DL libraries. Other works address cross-platform scenarios, such as [3], which bridges academic papers and code repositories, or [4], which concentrates on GitHub repository tagging. No cited prior work specifically targets the management and analysis of open-source ML resources within the Hugging Face ecosystem. This paper fills this gap by explicitly scoping its contributions to the unique relations and dependencies found in the Hugging Face community, such as model-to-dataset links.

**Scale and Source**
The data sources and scales of previous studies are often constrained by the availability of specific platform data or the complexity of cross-platform alignment. For instance, [3] relies on real-world data from GitHub and academic sources, but does not capture the massive scale of a single, unified community platform. In contrast, this paper leverages the extensive, community-constructed data from Hugging Face, resulting in a large-scale dataset with 2.6 million nodes. This scale ensures that the benchmark reflects actual usage patterns and the true magnitude of the open-source ML resource landscape, a characteristic not present in the cited prior works.

## References

[1] {DEKR:} Description Enhanced Knowledge Graph for Machine Learning
                  Method Recommendation
[2] Task-Oriented {ML/DL} Library Recommendation Based on a Knowledge
                  Graph
[3] paper2repo: GitHub Repository Recommendation for Academic Papers
[4] {GRETA:} Graph-Based Tag Assignment for GitHub Repositories