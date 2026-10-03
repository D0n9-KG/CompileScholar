## Related Work

**Emotional Granularity in Opinion Mining**
Prior research in opinion mining has predominantly focused on sentiment polarity or aspect-based analysis rather than fine-grained emotional states. For instance, [2] develops Appsent, a tool that extracts and visualizes user perceptions through aspect-based sentiment analysis (ABSA), while [1] presents a multilabel multiclass dataset that captures both sentiment and emotion from Indonesian mobile application reviews. In contrast, no cited prior work adopts the specific fine-grained emotion classification framework based on Plutchik’s taxonomy that this paper employs. By moving beyond general sentiment polarity and standard multilabel approaches, this study addresses the underexplored gap in capturing the nuanced complexity of emotions in app reviews.

**Domain Context and Application**
The application of emotion and sentiment analysis varies significantly across different domains, with mobile app reviews representing a distinct context characterized by specific linguistic constraints. Several studies have focused on this domain, including [1], which constructs a dataset from Indonesian app reviews, [2], which analyzes user insights from app reviews, and [3], which conducts an empirical analysis of user experience in mobile learning applications. Other works apply similar analytical techniques to different fields, such as political text [4], materials science [5], financial data [6], and pragmatics and discourse analysis [7]. While this paper shares the mobile app review domain with [1, 2, 3], it distinguishes itself by specifically adapting Plutchik’s emotion taxonomy to the unique contextual constraints of this domain, rather than applying general-purpose sentiment models.

**Primary Contribution Type**
The primary contribution of existing literature in this area often centers on tool development or specific annotation methodologies rather than the creation of standardized resources. For example, [2] focuses on tool development by creating Appsent, while [8] introduces MEGAanno+, a system for collaborative annotation, and [9] proposes an annotation methodology/framework for verifying LLM labels. Similarly, [1] contributes to resource creation by providing a multilabel multiclass sentiment and emotion dataset. This paper aligns with [1] in its primary contribution type, focusing on resource creation through the development of a structured annotation framework, clear guidelines, and a dedicated dataset. Unlike works that prioritize tooling [2, 8] or methodological verification [9], this study aims to establish foundational resources to address the lack of standardized fine-grained emotion data.

**Automation Strategy**
Recent advancements in natural language processing have led to the exploration of Large Language Models (LLMs) as tools for annotation, with a growing body of work assessing their feasibility and efficiency. Studies such as [4] investigate LLMs as substitutes for human experts in political text, [5] proposes a semi-automated approach using Gemini Pro for materials science, [6] studies the effectiveness of LLMs for financial data, and [7] evaluates LLM-assisted annotation for pragmatics. Additionally, [8] introduces a human-LLM collaborative system, and [9] focuses on verifying LLM-generated labels to improve quality. This paper shares the automation strategy of assessing the feasibility of LLMs for annotation in a human-in-the-loop manner with [4, 5, 6, 7, 8]. By evaluating the cost-effectiveness and agreement of LLMs with human annotators specifically for fine-grained emotion extraction, this work contributes to the understanding of where full automation remains challenging due to the complexity of emotional interpretation.

## References

[1] Multilabel multiclass sentiment and emotion dataset from Indonesian mobile application review
[2] Appsent A Tool That Analyzes App Reviews
[3] An empirical analysis of mobile learning app usage experience
[4] Large language models as a substitute for human experts in annotating political text
[5] {Annotating Materials Science Text: A Semi-automated Approach for Crafting Outputs with Gemini Pro}
[6] Large Language Models as Financial Data Annotators: A Study on Effectiveness and Efficiency
[7] Assessing the potential of LLM-assisted annotation for corpus-based
  pragmatics and discourse analysis: The case of apology
[8] {MEGA}nno+: A Human-{LLM} Collaborative Annotation System
[9] Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels