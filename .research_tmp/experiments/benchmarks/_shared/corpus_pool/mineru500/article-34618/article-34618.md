# CYBERPAL.AI: EMPOWERING LLMs WITH EXPERT-DRIVEN CYBERSECURITY INSTRUCTIONS

Matan Levi $^{1,2}$ , Yair Alluouche $^{1}$ , Daniel Ohayon $^{1}$ , Anton Puzanov $^{1}$

$^{1}$ IBM Research, $^{2}$ Ben-Gurion University

{matanle,yair,daniel.ohayon,antonp}@il.ibm.com

# ABSTRACT

Large Language Models (LLMs) have significantly advanced natural language processing (NLP), providing versatile capabilities across various applications. However, their application to complex, domain-specific tasks, such as cybersecurity, often faces substantial challenges. In this study, we introduce SecKnowledge and CyberPal.AI to address these challenges and train security-expert LLMs. SecKnowledge is a domain-knowledge-driven cyber-security instruction dataset, meticulously designed using years of accumulated expert knowledge in the domain through a multi-phase generation process. CyberPal.AI refers to a family of LLMs fine-tuned using SecKnowledge, aimed at building security-specialized LLMs capable of answering and following complex security-related instructions. Additionally, we introduce SecKnowledge-Eval, a comprehensive and diverse cyber-security evaluation benchmark, composed of an extensive set of cyber-security tasks we specifically developed to assess LLMs in the field of cyber-security, along with other publicly available security benchmarks. Our results show a significant average improvement of up to 24% over the baseline models, underscoring the benefits of our expert-driven instruction dataset generation process. These findings contribute to the advancement of AI-based cyber-security applications, paving the way for security-expert LLMs that can enhance threat-hunting and investigation processes.

# 1 INTRODUCTION

The rapid progress of LLMs offers a wide range of new capabilities that would have been considered unrealistic only a few years ago. LLMs have emerged as disruptive technology in domains ranging from healthcare to finance, changing the way we consume information and perform our daily tasks. As LLMs are trained on trillions of tokens, they should have fundamental knowledge of most domains available online.

One such domain is cyber-security. Yet, cyber-security is also a very complex domain. It requires deep understanding in multiple areas of expertise, such as operating systems, network and communication protocols, malware analysis, threat management, and many others. Furthermore, as cyber-security practice spans from security at the physical layer to security at the application layer, navigating this diverse landscape requires a comprehensive understanding and the ability to connect disparate elements effectively. Therefore, traditional data generation methods will not be effective (Mitra et al., 2023). As cyber-security is complex and highly domain-expert-driven, it is required to present LLMs with domain-specific data generated from expert knowledge to unlock and harness the potential of LLMs in the field.

Over the past decades, security experts have invested considerable time and resources into monitoring cyber activities, investigating incidents, and producing high-quality reports and comprehensive knowledge bases, which include detection rules for identifying and mitigating threats, among other crucial activities. This study seeks to utilize this extensive domain knowledge and, by integrating it with the robust capabilities of LLMs, create a highly valuable instruction-tuning dataset. By combining these efforts, we aim to unlock the potential of LLMs in cyber-security.

To achieve this, we present a domain-knowledge-driven instruction dataset, dubbed SecKnowledge. The objectives of SecKnowledge are twofold: firstly, to teach LLMs to follow both simple and

complex security instruction better; secondly, to enhance the comprehension of LLMs regarding the wide landscape of cyber-security and the intricate relationships between various security concepts.

Subsequently, we present CyberPal.AI, a family of Generative LLMs, fine-tuned from open-source state-of-the-art LLMs, using the SecKnowledge dataset. We demonstrate how CyberPal.AI outperforms baseline LLMs on various tasks in the cyber-security domain.

Lastly, we construct SecKnowledge-Eval, a diverse and comprehensive set of evaluation datasets specifically designed to test LLMs' ability to develop a holistic understanding of the cybersecurity domain and effectively navigate complex security concepts, such as threat investigation. SecKnowledge-Eval consists of evaluation datasets we developed and were designed to assess LLMs on complex cyber-security tasks such as relationships between different security aspects (e.g., the relationship between attack pattern and a specific attack technique), alongside other well-known public cyber-security benchmarks that primarily test general knowledge in the field.

Overall, we make the following contributions:

- We construct SecKnowledge, an instruction tuning dataset generated using an expert-driven process on a wide range of security-related datasets. The dataset construction involves two main steps. In the first step, we create instructions based on predefined schemas established through domain expertise. These schemas define templates that are filled with domain-expert knowledge and supplemented with LLM-generated content when necessary. In the second step, we expand the initial dataset through a hybrid synthetic content-based data generation process.   
- We train CyberPal.AI, a family of cyber-security expert LLMs, capable of understanding complex security concepts. CyberPal.AI demonstrates the advantages of enhancing LLMs with our domain-knowledge instruction dataset, SecKnowledge.   
- We developed SecKnowledge-Eval, a suite of evaluation datasets specifically designed to assess LLMs in the cyber-security domain. SecKnowledge-Eval consists of evaluation datasets we constructed to assess LLMs' capabilities on complex cyber-security tasks, alongside public benchmarks, intending to generate a comprehensive and diverse evaluation dataset for assessing both knowledge and understanding of models in the field of cyber-security. CyberPal.AI demonstrated superior performance over its baseline models, showing a substantial average improvement of up to 24% in training-aligned tasks and up to 10% in public cyber-security benchmarks.

# 2 RELATED WORK

# 2.1 GENERAL DOMAINS INSTRUCTION-TUNING

Instruction tuning (Wei et al., 2021; Longpre et al., 2023; Raffel et al., 2020; Xu et al., 2022; Sanh et al., 2021; Chung et al., 2022) demonstrates how fine-tuning Language Models (LMs) with NLP instructions enhances base models' performance in following instructions. In (Ouyang et al., 2022) OpenAI trained InstructGPT from GPT-3 (Brown et al., 2020). Vicuna (Chiang et al., 2023) used 70k conversations with ChatGPT collected from users via ShareGPT, and used it to fine-tune LLAMA (Touvron et al., 2023). Other works (Sun et al., 2024; Xu et al., 2023a; Wang et al., 2022; Yin et al., 2023; Xu et al., 2023b) developed various Synthetic Data Generation processes to generate more diverse/complex sets of instructions. In Aplaca (Taori et al., 2023), researchers used a small seed of 175 instructions, and prompt ChatGPT to generate a 52k instruction-tuning dataset by utilizing the Self-Instruct Wang et al. (2022) approach. WizardLM (Xu et al., 2023a) developed Evol-Instruct that can generate a set of instructions with increasing complexity.

# 2.2 DOMAIN SPECIFIC INSTRUCTION-TUNING

Our work falls within the line of research focuses on developing expert LLMs through instruction tuning in specific domains, such as writing assistants (Zhang et al., 2023), arithmetic (Liu & Low, 2023), translation (Jiao et al., 2023), medicine (Thawkar et al., 2023), code (Chaudhary, 2023; Luo et al., 2023), and many others.

Specifically for the domain of cyber-security, there have been several works that aimed at training security models. Although not directly related, a line of works trains Encoder-only architecture on security data either from scratch (Bayer et al., 2022; Park & You, 2023) or as continual pre-training (Ranade et al., 2021; Aghaei et al., 2022). However, these models are neither generative nor were trained to follow instructions. For the specific task of fine-tuning generative models for cybersecurity applications, VulDetect (Omar & Shiaeles, 2023) fine-tuned GPT-2 on a dataset containing both vulnerable and non-vulnerable code. The model is fine-tuned to detect anomalies that represent regular behavior. CyberBench was introduced by Liu et al. as a cyber-security evaluation dataset that was collected from different works and combined into one security benchmark that includes Name Entity Recognition (NER) tasks for cyber-security corpus, summarization of security blogs, multichoice Q&A and Classification tasks. SecureFalcon (Ferrag et al., 2023) was trained to differentiate between vulnerable and non-vulnerable C code samples, and is specialized in detecting software vulnerabilities. In contrast to previous efforts, we do not focus on one or more predefined set of tasks. We generate a highly complex and diverse dataset of security instructions spanning a broad spectrum of topics and skills using a domain-expert-driven instruction generation process. As will be described below, we use both domain-expert knowledge alongside LLM generation capabilities to populate our security instruction dataset. This comprehensive dataset enables us to train general-purpose security models.

# 3 SECKNOWLEDGE: DOMAIN-KNOWLEDGE DRIVEN CYBER-SECURITY INSTRUCTION DATASET

This section details the construction of SecKnowledge, a novel instruction tuning dataset specifically designed for the domain of cyber-security. We leverage expert knowledge and employ a two-step process to build a comprehensive and diverse dataset capable of supporting instruction tuning for various security-related tasks. The two-step process is defined as follows:

1. The first generation step focuses on creating high-quality instructions based on predefined schemas. These schemas are established through experts-driven in-depth analysis of the diverse set of security datasets, their individual characteristics, and the relationships between different entities within and between datasets. This ensures that the instructions are relevant, accurate, and capture the nuances of various security concepts and tasks. More specifically, each predefined schema consists of rules by which the data-source should be processed into instructions using parsers we developed, ensuring that the generated instructions focus on the important and unique characteristics of the data-source, and are representative of real-world security scenarios. Our method can be considered as an extension to methods such Wei et al. (2021); Longpre et al. (2023), where templates are simply assigned with predefined questions and answers. In Section 3.1 we break down the generation process.   
2. The second generation step expands the generated initial dataset and improves its diversity and complexity. To do so, we employ a hybrid synthetic content-grounded data generation process. More specifically, we fused Evol-Instruct (Xu et al., 2023a) and Self-Instruct (Wang et al., 2022) and combined them with content-grounded generation and evaluation pipelines. Additionally, we implemented a routing mechanism between the two generation methods that helps to reduce hallucinations. This process leverages the initial set of instructions and data from the first generation step to generate additional instructions that follow the established schemas but increase the model's overall generalizability. By incorporating content-grounded synthetic data, we increase the diversity and volume of the final dataset, ultimately leading to more robust and capable security models. In section 3.2, we further elaborate on the specifics of the generation process.

Our final SecKnowledge dataset consists of various instruction types, among which are: open/closed book question answering, yes/no questions, multi-choice Q&A, Chain of Thoughts (CoT) (Wei et al., 2022), summarization, logic validation, odd/leave one out multi-choice Q&A, question generation, query/rule explanation and generation, TTP mapping, and others.

Table 1 summarizes the security instruction sets composed in the first generation step. Unless otherwise specified, we use the open-source Mixtral (Jiang et al., 2024) model for both data generation and evaluation processes.

# 3.1 FIRST GENERATION STEP: DOMAIN KNOWLEDGE-DRIVEN INSTRUCTION GENERATION

Leveraging domain expertise, we first parse and enrich each one of the various security data sources using their unique characteristics and structure, derive connections between the documents in each data-source, and even derive connections between different data sources, as we will describe in the upcoming sections.

We establish a set of predefined, domain-knowledge-driven, schemas that capture the essential elements of different security tasks. Each schema consists of a series of pre-defined, domain-expertise-driven rules. Each rule is then translated into a parsing object. The parsing object will then generate and fill the instruction templates with the parsed data.

These schemas enable building instructions that capture each dataset's unique objectives and characteristics. This approach ensures that the instructions accurately reflect the desired model behavior and provide a strong foundation for effective instruction tuning.

The subsequent paragraphs provide a detailed description of the data sources and methodologies employed in the first step of the SecKnowledge data generation process.

<table><tr><td>Dataset</td><td># of generated instruction</td></tr><tr><td>MITRE ATT&amp;CK</td><td>45,901</td></tr><tr><td>CWE</td><td>4,080</td></tr><tr><td>CVE</td><td>8,447</td></tr><tr><td>CAPEC</td><td>3,917</td></tr><tr><td>Security Wiki</td><td>11,000</td></tr><tr><td>Security interview Q&amp;A</td><td>500</td></tr><tr><td>Threat reports</td><td>4,500</td></tr><tr><td>BRON</td><td>62,227</td></tr><tr><td>SIEM alert TTP mapping</td><td>400</td></tr><tr><td>Sigma rules</td><td>9,329</td></tr><tr><td>Security Stack Exchange</td><td>2,573</td></tr><tr><td>Total</td><td>152,874</td></tr></table>

Table 1: Overview of the initial instructions constructed from the datasets on the first step as described in 3.1. These instructions will be used as the seed for the second generation step.

# 3.1.1 STRUCTURE-DRIVEN INSTRUCTION GENERATION

The straightforward method for creating instructions dataset from the documents is to provide a teacher model with raw documents and instruct it to generate instructions based on the content. However, this method presents several challenges. Relying on models to produce instructions that simultaneously capture the unique characteristics of datasets while maintaining complexity and diversity proves to be a difficult task. One reason is that models tend to focus on specific or localized sections of a document when generating instructions. More significantly, models struggle to capture and exploit the relationships between different components within each dataset and the relationships between different datasets.

In this section, we introduce a different method for generating an instruction set from documents. Our approach exploits the structured nature of the various cyber-security documents to create a high-quality, diverse, and complex instruction dataset. We demonstrate the efficiency of our method using the MITRE framework, a comprehensive security resource that encapsulates years of expertise in the security domain. We developed specialized parsers that use predefined schemas and rules that harness the structured nature of the data to generate instructions. These parsers extract relationships between different entities within the datasets and, using their corresponding schemas, transform these documents into instructions

More specifically, we utilize the structured nature of the different MITRE datasets, among which we can find:

- MITRE ATT&CK - comprehensive knowledge base of adversary tactics and techniques based on real-world observations, MITRE ATT&CK also provides detection and mitigation methods for each technique and their use by threat groups and software tools $^{1}$ .   
- CWE (Common Weakness Enumeration) - community-developed list of software and hardware weakness types, serving as a common language for describing security vulnerabilities $^{2}$ .   
- CVE (Common Vulnerabilities and Exposures) - dictionary of publicly known cybersecurity vulnerabilities and exposures, aims at standardizing the way vulnerability information is shared $^{4}$ .   
- CAPEC (Common Attack Pattern Enumeration and Classification) - a structured catalog of common attack patterns describing how adversaries exploit application weaknesses and other cyber-enabled capabilities. Attack patterns describe the common attributes and approaches that adversaries employ to exploit known weaknesses in cyber-enabled capabilities $^{4}$ .

All of which are oriented towards the domain of cyber-security. The compilation of these frameworks encompasses a vast repository of cyber-security domain knowledge and offers extensive coverage of the security field, making it an excellent resource for fine-tuning our model to the specific requirements and nuances of cyber-security. For additional information on each framework, see appendix A.

Each MITRE framework comprises a structured format that categorizes different aspects of the subject matter, enabling organized analysis of the different security aspects. As such, we create a schema for each framework. The schema results in the following types of instructions:

1. A set of instructions designed to teach the model the specific characteristics of each object (i.e., tactic, technique, mitigation, detection, attack pattern, etc.) For instance, an instruction could detail the relationships between an attack pattern and its corresponding severity, prerequisites, or consequences.   
2. A set of instructions designed to guide the model in understanding the relationships between different objects within each dataset.

![](images/57200de6d8948e0abd5e22cc66661b45f34d8a0eb5160d78cb87dd1adf2d12d1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Data source"] --> B["Data component"]
    C["Tactics"] --> D["(Sub) Technique"]
    B <--> D
    E["Detection"] --> B
    F["Mitigation"] --> D
    G["Group"] --> H["Software"]
    H --> I["Campaign"]
    D <--> H
```
</details>

Figure 1: Relationship between different MITRE ATT&CK components.

Next, we provide an example of chain-of-thoughts instruction generated by utilizing the MITRE ATT&CK structure. Figure 1 demonstrates the relationships between different objects within the MITRE ATT&CK framework. Using these relationships, complex instructions are constructed on the wide range of the attack land-space. For instance, see Figure 2, where Chain of Thoughts (CoT) training example is created. This process begins by asking about the usage of specific malicious software and identifying the tactic to which this software is related. Our parser then relies on the knowledge of the relationships between different components (tactics, techniques, sub-techniques,

software) to derive the chain of connections from the specific malicious software to the relevant exploited tactic.

Note that no language model is used during the construction; the connections and relevant text are derived based on our knowledge of the dataset's structure.

![](images/727b2545e90d2de369912a972e846e5519ddf7b3af70efd5751c3b1020d6e53c.jpg)

<details>
<summary>text_image</summary>

Instruction:
Which tactic is related to the following application of software
{software_name}: {software_usage}

Answer:
{software_description}
{software_name} is identified by MITRE ID (software_id).

The provided usage of {software_name} ({software_id}) pertains to the
MITRE sub-technique {sub_technique_name} ({sub_technique_id}).
This is since using the given sub-technique,
{sub_technique_description}.

sub-technique {sub_technique_name} ({sub_technique_id}) is a type
of technique {technique_name} ({technique_id}).

{technique_name} ({technique_id}) is a type of {tactic_name}
({tactic_id}).

Therefore, the answer is: tactic {tactic_name} ({tactic_id}).

Instruction:
Which tactic is related to the following application of software "Kwampirs": Kwampirs copies
itself over network shares to move laterally on a victim network

Answer:
Kwampirs is a backdoor Trojan used by Orangeworm.
It has been found on machines which had software installed for the use and control of high-
tech imaging devices such as X-Ray and MRI machines.

Kwampirs is identified by MITRE ID S0236.

The provided usage of Kwampirs software (S0236) pertains to the MITRE sub-technique
"Remote Services: SMB/Windows Admin Shares" (T1021.002). This is since using the given
sub-technique, adversaries may use Valid Accounts to interact with a remote network share
using Server Message Block (SMB). The adversary may then perform actions as the logged-on
user.

sub-technique "Remote Services: SMB/Windows Admin Shares" (T1021.002) is a type of
technique Remote Services (T1021).

Technique Remote Services (T1021) is a type of tactic Lateral Movement (TA0008).

Therefore, the answer is: tactic Lateral Movement (TA0008).
</details>

Figure 2: Example of constructing CoT by utilizing our knowledge of the structure relationships between different components within the MITRE ATT&CK framework. On the left side, there is a template to map from a given malware usage to its corresponding tactic. On the right side, the template is assigned with a specific malware usage and its chain of connections up to the relevant tactic.

# 3.1.2 STRUCTURED LLM-AUGMENTED INSTRUCTION GENERATION

In the previous section, we demonstrated how to use raw data alongside domain expertise to populate our schema templates. Here, we combine the same abilities of robust predefined schemas and domain knowledge with the flexibility and reasoning abilities of LLMs to create comprehensive instructions. This approach leverages structured templates for consistency while utilizing a teacher model to dynamically generate and fill specific content, ensuring both accuracy and adaptability in instruction creation. More specifically, our main goal of using the teacher model is not to generate general content, but rather to harness the reasoning capabilities of the teacher model to guide our models to reason on complex security concepts based on the information we provide.

We use structured LLM-augmented instruction generation on the following datasets: BRON, SIEM Rules to TTP Mapping, and Sigma Rules.

BRON: BRON (Hemberg et al., 2021) is a graph that interconnects threat data sourced from MITRE ATT&CK, CAPEC, CWE, CVE, MITRE Engage, MITRE D3FEND, MITRE CAR, and exploitdb. This interconnected graph enhances the capabilities of security researchers in conducting advanced threat hunting.

After demonstrating in section 3.1.1 how the MITRE frameworks can be utilized (individually) to generate instructions on the specific characteristics of each MITRE object and the relationships between different objects within each framework, we will leverage BRON to generate instructions on the relationships between different objects across frameworks. See Figure 3 for a high-level overview of the graph's structure.

With hundreds of thousands of nodes and millions of edges interconnecting them, BRON's sheer scale makes it impractical to feed directly to an LLM with the expectation of comprehensively learning all relationships. Therefore, our primary objective is to generate an instruction set that teaches the model to reason if and how different entities are related.

Specifically, using BRON, we have two main goals: 1) construct instructions that will guide LLMs how to reason if two consecutive entities are related to each other (e.g., CWE and CVE nodes), and 2) showcase the reasoning process for LLMs to derive the path from a specific entity of interest to any other entity in the graph, to accommodate user instruction. This reasoning process will

enable a more comprehensive understanding of the relationships between different entities, such as the connection between a platform and its relevant weaknesses, which are not directly related.

![](images/c503e575edd6a9e78178b78fe286c3637544b9f14835905734ff081c6c1466ed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["D3fend Mitigations"] --> B["D3-BAN"]
    A --> C["D3-OTP"]
    D["Tactics"] --> E["TA0040"]
    D --> F["TA0008"]
    G["Groups"] --> H["G0098"]
    I["Technique Detections"] --> J["T1531"]
    K["Softwares"] --> L["S0437"]
    M["CAPEC Detections"] --> N["CAPEC-2"]
    O["CAPEC Mitigations"] --> P["M1018"]
    Q["CWE Detections"] --> R["CWE-654"]
    S["CWE Mitigations"] --> T["CWE-308"]
    U["CPEs"] --> V["CVEs"]
    W["barbican by OpenStack"] --> X["CVE-2023-49700"]
    Y["CWE Mitigations"] --> Z["CVE-2024-24771"]
    style A fill:#f9f,stroke:#333
    style D fill:#bbf,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style M fill:#bbf,stroke:#333
    style Q fill:#f9f,stroke:#333
    style S fill:#bbf,stroke:#333
    style U fill:#f9f,stroke:#333
    style W fill:#bbf,stroke:#333
```
</details>

Figure 3: BRON high-level graph structure overview

To meet the stated goals, the graph should be processed and traversed into paths, which we will later enrich with domain knowledge from different resources. These paths and explanations will effectively become chain-of-thoughts examples that can guide LLMs to perform effective, complex threat-hunting reasoning. Our processing requires four steps: path extraction - in which we perform a walk on the graph and extract paths, deriving connections between direct nodes, constructing CoT instructions based on the extracted paths, and multi-path CoT.

PATHS EXTRACTION First, we gather all one-step paths between nodes of different types that are directly connected in the graph (e.g., all connections between tactic nodes and technique nodes). Next, for all non-direct paths, we perform a random walk on the graph and construct up to 5000 paths between each pair of node types that are not directly connected (e.g., paths between tactic nodes and CVE nodes).

DERIVE THE CONNECTION BETWEEN DIRECT NODES After extracting one-step paths between nodes of different types (e.g., CAPEC node and CWE node) that are directly connected in the graph, we take these direct links and use the reasoning process of a teacher model to explain the connection between each pair of nodes. This involves sending the teacher model instructions that include the descriptions, alongside other information about each node, for each pair of nodes, requiring the teacher model to examine the information and decide if and how the nodes are connected. Additionally, we incorporate negative sampling to illustrate that not all nodes in the graph are connected, compelling the model to make decisions based on the nodes' information. The negative sampling stage is pivotal as it's impractical to present all existing paths (amounting to millions) to the models we fine-tune. Instead, as our models already see these data sources (ATT&CK, CWE, etc.) separately during the fine-tuning process, we aim to equip them with the ability to ascertain whether two nodes are linked based on their information, in the expectation that our fine-tuned models will generalize the reasoning process to paths they haven't encountered during training.

CONSTRUCTING CO T ON PATHS Additionally, we present the model with longer paths that involve multiple nodes, between non-direct nodes. When constructing these paths, we need to generate

an explanation for the connection between each two nodes in the path. To generate the explanations, we enrich the paths with the information we have on each node from our other datasets (MITRE ATT&CK, CAPEC, etc.). More specifically, for each edge in the path, we use the teacher model to explain the connection between the nodes based on the relationship status (e.g., CVE is a specific implementation of CWE) and the nodes' information (e.g., description) we provide, similar to the direct node processing. This explanation is then attached to the edge between the two nodes. As a result, a Chain-of-Thought (CoT) explanation is generated for each path.

MULTI-PATH CO T Lastly, we construct more complex instructions, i.e., instructions that can be answered only by involving multiple paths from the graph. One example can be a two-stage/paths instruction where we can first ask what the relevant attack patterns for a given weakness are, and in the second stage how to detect/mitigate them. See Appendix D.1 for additional details and Figure 4 for an example.

# Instruction:

Given software epolicy\_orchestrator (by mcafee) version 5.10.0, identify a CWE (Common Weakness Enumeration) that might be connected to it.

Then, suggest some strategies to mitigate it. Explain the rationale behind your choices and how they are relevant to the software.

# Answer:

The conceivable link between application epolicy\_orchestrator (by mcafee) version 5.10.0 and Authentication Bypass by Alternate Name (CWE-289) is illuminated by the sequence:

application epolicy\_orchestrator (by mcafee) version 5.10.0 => CVE-2022-0845 => Authentication Bypass by Alternate Name (CWE-289)

The relationship between application epolicy\_orchestrator (by mcafee) version 5.10.0 and Authentication Bypass by Alternate Name (CWE-289) is clarified by the reasoning outlined below:

# Step 1:

The CVE-2022-0845 affects the application epolicy\orchestrator version 5.10.0 by McAfee, as it is vulnerable to code injection due to a vulnerability in the pytorchlightning/pytorch-lightning GitHub repository prior to version 1.6.0.

# Step 2:

The CWE-289 Authentication Bypass by Alternate Name is being used in CVE-2022-0845 by exploiting the code injection vulnerability in pytorchlightning/pytorch-lightning prior to 1.6.0, where the authentication is performed based on the name of the resource being accessed, but it does not properly check all possible names for that resource.

# You can mitigate Authentication Bypass by Alternate Name (CWE-289) by:

\- Avoid making decisions based on names of resources (e.g. files) if those resources can have alternate names.

\- Inputs should be decoded and canonicalized to the application's current internal representation before being validated (CWE-180). Make sure that the application does not decode the same input twice (CWE-174). Such errors could be used to bypass allowlist validation schemes by introducing dangerous inputs after they have been checked.

\- Assume all input is malicious. Use an "accept known good" input validation strategy, i.e., use a list of acceptable inputs that strictly conform to specifications. Reject any input that does not strictly conform to specifications, or transform it into something that does. When performing input validation, consider all potentially relevant properties, including length, type of input, the full range of acceptable values, missing or extra inputs, syntax, consistency across related fields, and conformance to business rules

By following these approaches, you can mitigate Authentication Bypass by Alternate Name (CWE-289) effectively.

Figure 4: Illustration depicting the construction of a CoT by extracting paths from BRON, enriching them with data and domain knowledge, and using LLM to formulate connections based on the provided information.

SIEM Rules to TTP Mapping: SIEM (Security Information and Event Management) is a security platform that monitors and correlates threat intelligence, network, and user behavior anomalies to prioritize high-fidelity alerts. We have collected a list of 400 rules from IBM's SIEM, QRadar, along with their corresponding Tactics, Techniques, and Procedures (TTP) mappings.

TTP mapping of detection rules is critical in cyber-security as it enables organizations to systematically identify and counteract specific adversary behaviors, thereby enhancing the precision and effectiveness of threat detection.

QRadar's rules are well-structured and include fields such as rule ID, description, pattern, relevant MITRE tactic/technique ID and name, rule risk level, and more. In the following, we will demonstrate how we leverage this structure to develop a series of instructions for educating CyberPal.AI on mapping rules to Tactics, Techniques, and Procedures (TTPs). Our goal is not merely to create a simple mapping task but to teach the model to reason about TTP mapping. To achieve this, we com-

bine expert knowledge with LLMs to generate a comprehensive TTP reasoning instruction dataset, as we describe below.

The process of creating the TTP mapping instruction set involves retrieving the rule description, tactic/technique ID, and name for each rule and its corresponding TTP mapping. Using this information, we access the description and additional relevant data of the tactic/technique from the MITRE ATT&CK framework. We tailor a specific schema, that leverages the required information, and guides the teacher model to reason and clarify the relationship between the rule and the provided tactic/technique based on their descriptions and additional relevant data. Subsequently, the model generates an explanation, which undergoes evaluation for correctness by another teacher model (evaluator). Upon acceptance by the evaluator, a set of instructions is formulated based on the rule, the TTP mapping, and the explanation. The goal of the generated set is to demonstrate to CyberPal.AI the reasoning approach of mapping between rules and TTPs. See Figure 5 for an example of such instruction.

# Instruction:

What is the MITRE tactic relevant to the rule below. explain the connection. Add explanation about tactic.

Rule name: RFISI: Internal Connection with Botnet Command and Control

Rule description: This rule will notify when an internal host communicates with an IP known to be a botnet command and control server. Indicates that and internal infection is becoming active in a botnet.

# Answer:

This QRadar rule, named "RFISI: Internal Connection with Botnet Command and Control," is related to the MITRE ATT&CK tactic - Command and Control (TA0011).

The rule's description states that it will notify when an internal host communicates with an IP known to be a botnet command and control server. Based on this description, we can make several connections as to why this rule is mapped to the Command-and-Control tactic.

First, let's understand the Command and Control (C2) tactic. In the context of MITRE ATT&CK, C2 refers to the attacker's control of compromised systems within a victim's network. The attacker establishes a communication channel with these compromised systems to control and manage their malicious activities.

In this rule, the alert is triggered when an internal host communicates with an IP address that is flagged as a botnet command and control server. A botnet is a network of compromised machines (often called "bots") that are under the control of an attacker or a botnet operator. The command-and-control servers act as the central point of communication and coordination for the malicious activities conducted by the botnet.

By detecting internal hosts communicating with an IP address known to be a botnet command and control server, this rule helps identify potential indicators of compromise (IOCs) related to Command and Control. The alert notifies security analysts when an internal infection is becoming active in a botnet, indicating the presence of an unauthorized and malicious control infrastructure within the network.

Therefore, given that the rule is focused on identifying internal connections with botnet command and control servers, it aligns with the MITRE ATT&CK tactic - Command and Control (TA0011). This rule helps security analysts detect and respond to potential command and control activity within their network, contributing to the overall defense against sophisticated threats.

Figure 5: Example of generated instruction using our mapping process from SIEM rules to TTPs. The answer is the explanation that was generated in our construction process.

Sigma Rules: Sigma is a structured and open signature format that allows to define and describe detection logic. The rule format is flexible and platform-agnostic. The main purpose of Sigma is to provide a structured form in which researchers and analysts can describe their developed detection methods and make them shareable. SigmaHQ $^{5}$ is the main rule repository where detection engineers, threat hunters, and all defensive security practitioners collaborate on detection rules. Here, we leverage the repository's dataset, which contains over 3000 diverse and reliable detection rules as a baseline for our rule instruction set.

Sigma rules contain multiple fields, among which are: the "logsource" field which specifies the type of log data the rule applies to, and the "detection" field which defines the specific conditions that trigger the rule, including event attributes, expected values, and filters for accurate detection. The "level" field indicates the severity of the detected event. Each Sigma rule is connected to the attack it tried to detect.

We take advantage of the Sigma rules structure, and feed the relevant fields to a teacher model, and construct the following types of reasoning instructions:

- Step-by-step attack detection explanation using log source and rule detection filters within the detection field   
- Step-by-step reasoning for attack type mapping via detection indicators   
- Sigma rule generation from attack type and/or detection indicators

We customize a schema for each task type, ensuring it contains the necessary information. See Figure 6 for an example of such instruction. As with other generation processes, we also apply an evaluator (LLM) that tests the correctness of the generated text.

![](images/37c84646159ebd0cfa213d966241ad0637b245cf2c5f342b2b2babde3df98d98.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Sigma rule goal:<br>Detects Obfuscated Powershell via use MSHTA in Scripts"] --> B["Original Sigma Rule:<br>logsource:<br>  product: windows<br>  category: driver_load<br>  detection:<br>  selection:<br>    ImagePathcontainsllal:<br>    - 'set'<br>    - '&&<br>    - 'mshta'<br>    - 'vbscript:createobject'<br>    - '.run'<br>    - '(window.close)'<br>  condition: selection<br>  falsepositives:<br>  - Unknown<br>  level: high<br>  tags:<br>    - attack.defense_evasion<br>    - attack.t1027<br>    - attack.execution<br>    - attack.t1059.001"]
    C["Instruction:<br>How can I detect Obfuscated Powershell via use MSHTA in Scripts?"] --> D["Original Sigma Rule:<br>In order to Detect Obfuscated Powershell via use MSHTA in Scripts, you should: Monitor Windows drivers loading for any instances where the ImagePath contains a specific sequence of elements indicating a potential attempt at obfuscation.<br>Specifically, lookout for the simultaneous presence of 'set', '&', 'mshta', 'vbscript:createobject', '.run', and {(window.close)' within the ImagePath.<br>This pattern might suggest an attempt to execute PowerShell scripts in a manner designed to evade detection.<br>This detection is critical due to its high severity level, and while there may be unknown false positives, the absence of known benign use-cases for this precise combination of elements underscores its importance.<br>The relevant tactics and techniques from the MITRE ATT&CK framework associated with this activity include Defense Evasion and Execution, specifically identified as Obfuscated Files or Information (T1027) and Command and Scripting Interpreter: PowerShell (T1059.001)."]
```
</details>

Figure 6: Sigma instruction example - from the original Sigma rule, we construct a task to explain how to detect a specific attack. The Sigma rule will be processed, and based on the relevant instruction template and model prompt, we will construct the instruction and its corresponding answer.

# 3.1.3 ADDITIONAL DATASETS

We have collected and generated security-related instructions on various other security datasets: security interview Q&A, threat reports on various security threats, security and reverse-engineering stack-exchange, and Wikipedia pages related to computer security. For each dataset, we define a schema based on its structure and build instructions in a similar manner to the previously mentioned datasets.

# 3.2 SECOND GENERATION STEP: CONTENT-GROUND SYNTHETIC DATA GENERATION

In the second step of our security instruction generation process, we expand the generated initial dataset from 3.1 and improve its diversity and complexity.

For the purpose of synthetic data generation, we build upon the ideas of Self-Instruct (Wang et al., 2022) and Evol-Instruct (Xu et al., 2023a) and fuse them alongside content-grounded generation and an instruction routing mechanism.

Evol-Instruct starts with an initial set of instructions and rewrites them step by step into more complex instructions. More specifically, Evol-Instruct uses In-depth and In-breath evolving. The In-depth Evolving includes five types of operations: adding constraints, deepening, concretizing, increasing reasoning steps, and complicating input. The In-breadth Evolving is a mutation, i.e., generating a completely new instruction based on the given instruction.

Self-Instruct also starts with an initial set of instructions. Then, the model is prompted to generate instructions for new tasks. It leverages the existing collection of instructions to create more broad-coverage instructions that define, often new, tasks. We can think of In-breadth Evolving as a variant of Self-Instruct.

In the cyber-security domain, accuracy is paramount, as hallucinations can lead to disastrous consequences, such as applying incorrect detection and mitigation strategies, resulting in security breaches and inadequate organizational responses.

Therefore, we fuse the two methods with content-grounded generation, meaning we try to force the model to generate a more complex instruction that is grounded by the document from which the previous instruction was generated.

We found that using Evol-Instruct with content-grounded generation tends to diverge and generate inaccurate instructions after several iterations (usually around 3 iterations), resulting in instructions that an LLM cannot answer, non-grounded instructions or instructions that deviate from the relevant topic. Therefore, we incorporate a dynamic mechanism that combines Evol-Instruct together with Self-Instruct (See the "Instructions router" in Figure 7), where in the early stages of the synthetic data generation, we mainly focus on In-depth Evolving with the goal of generating more complex instructions on the same topic, and as the generation process progresses, we shift the focus towards Self-Instruct (which can be thought of as equivalent to the In-breath instruction generation from Evol-Instruct), where we mainly focus on generating new tasks, while keeping them grounded and in the same domain as the document. More specifically, the probability of Self-Instruct being chosen is doubled every two iterations. We find that combining Evol and Self-Instruct leads to better content-grounded instructions (due to the difficulty of preserving complex content-grounded instructions in later stages).

Additionally, we incorporate an internal evaluation mechanism using an LLM evaluator. The evaluator is defined by the following objectives:

- Evaluate if the new instruction is more challenging/complex/rare or diverse   
- Evaluate if the new instruction is of the same domain as the given instruction based on the document, and evaluate that new instruction can be answered by the document   
- Evaluate if the generated answer correctly answers the new instruction, and that the generated answer is grounded by the document

An instruction will be added to the instruction pool and be used in the next iteration only if it passes the evaluator's assessment. Note that not all datasets are suitable for full SDG (i.e., synthetic data generation without domain expert intervention). For example - we cannot use SDG without domain expert intervention on BRON as it may hallucinate paths. We also cannot use SDG to generate new rules as it is not trivial for a model to generate new and accurate rules in non-common rule languages (e.g., in Sigma rules). Therefore, we employ our content-grounded SDG process to Security Wiki, Security interview Q&A, Security Stack Exchange, MITRE ATT&CK, CWE, CVE, and CAPEC. In the second generation step, we generate an additional 250,000 instructions, consisting of all instructions from different iterations stored in the instruction pool. In total, the final dataset comprises approximately 400,000 complex and diverse instructions following the two generation steps.

# 4 SECKNOWLEDGE-EVAL: A COMPREHENSIVE SECURITY EVALUATION DATASET

To assess CyberPal.AI's performance, we constructed a diverse set of seven new evaluation datasets aimed at testing the model's capabilities in cyber threat intelligence. To ensure no data contamination between the fine-tuning and testing phases, we partitioned the raw documents into train and test sets, such that the model did not encounter any test-related documents during fine-tuning. For example, if a specific CWE instance was included in the test split, it was not seen by the model during the fine-tuning process. After splitting the data, we transformed the documents from the test split into the evaluation tasks described below. Furthermore, we benchmarked CyberPal.AI against another seven public and general cyber-security evaluation datasets to demonstrate its robustness and comprehensive understanding of security concepts. Overall, our evaluation benchmark consists of 14 diverse datasets, with various task types. To the best of our knowledge, this is the most comprehensive cyber-security evaluation benchmark, composed of an extensive set of cybersecurity tasks alongside other publicly available security benchmarks. For high-level statistics about SecKnowledge-Eval, see Appendix C. The following paragraphs introduce these tasks and datasets.

![](images/6db76893d60c110da726c85606bbcb911af7a6349b27e49187c952b97b32aba3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Current Instruction"] --> B["Instructions router"]
    B --> C["In-depth content grounded generation"]
    B --> D["In-breath content grounded generation"]
    C --> E["LLM Evaluator"]
    D --> E
    E --> F["Add instruction to instructions pool if all conditions hold"]
    E --> G["Eliminate if conditions are not met"]
    H["Document"] --> C
    I["Document"] --> D
    J["Document"] --> C
    K["Repeat for k iterations"] --> E
    L["Container"] --> E
```
</details>

Figure 7: High-level illustration of the content-grounded Synthetic Data Generation (SDG) process.

# Multiple Choice Tasks

Questions in this section are formatted with four multiple-choice answers, similar to the question template from (Hendrycks et al., 2020).

Adversarial MITRE ATT&CK We compiled this dataset using various MITRE ATT&CK sources. It is designed to assess the model's knowledge of malicious software, campaigns, attack tactics and techniques, data sources, and potential detections and mitigations for different attacks. The input consists of information about a given MITRE instance (e.g., description), and the correct answer is the source from which it was derived.

To enhance the dataset's difficulty and test our models' robustness, we developed a novel adversarial attack (Goodfellow et al., 2014; Carlini & Wagner, 2017; Levi & Kontorovich, 2023) for multi-choice questions on closed domains. For example - in domains where the options for the answer are taken from a closed list of possible options, our goal is to choose the false options that will confuse the model with the highest probability without manipulating the question itself. The main idea is as follows: assume a multiple-choice question, where the choices are taken from a closed list of possible options, with size $k$ . For each of the $k-1$ false options, we construct a new classification question, where the question is the original question, and there are two options the model should select from: one is the correct option and the other is one of the $k-1$ false options. For each such question, we query an LLM with $k-1$ such queries, and select the false options that were the most likely to be selected with respect to the question and the correct option.

As an example, given an instruction I from a specific domain within the MITRE framework such as techniques, we generate k-1 queries for each I, where k represents the number of all existing techniques, and subsequently query a third-party LLM k-1 times in the following way: for each false technique option from the list of k techniques, the LLM is tasked to classify whether the instruction corresponds to the correct technique or the false technique. When constructing the adversarial questions, our objective is to identify techniques where the Log-Likelihood is the lowest when compared to the correct answer. Appendix B presents more details about the attack, which results in our adversarial evaluation dataset.

SIEM Rule TTP Mapping SIEM solutions usually include rules that detect a wide range of activities, including excessive firewall denies, multiple failed login attempts, and potential botnet activity. We developed a dataset comprising IBM's QRadar rules, aiming to classify each rule according to the appropriate tactic or technique. This TTP classification dataset is structured as a multiple-choice dataset with four options due to multiple possible correct tags.

CTI Detection and Mitigation Mapping As outlined, BRON captures the interrelationships between different Cyber Threat Intelligence (CTI) frameworks, such as MITRE ATT&CK, CWE, CVE, CAPEC, and more. We created a dataset designed to assess models' capabilities in understanding these interconnections. Specifically, we evaluate the model's proficiency in mapping from tactics, techniques, attack patterns, weaknesses, and vulnerabilities to potential detections and mitigations. The ability to accurately map and explain these detections and mitigations is crucial for the model to perform effectively as a CTI security assistant.

CWE Technical Impact Mapping In CWE, each weakness, if successfully exploited, can lead to one or more technical impacts out of eight options: modify data, read data, DoS: unreliable execution, DoS: resource consumption, execute unauthorized code or commands, gain privileges / assume identity, bypass protection mechanism, and hide activities. We have developed an evaluation set that presents the model with CWEs and their descriptions, and the goal is to classify each CWE to its related technical impact. To ensure an accurate evaluation, we selected only CWEs with exactly one impact. For CWEs with multiple possible impacts, we converted the question into a multiple-choice format.

CISSP Assessment Questions The Certified Information Systems Security Professional (CISSP) is a well-known recognized certification in the field of cyber-security. It validates a professional's deep technical and managerial competence in designing, engineering, and managing an organization's overall security posture. We have developed an evaluation set based on multiple-choice questions drawn from the assessment tests within the CISSP learning material.

MMLU Computer Security (SecMMLU) Is a subset of the MMLU (Measuring Massive Multitask Language Understanding) Hendrycks et al. (2020) specifically focused on the computer security domain. The original MMLU encompasses multiple-choice questions from a variety of fields, but only those about computer security are utilized in our evaluation.

Cybersecurity Skill Assessment Is a multiple-choice cyber-security subset of the practice questions used for professional skill assessments on LinkedIn. These questions aim to evaluate candidates' general knowledge in the cyber-security domain $^{6}$ .

CyberMetric CyberMetric Tihanyi et al. (2024) is a benchmark dataset for evaluating Large Language Models Knowledge in cyber-security. The questions for the benchmark were created through a collaborative process, i.e., merging expert knowledge with LLMs. We used the 500-question dataset, verified by human evaluators, which covers a wide range of topics within cyber-security, chosen by 30 security experts.

Cyber Threat Intelligence Multiple Choice Questions (CTI-MCQ) CTI-MCQ is a benchmark dataset developed by Alam et al. (2024) for assessing LLMs' knowledge and capabilities on attack patterns, threat actors, APT campaigns, detection methods, mitigation strategies, common software vulnerabilities, attack pattern enumeration, alongside public CTI quizzes.

SecEval SecEval Li et al. (2023) is a benchmark for evaluating cyber-security knowledge in Foundation Models (FMs), offering over 2000 multi-choice, multi-option questions across 9 domains, generated by OpenAI GPT-4 using authoritative sources such as open-licensed textbooks, official documentation, and industry guidelines and standards.

# Classification Tasks

CTI Relationship Prediction A major role of our model is to learn and understand the relationships between different CTI frameworks. For example, it must determine if and how a given CVE and CWE are related. To test this ability, we have built a dataset that presents the model with two entities (e.g., instances of CVE and CWE) and two possible explanations—one explaining why the entities are related and another explaining why they are not. The model's objective is to classify which explanation is correct, or, in other words, to determine if the two entities are related or not.

CTI Entity Classification We have developed a dataset consisting of various descriptions corresponding to different CTI entities (such as tactics, techniques, software, etc.). The model's objective is to classify whether a given description is related to the specified entity.

Cyber Threat Intelligence Root Cause Mapping (CTI-RCM) CTI-RCM was developed by Alam et al. (2024) to identify the fundamental causes of vulnerabilities by correlating CVE records and bug tickets with their associated weaknesses (CWE entities). Accurate root cause mapping is essential for guiding investments, policies, and practices aimed at addressing and eliminating these vulnerabilities.

# Summarization Tasks

CWE Description Summarization We have developed a dataset containing weaknesses from the CWE dataset, intending to summarize the extended descriptions of each CWE. The target of the summarization is the short description provided for each CWE, which aims to offer a concise explanation of the CWE's extended description.

# 5 EXPERIMENTS

# 5.1 CYBERPAL.AI TRAINING DETAILS

Similar to (Mitra et al., 2023), we've empirically noticed that presenting the model with instructions of increasing length improves the model's learning ability. We extend on this idea and employ an incremental training methodology, organized at the dataset level. We sort the datasets of SecKnowledge into two hierarchical orders: first, we sequence the datasets by their data source category, with instructions from simpler data sources introduced first. For example - BRON-related instructions will be presented only after we present the model with MITRE ATT&CK and the other frameworks BRON is composed of. In the second hierarchy, within each category, we arrange the instructions based on the increasing length of their outputs.

To train CyberPal.AI, we use our generated SecKnowledge instruction dataset. As our baseline models, we used Llama-3 instruct 8B (AI@Meta, 2024), Mistral instruct 7B v0.3 (Jiang et al., 2023), and Phi-3-medium-4k-instruct (Abdin et al., 2024).

We employ a learning rate of 4e-5 for Llama and Phi, and 3e-5 for mistral. Additionally, we employ linear warm-up for 125 steps. The context length is set to 4096, and an effective batch size of 2048 is achieved using gradient accumulation. Based on our empirical findings, beyond 2 epochs, we observed that additional epochs have negligible impact on the final loss before the model starts to overfit.

# 5.2 EVALUATION METRICS

Assessing LLMs on selected datasets requires appropriate evaluation metrics. We apply suitable metrics for each task as described below. For Multiple-choice Q&A, we employ the common HELM Liang et al. (2022) evaluation method where the token with maximum probability is chosen. For summarization tasks, we use the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Lin (2004). Finally, for classification tasks, we use accuracy as the metric. To calculate the average score for evaluation tasks, a straightforward averaging technique is utilized. For summarization tasks specifically, the mean of ROUGE-1, ROUGE-2, and ROUGE-L scores is first determined before calculating the overall average. All evaluation tasks were done in a zero-shot setting.

# 5.3 CYBERPAL.AI RESULTS

To demonstrate the effectiveness of SecKnowledge, we present evaluation results of SecKnowledgeEval for both baseline models and their fine-tuned version, trained using SecKnowledge. As mentioned above, we used Llama-3 instruct 8B, Mistral instruct 7B v0.3, and Phi-3-medium-4k-instruct (a 13B parameters model) as our baseline models. We tested each model before and after applying fine-tuning with SecKnowledge. Results are presented in Tables 2 and 3. Additional results for Gemma-2 models (Team, 2024) can be found in Appendix E.

As can be seen by Table 2, on training aligned tasks (i.e., tasks that the model has seen during fine-tuning), our fine-tuned CyberPal.AI models exhibit significant and consistent improvement

<table><tr><td>Model</td><td>Original/Adv. MITRE ATTACK</td><td>SIEM Rule TTP Mapping</td><td>CTI Detection and Mitigation</td><td>CWE Summarization (R-1/2/L)</td><td>Technical Impact Mapping</td><td>CTI Relationship Prediction</td><td>CTI Entity Classification</td><td>Avg.</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>73.24/59.57</td><td>52.05</td><td>56.22</td><td>28.25/8.16/20.57</td><td>59.59</td><td>52.44</td><td>65.31</td><td>52.02</td></tr><tr><td>CyberPal.AI-Mistral</td><td>98.87/92.54</td><td>67.12</td><td>70.26</td><td>56.78/51.79/54.71</td><td>69.05</td><td>97.81</td><td>83.66</td><td>76.41</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>78.59/59.57</td><td>60.27</td><td>55.77</td><td>26.38/8.16/18.33</td><td>59.02</td><td>59.76</td><td>55.77</td><td>52.54</td></tr><tr><td>CyberPal.AI-Llama</td><td>97.04/87.74</td><td>64.38</td><td>81.95</td><td>46.43/38.45/43.88</td><td>66.18</td><td>97.81</td><td>81.95</td><td>74.70</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>77.32/64.50</td><td>55.47</td><td>67.92</td><td>27.96/7.83/19.94</td><td>66.76</td><td>63.75</td><td>67.92</td><td>57.84</td></tr><tr><td>CyberPal.AI-Phi</td><td>96.76/89.57</td><td>65.07</td><td>81.24</td><td>48.23/39.24/44.67</td><td>68.20</td><td>96.27</td><td>81.24</td><td>75.10</td></tr></table>

Table 2: Evaluation results for CyberPal.AI models compared to the base model on designated datasets constructed to evaluate the models' performance on training-aligned security tasks. For the MITRE ATT&CK evaluation dataset, we provide results for both the original evaluation dataset and its adversarial version, where we can see that our fine-tuned versions demonstrate greater robustness.

<table><tr><td>Model</td><td>CISSP Assessment</td><td>SecMMLU</td><td>Cybersecurity Skill Assessment</td><td>CyberMetric</td><td>CTI-MCQ</td><td>CTI-RCM</td><td>SecEval</td><td>Avg.</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>63.63</td><td>67.00</td><td>78.69</td><td>80.80</td><td>58.03</td><td>45.85</td><td>32.98</td><td>60.99</td></tr><tr><td>CyberPal.AI-Mistral</td><td>89.93</td><td>74.00</td><td>78.11</td><td>81.60</td><td>65.33</td><td>58.20</td><td>42.30</td><td>69.92</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>71.71</td><td>74.00</td><td>82.24</td><td>83.20</td><td>63.28</td><td>41.45</td><td>32.61</td><td>64.07</td></tr><tr><td>CyberPal.AI-Llama</td><td>90.40</td><td>77.00</td><td>86.98</td><td>84.80</td><td>66.41</td><td>60.65</td><td>55.04</td><td>74.47</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>77.27</td><td>78.00</td><td>83.43</td><td>87.20</td><td>65.53</td><td>30.68</td><td>45.36</td><td>66.78</td></tr><tr><td>CyberPal.AI-Phi</td><td>90.40</td><td>80.00</td><td>86.39</td><td>91.00</td><td>72.65</td><td>53.00</td><td>67.47</td><td>77.27</td></tr></table>

Table 3: Evaluation results for CyberPal.AI models compared to the base on public and general cyber-security benchmarks datasets. Although our fine-tuned models were not trained on these types of tasks, they exhibit significant and consistent improvement over baselines.

across different tasks, which include Multi-choice Q&A, Summarization, and classification. Overall, our fine-tuned versions obtained a significant average improvement of 18-24% across all CTI evaluation datasets.

As a notable instance, when testing the model with the Adversarial MITRE evaluation dataset and its non-adversarial version, our fine-tuned CyberPal.AI models demonstrate greater robustness compared to the base models. This finding suggests that CyberPal.AI is more resilient and has successfully generalized to the domain of cyber-security. More specifically, for Mistral, our fine-tuned model exhibits a degradation of 6% in accuracy when tested with the adversarial version of the dataset, compared to Mistral, which exhibits a degradation of 14%. The same goes for Llama, where our fine-tuned model exhibits a degradation of 9% in accuracy with the adversarial version of the dataset, compared to Llama, which shows a degradation of 19%. Lastly, for Phi, our fine-tuned model exhibits a degradation of 7% in accuracy with the adversarial version of the dataset, compared to Phi which shows a degradation of 13%. Overall, these results demonstrate the robust knowledge CyberPal.AI gained during our fine-tuning process. See Appendix B for more details.

Additionally, we compared the performance of CyberPal.AI with seven general and public cyber-security benchmarks to test if our fine-tuned models could improve on different types of public cyber-security benchmarks. In Table 3, we tested seven general and public cyber-security evaluation datasets: the cyber-security section in MMLU (SecMMLU), LinkedIn cyber-security skills assessment test, publicly available CTI evaluation dataset, CTI root cause mapping dataset, and two additional cyber-security benchmarks named CyberMetric and SecEval. We also extracted multiple-choice Q&A questions from the CISSP assessment tests, aimed at validating security analysts' understanding of cyber-security. Overall, our fine-tuned models achieved an average improvement of $9 - 10\%$ across the general cyber-security evaluation datasets. This is impressive, considering that our models were fine-tuned with different kinds of datasets, which also indicates that our fine-tuned models managed to generalize and better understand the domain.

In summary, our evaluation of CyberPal.AI models on diverse data sources, including both proprietary and public datasets, demonstrated significant and consistent enhancements across multiple tasks, such as multiple-choice Q&A, classification, and summarization.

# 6 CONCLUSION

In this work, we introduced SecKnowledge, SecKnowledge-Eval, and CyberPal.AI. SecKnowledge is a domain-knowledge-driven cyber-security instruction dataset aimed at fine-tuning LLMs for the security domain. The dataset construction involves two main steps. In the first step, we create instructions based on predefined schemas established through domain expertise. In the second step, we expand the initial dataset through a hybrid synthetic content-grounded data generation process. CyberPal.AI represents a family of LLMs fine-tuned using SecKnowledge, aimed at developing security-specialized models capable of answering and following complex security-related instructions. To evaluate CyberPal.AI, we introduced SecKnowledge-Eval, a comprehensive suite of evaluation datasets that includes a diverse range of cyber-security tasks we developed, along with other publicly available security benchmarks. This suite is specifically designed to assess the performance of LLMs in the cyber-security domain. Our fine-tuned CyberPal.AI models demonstrated impressive performance on various security-related tasks, including threat hunting (e.g., up to 26% improvement on CTI Detection and Mitigation), TTP mapping (e.g., up to 17% improvement in SIEM Rule TTP mapping), summarization (e.g., up to 35% improvement in CWE Summarization), and impact mapping (e.g., up to 11% improvement in CWE technical impact mapping). Additionally, our models also effectively captured relationships between different components and concepts within various security frameworks (e.g., up to 45% in CTI relationship prediction). CyberPal.AI also demonstrated enhanced performance on general security knowledge benchmarks such as the security portion of the MMLU, skill assessment tests, and analysts' assessment tests, among others. Overall, CyberPal.AI models outperformed their baseline counterparts, achieving significant average improvement of up to 24% on training-aligned tasks and up to 10% average improvement on public cyber-security benchmarks. These results underscore the extensive knowledge and deep understanding gained through fine-tuning the models with our SecKnowledge dataset.

# REFERENCES

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.   
Ehsan Aghaei, Xi Niu, Waseem Shadid, and Ehab Al-Shaer. Securebert: A domain-specific language model for cybersecurity. In International Conference on Security and Privacy in Communication Systems, pp. 39–56. Springer, 2022.   
AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/blob/main/MODEL\_CARD.md.   
Md Tanvirul Alam, Dipkamal Bhushl, Le Nguyen, and Nidhi Rastogi. Ctibench: A benchmark for evaluating llms in cyber threat intelligence. arXiv preprint arXiv:2406.07599, 2024.   
Markus Bayer, Philipp Kuehn, Ramin Shanehsaz, and Christian Reuter. Cysecbert: A domain-adapted language model for the cybersecurity domain. arXiv preprint arXiv:2212.02974, 2022.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pp. 3–14, 2017.   
Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/codealpaca, 2023.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. See https://vicuna.lmsys.org (accessed 14 April 2023), 2(3):6, 2023.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.   
Mohamed Amine Ferrag, Ammar Battah, Norbert Tihanyi, Merouane Debbah, Thierry Lestable, and Lucas C Cordeiro. Securefalcon: The next cyber reasoning system for cyber security. arXiv preprint arXiv:2307.06616, 2023.   
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.   
Erik Hemberg, Jonathan Kelly, Michal Shlapentokh-Rothman, Bryn Reinstadler, Katherine Xu, Nick Rutar, and Una-May O'Reilly. Linking threat tactics, techniques, and patterns with defensive weaknesses, vulnerabilities and affected platform configurations for cyber hunting, 2021.   
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.   
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.   
Wenxiang Jiao, Jen-tse Huang, Wenxuan Wang, Xing Wang, Shuming Shi, and Zhaopeng Tu. Parrot: Translating during chat using large language models. arXiv preprint arXiv:2304.02426, 2023.   
Matan Levi and Aryeh Kontorovich. Splitting the difference on adversarial training. arXiv preprint arXiv:2310.02480, 2023.   
Guancheng Li, Yifeng Li, Wang Guannan, Haoyu Yang, and Yang Yu. Seceval: A comprehensive benchmark for evaluating cybersecurity knowledge of foundation models. https://github.com/XuanwuAI/SecEval, 2023.   
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.   
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.   
Tiedong Liu and Bryan Kian Hsiang Low. Goat: Fine-tuned llama outperforms gpt-4 on arithmetic tasks. arXiv preprint arXiv:2305.14201, 2023.   
Zefang Liu, Jialei Shi, and John F Buford. Cyberbench: A multi-task benchmark for evaluating large language models in cybersecurity.   
Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pp. 22631–22648. PMLR, 2023.   
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.

Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. Orca 2: Teaching small language models how to reason. arXiv preprint arXiv:2311.11045, 2023.   
Marwan Omar and Stavros Shiaeles. Vuldetect: A novel technique for detecting software vulnerabilities using language models. In 2023 IEEE International Conference on Cyber Security and Resilience (CSR), pp. 105–110. IEEE, 2023.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Youngja Park and Weiqiu You. A pretrained language model for cyber threat intelligence. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 113–122, 2023.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
Priyanka Ranade, Aritran Piplai, Anupam Joshi, and Tim Finin. Cybert: Contextualized embeddings for the cybersecurity domain. In 2021 IEEE International Conference on Big Data (Big Data), pp. 3334–3342. IEEE, 2021.   
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.   
Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. Principle-driven self-alignment of language models from scratch with minimal human supervision. Advances in Neural Information Processing Systems, 36, 2024.   
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: An instruction-following llama model, 2023.   
Gemma Team. Gemma. 2024. doi: 10.34740/KAGGLE/M/3301. URL https://www.kaggle.com/m/3301.   
Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen, and Fahad Shahbaz Khan. Xraygpt: Chest radiographs summarization using medical vision-language models. arXiv preprint arXiv:2306.07971, 2023.   
Norbert Tihanyi, Mohamed Amine Ferrag, Ridhi Jain, and Merouane Debbah. Cybermetric: A benchmark dataset for evaluating large language models knowledge in cybersecurity. arXiv preprint arXiv:2402.07688, 2024.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. arXiv preprint arXiv:2212.10560, 2022.   
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023a.   
Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. Baize: An open-source chat model with parameter-efficient tuning on self-chat data. arXiv preprint arXiv:2304.01196, 2023b.   
Hanwei Xu, Yujun Chen, Yulun Du, Nan Shao, Yanggang Wang, Haiyu Li, and Zhilin Yang. Zero-prompt: scaling prompt-based pretraining to 1,000 tasks improves zero-shot generalization. arXiv preprint arXiv:2201.06910, 2022.   
Da Yin, Xiao Liu, Fan Yin, Ming Zhong, Hritik Bansal, Jiawei Han, and Kai-Wei Chang. Dynosaur: A dynamic growth paradigm for instruction-tuning data curation. arXiv preprint arXiv:2305.14327, 2023.   
Yue Zhang, Leyang Cui, Deng Cai, Xinting Huang, Tao Fang, and Wei Bi. Multi-task instruction tuning of llama for specific scenarios: A preliminary study on writing assistance. arXiv preprint arXiv:2305.13225, 2023.

# A MITRE DATASETS IN-DEPTH DESCRIPTION

MITRE ATT&CK: is a comprehensive knowledge base of adversary tactics and techniques based on real-world observations. It provides a common language and structure that enables security practitioners to describe, assess, and respond cyber threats and attacks.

More specifically, MITRE ATT&CK consists of 14 tactics, representing the high-level goals of adversaries, such as lateral movement, initial access, and execution. Each tactic has a set of techniques, which describe the specific methods used to achieve the tactic's goals. For example, the lateral movement tactic includes techniques like internal spear-phishing. Moreover, some techniques are further divided into sub-techniques, providing even greater granularity. The ATT&CK framework also includes information on attack campaigns, adversary groups, and the software tools they commonly use. Lastly, the framework also provides mitigations and detections to defend against the different attack vectors.

CWE: is a community-developed list of software and hardware weakness types, serving as a common language for describing security vulnerabilities. More specifically, weaknesses usually contain relationships to other weaknesses, implementation, affected platforms, consequences, examples (which are linked to CVEs), potential mitigations, and correlation to related attack patterns (which are linked to CAPEC). As with MITRE ATT&CK, the dataset is constructed in such a way that enables us to utilize the connection between different components of each weakness, the connection between weaknesses, and the connections to other frameworks such as CVE, MITRE ATT&CK, and CAPEC.

CVE: is a dictionary of publicly known cyber-security vulnerabilities and exposures, that aims at standardizing the way we share information regarding vulnerabilities. Each CVE typically contains several key pieces of information, including a unique identifier assigned to the vulnerability, description, affected products, severity score, connection to weaknesses (CWEs), etc.

CAPEC: is a structured catalog of common attack patterns that helps users understand how adversaries exploit application weaknesses and other cyber-enabled capabilities. Attack patterns are descriptions of the common attributes and approaches employed by adversaries to exploit known weaknesses in cyber-enabled capabilities. Attack patterns define the challenges that an adversary may face and how they go about solving them. They are derived from the concept of design patterns applied in a destructive rather than constructive context and are generated from in-depth analysis of specific real-world exploit examples. CAPEC provides a structured way to categorize and describe these attack patterns, including information on how they work, attack severity, the likelihood of the attack, relationships with other attack patterns, execution flow, pre-requirements, required resources,

and skills, consequences of the attack, connection to CWEs, attack examples that can be linked to CVEs, and how to mitigate the attack. $^{7}$

# B ADVERSARIAL MULTIPLE-CHOICE QUESTIONS GENERATION PROCESS

To increase the difficulty of multiple-choice questions, we developed a novel adversarial attack targeting closed-domain options, where the choices are drawn from a fixed list. Here's how it works:

1. Assume a multiple-choice question with choices from a closed list of size k.   
2. For each of the k-1 incorrect options, we create a new classification question. This classification question retains the original question but presents only two options: the correct choice and one of the k-1 incorrect options.   
3. We then query a language model (LLM) with each of these $k - 1$ binary classification questions.   
4. From the responses, we identify the incorrect options that the model is most likely to select, given the original question and the correct answer, using the conditional loss on the incorrect option.

The process ensures that the false options selected are those most likely to confuse the model, thereby enhancing the overall difficulty of the dataset. See Figure 8 for an example where we asked a question related to one of the MITRE ATT&CK tactics, and chose the other three tactics from the list of all possible tactics that are the most likely to fool a third-party LLM. The attack is an adversarial transfer attack, as we use Phi-3-small as the reference model (the model we attack), and test the attack results using the adversarial generated dataset on the other CyberPal.AI models.

Note that attacking MITRE ATT&CK tactics requires less computational resources since the list contains only 14 possible options, but on other types of tasks, i.e., technique/software-related tasks, there are hundreds of possible options. Therefore, this attack is time and resource-consuming, but it is done only once, during the generation of the adversarial evaluation dataset.

As for the performance degradation of the models on the adversarial dataset: when testing the model with the Adversarial MITRE evaluation dataset and its non-adversarial version, our fine-tuned CyberPal.AI models demonstrate greater robustness compared to the base models. As can be seen in Table 4, our models experience smaller degradation in results between the adversarial and non-adversarial version of the MITRE ATT&CK dataset: for Mistral, our fine-tuned model exhibits a degradation of $6\%$ in accuracy when tested with the adversarial version of the dataset, compared to the original Mistral model, which exhibits a degradation of $14\%$ . The same goes for Llama, where our fine-tuned model exhibits a degradation of $9\%$ in accuracy when tested with the adversarial version of the dataset, compared to Llama, which shows a degradation of $19\%$ . Lastly, for Phi, our fine-tuned model exhibits a degradation of $7\%$ in accuracy when tested with the adversarial version of the dataset, compared to Phi's base model which shows a degradation of $13\%$ . These results demonstrate the robust knowledge CyberPal.AI gained during our fine-tuning process and suggest that CyberPal.AI is more resilient and has successfully generalized to the domain of cyber-security.

# C SECKNOWLEDGE-EVAL STATISTICS

In Table 5, we provide high-level statistics of the different datasets composing SecKnowledge-Eval.

# D SECKNOWLEDGE: ADDITIONAL DATA GENERATION DETAILS

# D.1 BRON

In addition to the main ideas presented in the paper, we provide a more detailed explanation of the various aspects that were not included initially regarding the BRON dataset generation. We will refer to the stages demonstrated in the article.

![](images/a7d99a111148aa1d266a555c4b4a73e6504126e6ea4e28a85fa84c2b2296e99c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Question: Identify the MITRE tactic described here: The tactic includes methods where adversaries collect data, actively or passively. The information may cover the victim organization, etc. Adversaries use this data to aid in planning Initial Access, prioritizing post-compromise objectives, or guiding further activities."] --> B{Correct answer: Reconnaissance}
    B --> C1["Identify the MITRE tactic described here: The tactic includes methods.... Choose between the following two options: A. Reconnaissance B. Resource Development"]
    B --> C2["Identify the MITRE tactic described here: The tactic includes methods.... Choose between the following two options: A. Reconnaissance B. Privilege Escalation"]
    B --> C3["Identify the MITRE tactic described here: The tactic includes methods.... Choose between the following two options: A. Impact B. Reconnaissance"]
    C1 --> D["Choose 3 wrong options with the lowest conditional log prob + add the correct option"]
    C2 --> D
    C3 --> D
    D --> E["Final question: Identify the MITRE tactic described below. Choose the correct options: The tactic includes methods where adversaries collect data, actively or passively. The information .... Options: A. Exfiltration B. Impact C. Reconnaissance D. Resource development"]
    E --> F["Overall 14 binary classification questions"]
    E --> G["Conditional loss on the incorrect answer"]
```
</details>

Figure 8: Adversarial MITRE ATT&CK generation pipeline example on a question related to MITRE ATT&CK tactics, where there are 14 possible tactics, one is the correct option, and we choose the other three options that were most likely to fool a third-party LLM. 

<table><tr><td>Model</td><td>Original MITRE ATTACK</td><td>Adversarial MITRE ATTACK</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>73.24</td><td>59.57 (-13.67)</td></tr><tr><td>Sec-Mistral (Ours)</td><td>98.87</td><td>92.54 (-6.3)</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>78.59</td><td>59.57 (-19.0)</td></tr><tr><td>Sec-Llama (Ours)</td><td>97.04</td><td>87.74 (-9.3)</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>77.32</td><td>64.50 (-12.8)</td></tr><tr><td>Sec-Phi-3-medium (Ours)</td><td>96.76</td><td>89.57 (-7.1)</td></tr></table>

Table 4: Models' results before and after applying our adversarial attack to generate the adversarial multiple-choice dataset. The “Original MITRE ATTACK” column presents the evaluation dataset results prior to the application of our adversarial method. The “Adversarial MITRE ATTACK” column shows the results after applying our adversarial technique. It is evident that CyberPal.AI models exhibit greater robustness to adversarial changes, with their results showing less drastic variation compared to those of the non-security models.

EXPENDING BRON The BRON Knowledge Graph (KG) consists of the following data sources: MITRE ATT&CK, CAPEC, CWE, CVE, MITRE Engage, MITRE D3FEND, MITRE CAR, and, exploitdb. We've extended BRON and added additional available information. Specifically, we've added the following information: Descriptions for technique mitigations (source: MITRE ATT&CK), Connections between techniques and their mitigations (source: MITRE ATT&CK), Descriptions for D3FEND mitigations (source: MITRE D3FEND), Descriptions for the relationships between Software and Technique (source: MITRE ATT&CK), Enhanced CAPEC descriptions, including those with minimal or missing information (source: MITRE ATT&CK). We've added the

Table 5: SecKnowledge-Eval statistics 

<table><tr><td>Eval Dataset</td><td>task type</td><td># of questions</td></tr><tr><td>Adversarial MITRE ATT&amp;CK (Ours)</td><td>MCQA</td><td>710</td></tr><tr><td>SIEM Rule TTP Mapping (Ours)</td><td>MCQA</td><td>146</td></tr><tr><td>CTI Detection and Mitigation Mapping (Ours)</td><td>MCQA</td><td>1012</td></tr><tr><td>CWE Technical Impact Mapping (Ours)</td><td>MCQA</td><td>349</td></tr><tr><td>CISSP Assessment Questions (Ours)</td><td>MCQA</td><td>206</td></tr><tr><td>SecMMLU</td><td>MCQA</td><td>100</td></tr><tr><td>CyberMetric</td><td>MCQA</td><td>500</td></tr><tr><td>CTI-MCQ</td><td>MCQA</td><td>2500</td></tr><tr><td>SecEval</td><td>MCQA</td><td>2189</td></tr><tr><td>Cybersecurity Skill Assessment</td><td>MCQA</td><td>170</td></tr><tr><td>CTI-RCM</td><td>Classification</td><td>2000</td></tr><tr><td>CTI relationship prediction (Ours)</td><td>Classification</td><td>778</td></tr><tr><td>CTI Entity Classification (Ours)</td><td>Classification</td><td>1983</td></tr><tr><td>CWE Description Summarization (Ours)</td><td>Summarization</td><td>92</td></tr></table>

entities of the new information and managed to construct the connections between entities by using the structured nature of the MITRE data sources.

PATHS EXTRACTION We ensured that longer paths do not include shorter ones. This is essential because we want to avoid generating similar questions and prefer unique paths to represent as many different routes as possible.

In this stage, we also gathered node pairs for negative sampling. For nodes of directly connected types, we observed that it can sometimes be quite easy to determine whether they are related, as their descriptions may be completely unrelated. To make the questions more challenging, for each node, we randomly sampled 100 nodes of the second type that are not connected to the original node. From these, we used a keywords database, and a classifier to extract features about the descriptions, to identify the node with the most similar description.

COT ON PATHS As explained in the paper, we needed an explanation for each edge in the path to provide a complete understanding. Although the primary approach was to generate explanations using LLMs, we tried to rely as much as possible on existing knowledge. For some edge types, we had sources that explained the connections between nodes. For example, we used the MITRE API to find explanations for the connections between specific Software and Techniques (see Figure 9).

![](images/51b85c36a708e7e11b30ac2341ba58bba88a4b502b941a0ce23fbc689eebad89.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Software S0001 &quot;Trojan.Mebromi&quot;"]
    B["Trojan.Mebromi performs BIOS modification and can download and execute a file as well as protect itself from removal. (Citation: Ge 2011)"]
    C["Technique T1542.001 &quot;System Firmware&quot;"]
    D["Adversaries may modify system firmware to persist on systems. The BIOS (Basic Input/Output System) and The Unified Extensible Firmware Interface (UEFI) or Extensible Firmware Interface (EFI) are examples of system firmware that operate as the software interface between the operating system and hardware of a computer; System firmware like BIOS and (UEFI) underly the functionality of a computer and may be modified by an adversary to perform or assist in malicious activity. Capabilities exist to overwrite the system firmware, which may give sophisticated adversaries a means to install malicious firmware updates as a means of persistence on a system that may be difficult to detect."] --> E["End"]
```
</details>

Figure 9: BRON example of existing knowledge fetched from MITRE API to explain the relation between a specific software and technique.

DEFINE INSTRUCTION SCHEMAS BRON can be viewed as a Directed Acyclic Graph (DAG), where nodes are categorized by type (e.g., CAPEC, CWE). Within each node type, there are direct

connections (e.g., Technique to Sub-Technique) and connections to other node types (e.g., Tactic to Technique). Our goal is to enable the model to learn and understand the relationships between these different node types, which correspond to distinct datasets. See Figure 10 for the generation process overview.

![](images/21c6a99e0799db374d92aa5786c0a61961b8514f97d5240a647d516d2b92dafe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Extract Path"] --> B["Explain e"]
    B --> C{Has information on edge?}
    C -->|True| D["Use information on edge"]
    C -->|False| E["Prompt LLM (e.g. Mixtral) for explanation based on the descriptions of the nodes and the relationship between the layers"]
    D --> F["Edge Prefix"]
    E --> F
    F --> G["Transition Explanation"]
    G --> H["Path prefixes"]
    H --> I["Transitions Explanations"]
    I --> J["Path suffix"]
    J --> K["Path Explanation"]
```
</details>

Figure 10: BRON path explanation generation process being used in explaining the selected paths. This iterative process is used heavily to construct answers that involve path explanations

After expending BRON, we define five different families of instructions:

1. Direct Node to node - Given two consecutive nodes and their descriptions, describe the relation between the nodes. See Figure 11 for an example.   
2. Indirect Node to node - Define the path/connection between 2 specific nodes. See Figure 12 for an example.   
3. Node type to specific node (and vice versa) – Describe the connection between a node type, to a specific node of another node type. See Figure 13 for an example.   
4. Node type to node type - Describe the connection between different node types using examples. See Figure 14 for an example.   
5. two-step detection/mitigation – Describe the connection between two nodes, and how to mitigate/detect the entity represented by the destination node. See 4 for an example.

# Instruction

Explore the information presented in the CAPEC (Common Attack Pattern Enumeration and Classification) and CWE (Common Weakness Enumeration) below:

CAPEC-65 (Sniff Application Code):

An adversary passively sniffs network communications and captures application code bound for an authorized client. Once obtained, they can use it as-is, or through reverse-engineering glean sensitive information or exploit the trust relationship between the client and server. Such code may belong to a dynamic update to the client, a patch being applied to a client component or any such interaction where the client is authorized to communicate with the server.

Protection Mechanism Failure (CWE-693):

The product does not use or incorrectly uses a protection mechanism that provides sufficient defense against directed attacks against the product.

Can you identify any underlying association between them? Please expound upon your analysis and discuss the implications of any correlation inferred.

# Answer

The attack pattern exploits the weakness by intercepting and capturing sensitive application code transmitted over the network, which is only possible because the protection mechanism, intended to safeguard the code, has failed. This failure allows the adversary to access and potentially reverse-engineer the code, leading to unauthorized access to sensitive information or exploitation of the trust relationship between the client and server.

Figure 11: BRON example of direct connection instruction and its output.

# Instruction

Could you outline the association between malware Winnti for Linux (S0430) and application inventory\_management\_system (by inventory\_management\_system\_project) version 1.0? I would appreciate it if you could delve into the details of each link and share your thought process.

# Answer

The connection of S0430 (malware Winnti for Linux) to application inventory\_management\_system (by inventory\_management\_system\_project) version 1.0 is possible through the following reasoning:

1. S0430 (malware Winnti for Linux) is associated with T1014 (Rootkit)

2. T1014 (Rootkit) is associated with Install Rootkit (CAPEC-552)

3. Install Rootkit (CAPEC-552) is associated with Improper Access Control (CWE-284)

4. Improper Access Control (CWE-284) is associated with CVE-2023-4183

5. CVE-2023-4183 is associated with application inventory\_management\_system (by inventory\_management\_system\_project) version 1.0

# Here is why:

1. [Winnti for Linux](https://attack.mitre.org/software/S0430) has used a modified copy of the open-source userland rootkit Azazel, named libxselinux.so, to hide the malware's operations and network activity.(Citation: Chronicle Winnti for Linux May 2019)

2. The attack pattern "Install Rootkit" uses the "Rootkit" technique by exploiting a weakness in authentication to install malware that alters operating system API calls, thereby hiding the presence of malicious system components.

3. The attack pattern exploits the weakness by taking advantage of the lack of proper access control to gain unauthorized access to the system, allowing the installation of malware (rootkit) that can alter operating system API calls, hide system components, and conceal malicious activities.

4. In CVE-2023-4183, the weakness of Improper Access Control (CWE-284) is exploited by manipulating the "user\_id" argument in the Password Handler's edit\_update.php file, allowing unauthorized remote access to restricted resources.

5. The Inventory Management System 1.0 is affected by CVE-2023-4183, which allows remote attackers to exploit improper access controls in the Password Handler component, specifically in the edit\_update.php file, by manipulating the user\_id argument.

To summarize, the linkage of S0430 (malware Winnti for Linux) to application inventory\_management\_system (by inventory\_management\_system\_project) version 1.0 is underscored by the reasoning expounded upon.

Figure 12: BRON example of node to node CoT instruction and its answer. The process used to generate the path explanation step is described in Figure 10

# E ADDITIONAL EVALUATION RESULTS

Here, we provide results for Google's Gemma-2 series of models. More specifically, we test the Gemma-2-2b and Gemma-2-9b models. We note that for Gemma models, specifically for Gemma-2-2b, we had to use 4-shot settings in SecEval to achieve good results. See Tables 6 and 7 for results.

<table><tr><td>Model</td><td>Original/Adv. MITRE ATTACK</td><td>SIEM Rule TTP Mapping</td><td>CTI Detection and Mitigation</td><td>CWE Summarization (R-1/2/L)</td><td>Technical Impact Mapping</td><td>CTI Relationship Prediction</td><td>CTI Entity Classification</td><td>Avg.</td></tr><tr><td>Gemma-2-2b</td><td>72.25/56.48</td><td>47.94</td><td>49.60</td><td>23.51/5.30/15.87</td><td>55.87</td><td>56.94</td><td>58.70</td><td>48.63</td></tr><tr><td>CyberPal.AI-Gemma-2b</td><td>94.64/81.97</td><td>60.27</td><td>61.66</td><td>35.81/43.12/44.40</td><td>61.32</td><td>95.11</td><td>78.62</td><td>68.58</td></tr><tr><td>Gemma-2-9b</td><td>77.46/66.62</td><td>47.26</td><td>61.46</td><td>23.95/6.10/16.60</td><td>64.18</td><td>70.43</td><td>68.28</td><td>56.25</td></tr><tr><td>CyberPal.AI-Gemma-9b</td><td>96.62/90.70</td><td>69.20</td><td>71.44</td><td>45.59/38.63/43.29</td><td>65.32</td><td>97.30</td><td>83.35</td><td>74.26</td></tr></table>

Table 6: Evaluation results for Gemma-2 and CyberPal.AI fine-tuned models compared to the base model on designated datasets constructed to evaluate the models' performance on training-aligned security tasks. For the MITRE ATT&CK evaluation set, we provide results for both the original evaluation set and its adversarial version, where we can see that CyberPal.AI demonstrates greater robustness.

# Instruction

How can you mitigate Insufficient Logging (CWE-778)?

# Answer

To mitigate CWE-778 (Insufficient Logging), try the following approaches:

- Use a centralized logging mechanism that supports multiple levels of detail.   
- Ensure that all security-related successes and failures can be logged. When storing data in the cloud (e.g., AWS S3 buckets, Azure blobs, Google Cloud Storage, etc.), use the provider's controls to enable and capture detailed logging information.   
- Be sure to set the level of logging appropriately in a production environment. Sufficient data should be logged to enable system administrators to detect attacks, diagnose errors, and recover from attacks. At the same time, logging too much data (CWE-779) can cause the same problems, including unexpected costs when using a cloud environment.   
- To enable storage logging using Azure's Portal, navigate to the name of the Storage Account, locate Monitoring (CLASSIC) section, and select Diagnostic settings (classic). For each of the various properties (blob, file, table, queue), ensure the status is properly set for the desired logging data. If using PowerShell, the Set-AzStorageServiceLoggingProperty command could be called using appropriate -ServiceType, -LoggingOperations, and -RetentionDays arguments.

By implementing these strategies, you can effectively mitigate CWE-778 (Insufficient Logging).

Figure 13: BRON example of direct detection/mitigation instruction and its output. Note that in this case, the data was not generated by LLM. Also, here the source node is CWE-778, and the destination node type is CWE mitigation.

# Instruction

What are 2 cases that might link MITRE ATT&CK tactic and CAPEC (Common Attack Pattern Enumeration and Classification)? Can you provide a detailed of these links, including their relevance and importance to the overall structure?

# Answer

No doubt about it! Right here, you'll see 2 connections originating at MITRE ATT&CK tactic and guiding you to CAPEC (Common Attack Pattern Enumeration and Classification):

# Below is path 1:

Exploring the link between defense-evasion (TA0005) and Signature Spoof (CAPEC-473), we find the

path:

[Path 1 explanation as in node to node]...

# Path 2:

Examining the correlation between persistence (TA0003) and Run Software at Logon (CAPEC-564), we identify the pathway:

persistence (TA0003) => Boot or Logon Initialization Scripts (T1037) => Run Software at Logon (CAPEC-564)

Allow me to provide you with a detailed explanation of the underlying logic of the relations above: [Path 2 explanation as in node to node]...

In essence, the tie between persistence (TA0003) and Run Software at Logon (CAPEC-564) is elucidated by the logic presented.

Displayed above are some 2 paths between MITRE ATT&CK tactic and CAPEC (Common Attack Pattern Enumeration and Classification), providing context for their association.

Figure 14: BRON example of node type to node type instruction and its output. 

<table><tr><td>Model</td><td>CISSP Assessment</td><td>SecMMLU</td><td>Cybersecurity Skill Assessment</td><td>CyberMetric</td><td>CTI-MCQ</td><td>CTI-RCM</td><td>SecEval</td><td>Avg.</td></tr><tr><td>Gemma-2-2b</td><td>61.62</td><td>65.00</td><td>77.51</td><td>77.60</td><td>53.79</td><td>3.11</td><td>44.90</td><td>54.79</td></tr><tr><td>CyberPal.AI-Gemma-2b</td><td>75.76</td><td>69.00</td><td>78.10</td><td>78.80</td><td>61.96</td><td>49.45</td><td>46.27</td><td>65.62</td></tr><tr><td>Gemma-2-9b</td><td>77.78</td><td>79.00</td><td>85.20</td><td>86.60</td><td>62.80</td><td>53.35</td><td>62.99</td><td>72.53</td></tr><tr><td>CyberPal.AI-Gemma-9b</td><td>89.40</td><td>79.00</td><td>85.79</td><td>88.00</td><td>68.74</td><td>62.80</td><td>64.00</td><td>76.82</td></tr></table>

Table 7: Evaluation results for Gemma-2 and CyberPal.AI fine-models models compared to the base on public and general cyber-security benchmarks datasets. Although our models were not trained on these tasks, they exhibit significant and consistent improvement.