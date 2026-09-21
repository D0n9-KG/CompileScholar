# Multi-431 词表人工审包件（预注册 S1 用户闸门）

生成：确定性抽样，零 LLM。数据源：registry.json（2659 实体）+ dim_vocab_v1.json + 两份 QC。


## 1. Registry（实体归一）

- 实体 2659（{'out_of_corpus': 487, 'method': 1742, 'mechanism': 281, 'practice': 148, 'field': 1}）｜多别名实体 492｜surface_index 3779
- 合并账：round1 374 + 恢复 692 + 跨块 [434, 14]｜守卫拦截：cross [24, 36]，recovery {'affix_extension': 13, 'short_token': 4}｜坍缩组 3｜通道边 {'acronym': 579, 'inclusion': 1061}

### 1a. Top-15 高提及合并（错并影响最大，重点读）
  - [cs_nlp|n=13|method] **GPT-3** ← ['GPT-3', 'GPT-3 13B', 'GPT-3 175B', 'GPT-3 2.7B', 'GPT-3 6.7B', 'GPT-3 Large']…
  - [cs_nlp|n=12|method] **RLHF** ← ['RLHF', 'Reinforcement Learning from Human Feedback', 'iterated online RLHF', 'preference modeling and reinforcement learning from human feedback', 'reinforcement learning from human feedback (RLHF)']
  - [cs_nlp|n=10|method] **Text-to-Text Transfer Transformer** ← ['T5', 'Text-to-Text Transfer Transformer', 'text-to-text framework']
  - [photonics|n=9|method] **3D finite-difference time domain (FDTD) simulation** ← ['3D finite-difference time domain (FDTD) simulation', 'FDTD numerically simulated value', 'FDTD-analysis', 'finite-difference time-domain', 'finite-difference time-domain (FDTD) algorithm', 'finite-difference time-domain (FDTD) method']…
  - [cs_nlp|n=9|method] **LoRA** ← ['LoRA', 'Low Rank Adaptation (LoRA)']
  - [cs_nlp|n=8|method] **retrieval-augmented LMs** ← ['RAG', 'Retrieval-Augmented Generation', 'Retrieval-Augmented Generation (RAG)', 'retrieval-augmented LMs', 'retrieval-augmented language models']
  - [cs_nlp|n=8|method] **distillation** ← ['Distillation', 'Knowledge Distillation', 'learning a small acoustic model by matching the class probabilities of an already trained larger model', 'model distillation', 'soft targets']
  - [cs_nlp|n=8|method] **chain-of-thought prompting** ← ['Chain-of-Thought (CoT)', 'Chain-of-Thought (CoT) prompting', 'chain-of-thought', 'chain-of-thought prompting']
  - [photonics|n=8|method] **optical WGM microcavity** ← ['Whispering-gallery mode (WGM) resonators', 'optical WGM microcavity', 'whispering gallery mode (WGM) microresonators', 'whispering gallery mode (WGM) resonance', 'whispering gallery mode resonators', 'whispering gallery resonators']…
  - [bio|n=7|method] **Phage display technology** ← ['antibody phage display', 'phage display', 'phage display technology']
  - [photonics|n=7|mechanism] **Topological Nature of Optical Bound States in the Continuum** ← ['BICs', 'Topological Nature of Optical Bound States in the Continuum', 'bound state in the continuum (BIC)', 'bound states in the continuum', 'bound states in the continuum (BIC)', 'bound states in the continuum (BICs)']…
  - [cs_nlp|n=7|method] **sentence-BERT** ← ['SBERT', 'sentence transformers', 'sentence-BERT']
  - [cs_nlp|n=7|method] **Llama-2** ← ['Llama 2', 'Llama-2']
  - [photonics|n=6|method] **COMSOL** ← ['COMSOL', 'COMSOL Multiphysics', 'COMSOL simulation']
  - [photonics|n=6|mechanism] **Purcell effect** ← ['Purcell effect', 'Purcell factor']

### 1b. 随机 15
  - [photonics|n=1|method] **Whispering-gallery-mode microdisk lasers produced by femtosecond laser direct writing** ← ['FsLDW', 'WGM microdisk lasers', 'Whispering-gallery-mode microdisk lasers produced by femtosecond laser direct writing', 'femtosecond laser direct writing', 'femtosecond laser direct writing (FsLDW)']
  - [photonics|n=1|method] **Leaky-mode assisted fluorescence extraction** ← ['Leaky-mode assisted fluorescence extraction', 'enhanced extraction effect', 'fluorescence enhancement biosensors']
  - [bio|n=1|mechanism] **inflammation exacerbation (IE)** ← ['IE', 'LPS and LNP-mediated IE', 'inflammation exacerbation (IE)', 'inflammation-exacerbation (IE)']
  - [physics|n=2|out_of_corpus] **Pierre Auger cosmic ray observatory** ← ['Pierre Auger Observatory', 'Pierre Auger cosmic ray observatory']
  - [cs_nlp|n=1|method] **CoNT** ← ['CoNT', 'Contrastive Neural Text generation framework']
  - [photonics|n=2|method] **Duan-Lukin-Cirac-Zoller (DLCZ) protocol** ← ['Duan-Lukin-Cirac-Zoller (DLCZ) protocol', 'Duan–Lukin–Cirac–Zoller protocol']
  - [biophysics|n=1|out_of_corpus] **Speckle Variance OCT (SV-OCT)** ← ['Speckle Variance OCT (SV-OCT)', 'Speckle Variance-OCT (SVOCT)']
  - [cs_nlp|n=4|out_of_corpus] **The Pile** ← ['Pile', 'The Pile']
  - [physics|n=1|practice] **GWTC-1** ← ['GWTC-1', 'Gravitational-Wave Transient Catalog']
  - [cs_nlp|n=2|mechanism] **double descent** ← ['double descent', 'double-descent']
  - [cs_nlp|n=4|method] **DINO** ← ['DINO', 'self-distillation with no labels']
  - [photonics|n=1|method] **DLCZ protocol** ← ['DLCZ', 'Duan, Lukin, Cirac and Zoller', 'photon-phonon quantum interface']
  - [cs_nlp|n=6|method] **kNN-LM** ← ['Nearest Neighbor Language Models', 'k-Nearest Neighbor Language Model (Khandelwal et al., 2019) (kNN-LM)', 'kNN LM', 'kNN-LM', 'kNN-LMs']
  - [cs_nlp|n=1|method] **Teacher-free Knowledge Distillation** ← ['Teacher-free Knowledge Distillation', 'Tf-KD', 'Tf-KD_reg', 'Tf-KD_self']
  - [cs_nlp|n=1|method] **ProMoT** ← ['ProMoT', 'ProMoT (Ours)', 'Prompt Tuning with MOdel Tuning']

### 1c. bio/photonics/biophysics 定向 12（用户点名观察点：域覆盖）
  - [photonics|n=1|method] **DENIS** ← ['DENIS', 'DENIS device', 'DENIS platform', 'DENIS system', 'portable digital nanoparticle-enhanced plasmonic imager']
  - [bio|n=1|method] **Light chain LC–MS method** ← ['Light chain LC–MS method', 'MS based methods', 'light chain liquid chromatography–mass spectrometry (LC–MS) method', 'marker peptide method', 'mass spectrometry (MS) based techniques']
  - [biophysics|n=1|method] **biomimetic tumor tissue phantom** ← ['biomimetic tumor tissue phantom', 'phantom A', 'phantom B', 'tumor-mimicking phantom']
  - [photonics|n=1|method] **capped adiabatic tapered fibers** ← ['SU8 capped tapered fiber', 'capped adiabatic tapered fibers', 'capped, terminating, tapered fibers', 'clad fiber']
  - [bio|n=1|method] **Indomethacin-loaded PLGA nanoparticles and Nanostructured Lipid Carriers (NLC)** ← ['Indomethacin-loaded PLGA nanoparticles and Nanostructured Lipid Carriers (NLC)', 'NLC', 'PLGA-NP', 'Poly (DL-lactic-co-glycolic acid) nanoparticles', 'nanostructured lipid carriers']
  - [photonics|n=1|method] **on-chip platform with integrated force sensors and actuators** ← ['monolithic on-chip platform', 'on-chip platform with integrated force sensors and actuators', 'silicon nanomechanical components with arrays of T-shaped protrusions']
  - [photonics|n=5|method] **Measurement of the Instantaneous Velocity of a Brownian Particle** ← ['Measurement of the Instantaneous Velocity of a Brownian Particle', 'counterpropagating dual-beam optical tweezers', 'dual-beam counter-propagating optical trap', 'dual-beam optical tweezer', 'optical tweezer', 'optical tweezers']
  - [photonics|n=1|method] **Pulsed excitation dynamics of an optomechanical crystal resonator** ← ['Pulsed excitation dynamics of an optomechanical crystal resonator', 'phonon counting', 'pulsed optical excitation', 'single-phonon-counting techniques']
  - [photonics|n=1|mechanism] **non-Hermitian edge burst** ← ['edge burst', 'non-Hermitian edge burst', 'non-Hermitian edge burst in quantum dynamics']
  - [photonics|n=3|mechanism] **High-Q supercavity modes in subwavelength dielectric resonators** ← ['BIC', 'High-Q supercavity modes in subwavelength dielectric resonators', 'supercavity modes']
  - [photonics|n=1|method] **Optically Levitated Nanodumbbell Torsion Balance** ← ['GHz Nanomechanical Rotor', 'Optically Levitated Nanodumbbell Torsion Balance', 'levitated nanodumbbell', 'nanodumbbell torsion balance']
  - [biophysics|n=1|method] **Real-Time Imaging of Mitochondrial ATP Dynamics** ← ['ATP live cell imaging', 'Real-Time Imaging of Mitochondrial ATP Dynamics', 'live cell imaging of mitochondrial ATP dynamics', 'mitochondrial ATP imaging approach']

### 1d. 坍缩组（同 entity_id 多组，卡片 alias 质量问题）
  - ['J0 null technique', 'J0 null technique', 'J0 null technique']
  - ['ADM', 'ADM', 'ADM', 'ADM']
  - ['viscous Euler fluid-particle system', 'viscous Euler fluid-particle system', 'viscous Euler fluid-particle system', 'viscous Euler fluid-particle system']

### 1e. 自查旗标（Claude 抽读发现，待用户裁定处置）
  1. 'phantom A'/'phantom B' 并入同一实体（论文中两个不同试样——真错误，来源=卡片 alias 塞爆）
  2. 'optical tweezer'（通用技术）被并入某具体论文实体（泛称被狭义吸收）
  3. 'BIC' 同时出现在两个实体的 alias 表（supercavity 与 Topological-BIC）——surface_index 后写覆盖
  4. GPT-3 尺寸变体（13B/175B/2.7B/6.7B）全并入 GPT-3 家族——alias 保留表面名，记录级可分辨，待定夺
  5. DLCZ protocol 分裂成两个实体（漏合，~6% 残留类）
  6. 恢复通道 692 合并（22% 接受率 vs 遗产 8%）——建议抽读下方 variant/subject 家族样本代偿
  7. 'Shokri et al. (2017)' 类引文残渣成为 out_of_corpus 实体（无害，broker 阶段可过滤）

## 2. Vocab（维度词表）

- subject: 候选 2251 → 批 14，跨批合并 117
- setup: 候选 3786 → 批 20，跨批合并 186
- variant: 候选 3504 → 批 18，跨批合并 49
- hyperparam_item: 候选 1346 → 批 12，跨批合并 64
- 落盘：subject 家族 1381｜setup 3156｜variant 3152｜hyperparam 971
- 合规事件 412：{'unusable_groups_shape': 228, 'empty_response': 3, 'final_shortfall': 1, 'shape_abort_downgraded': 180}
- 确定性清理：{'setup_removed_budget_like': 28, 'setup_removed_repeats_like': 0}｜仲裁队列旗标：family 453 / type 66

### 2a. subject 最大 12 家族（跨域归组质量代表）
  - **Other**（118）: ['1.14 ± 0.03 μm PS particles', '4A3-SC8-based LNPs', 'AaET', 'AaET-NL', 'AaLT', 'AaLT-NL']
  - **Molecule/Chemical**（29）: ['25-hydroxyvitamin D', '[Ru(dpp)3]Cl2', 'acetamiprid', 'Acetylcholine (ACh)', 'adenosine', 'amine']
  - **Nanoparticles**（28）: ['albumin nanoparticles', 'composite nanoparticles', 'copper-based nanoparticles', 'drug nanocarriers', 'magnetic NPs', 'nanoemulsions']
  - **Optical Component**（21）: ['1D nanobeam OMC cavities', '1D-periodic structure in air', 'AAO membrane', 'AlN-on-oxide silicon microchips', 'AuNS-array-on-waveguide hybrid structure', 'AuNS–PCGR hybrid']
  - **Receptors**（18）: ['canine tyrosine-protein kinase receptor CD117 (c-Kit)', 'CCR4', 'CD146', 'CD22', 'CD36', 'CD44']
  - **Cells**（14）: ['293 F eukaryotic cells', 'A431', 'B cells', 'BMSCs', 'C2C12 myoblast cancer cells', 'CHO cells']
  - **Antibodies**（14）: ['anti-CD30 antibody', "anti-EGFR antibody fragment (Fab')", 'Antibody 1', 'Antibody 2', 'antibody light chains', 'humanized IgG4']
  - **Proteins**（14）: ['bRBD:His', 'GFP', 'GST-ACLyz', 'His:bN', 'LacZ', 'MBP-FumI fusion protein']
  - **mechanical resonator**（14）: ['mechanical resonator', 'SAW resonator', 'hBAR', 'mechanical oscillator', 'nanomechanical resonator', 'nanomechanical resonators']
  - **Photonic Crystal (PC)**（14）: ['cavity-coupled PC', 'InGaAs/InP 2-D photonic crystals', 'inverse opal PCs', 'PC slab with circular holes', 'PC slab with equilateral triangular air holes', 'PhC samples']
  - **WMT**（13）: ['EnDe', 'EnFr', 'EnRo', 'WMT development set', 'WMT test sets', "WMT'14 En-De"]
  - **Device/Instrument**（12）: ['Biotage/454 Life Sciences pyrosequencers', 'Cepheid GeneXpert®', 'CombiMatrix Corp.', 'Directif Diagnostic Solutions', 'GeneOhm ePlex', 'Idaho Technology Inc. LightScanner']

### 2b. subject 随机 12 家族
  - **SCIERC**（1）: ['SCIERC']
  - **solid lipid nanoparticles (SLNs)**（1）: ['solid lipid nanoparticles (SLNs)']
  - **silica nanoparticles**（10）: ['silica nanoparticles', 'SiO2 nanoparticles', '2.5 nm silica nanoparticles', 'biotinylated silica nanoparticles', 'silica nanobeads', 'silica nanosphere']
  - **Belly**（1）: ['belly']
  - **Curation Corpus**（1）: ['Curation Corpus']
  - **SimpleQA**（1）: ['SimpleQA verified']
  - **Phantom A**（1）: ['phantom A']
  - **Surface Plasmon Resonance Sensors**（3）: ['SPR (grating coupled)', 'SPR (prism coupled)', 'Surface plasmon resonance']
  - **ChessPieces**（1）: ['ChessPieces']
  - **Ru Dye**（1）: ['Ru dye']
  - **Cellular Behavior**（1）: ['cellular behavior']
  - **BT-20**（1）: ['BT-20']

### 2c. setup 随机 10 条（规范名+别名）
  - barotropic equations of state ← ['barotropic equations of state']
  - polyadenylate polymerase (PAP) ← ['polyadenylate polymerase (PAP)']
  - free-living, unsupervised setting ← ['free-living, unsupervised setting']
  - central serous chorioretinopathy ← ['central serous chorioretinopathy']
  - binary text classification ← ['binary text classification']
  - Z score cut-off of 2.0 ← ['Z score cut-off of 2.0']
  - multicenter, open-label phase 2a trial ← ['multicenter, open-label phase 2a trial']
  - sample thickness of 180 μm ← ['sample thickness of 180 μm']
  - NA = 0.95 ← ['NA = 0.95']
  - 340 nm-thickness Si-layer ← ['340 nm-thickness Si-layer', '340nm thick waveguide']

### 2d. variant 随机 10 条（含 family_hints）
  - ToolAlpaca ← ['ToolAlpaca'] | hints []
  - LoraRetriever ← ['LoraRetriever'] | hints ['Model MoErging']
  - Regular ← ['Regular'] | hints ['context-aware decoding']
  - HESE Gold ← ['HESE Gold'] | hints ['IceCube Event Catalog of Alert Tracks']
  - Iron oxide nanoparticle ← ['Iron oxide nanoparticle', 'iron oxide nanoparticles'] | hints ['spherical inclusion phantom']
  - WHP ← ['WHP'] | hints ['MUSE']
  - Logit ensemble ← ['Logit ensemble'] | hints ['AdapterSoup']
  - Label prob. ← ['Label prob.'] | hints ['Verbalized Confidence Elicitation']
  - metal bottom reflector ← ['metal bottom reflector'] | hints ['Polarization-Splitting Grating Coupler on Lithium Niobate Thin Film']
  - LLaVA ← ['LLaVA'] | hints []

### 2e. hyperparam 随机 8 条
  - K_beta ← ['K_beta']
  - PVA molecular weight ← ['PVA molecular weight']
  - K_AnsAug ← ['K_AnsAug']
  - resonator diameter ΔD ← ['resonator diameter ΔD']
  - d (strip length) ← ['d (strip length)']
  - trapping power ← ['trapping power']
  - grating period ← ['grating period', 'grating period (Λ)', 'grating period Λ']
  - β_ej ← ['β_ej']

### 2f. suspect_family_merge 抽样 10/453（家族名与成员词面不重叠的旗标）
  - family='Other' member='chimeric BR96'
  - family='10M Model' member='180M'
  - family='Bacteria' member='E. coli proteome'
  - family='Other' member='EP14rec'
  - family='Other' member='PyLPC/RMC'
  - family='Other' member='DOPC bilayers'
  - family='Other' member='AaET'
  - family='Antibodies' member='humanized IgG4'
  - family='Receptors' member='CD22'
  - family='mRNA-LNP' member='MC3-based LNPs'

## 3. Blocklist 扩域提案（540 条，待用户批）

### [bio] 5 条
  - CHEMPROT（Named chemical-protein interaction database）
  - NRHybSur3dq8（Named reference genome assembly.）
  - RACE-h（Named RNA-seq benchmark）
  - RACE-m（Named RNA-seq benchmark）
  - ncbi（Major biological database）
### [cs] 533 条
  - 3D-Chat（Named 3D benchmark）
  - 3D-TD（Named 3D benchmark）
  - ACE2004（Named dataset (ACE)）
  - AGIEval（Named evaluation benchmark）
  - AGIE Gaokao Math（Named benchmark subset）
  - AGIE SAT Math（Named benchmark subset）
  - AI2D（Named dataset (AI2D)）
  - AIDA（Named dataset (AIDA)）
  - AIME25（Named benchmark (AIME)）
  - AIR benchmark 24.04（Named benchmark version）
  - AMR2.0（Named benchmark version）
  - ANLI Round 3（Named benchmark subset）
  - ActivityNetQA（Named video QA dataset）
  - AdvBench（Named safety benchmark）
  - AerialMaritimeDrone(tiled)（Named dataset split）
  - AmazonFood（Named dataset (AmazonFood)）
  - AmbigNQ（Named QA dataset）
  - AmericanSignLanguageLetters（Named dataset (ASL)）
  - Ape210k（Named dataset (Ape210k)）
  - ArXiv（Named corpus/database）
  - Avocado dataset（Named dataset (Avocado)）
  - BBH（Named benchmark (BBH)）
  - BCCD（Named dataset (BCCD)）
  - BEC-Pro（Named benchmark (BEC-Pro)）
  - BEIR（Named benchmark suite）
  - BEIR benchmark（Named benchmark suite）
  - BFCL（Named benchmark (BFCL)）
  - BOLD（Named dataset (BOLD)）
  - BOOKS（Named dataset (BOOKS)）
  - BioASQ-Y/N（Named QA benchmark subset）
  - Bot Adversarial Dialogues（Named adversarial dialogue dataset）
  - BrackishUnderwater（Named image/video dataset）
  - CINIC-10（Named image classification benchmark）
  - COCO2017-val（Named image dataset split）
  - COM2SENSE（Named NLP benchmark）
  - CWEB（Named web corpus/dataset）
  - ChartQA（Named chart QA benchmark）
  - Chat Benchmark（Named chatbot benchmark）
  - ChessPieces（Named image dataset）
  - Chinese hotel reviews（Named review dataset）
  - …余 493 条见 blocklist_proposals.json
### [physics] 2 条
  - IceCube data（Named experimental dataset）
  - IceCube-170922A（Named specific event/data）
