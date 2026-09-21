# Leveraging Semiconducting Polymer Nanoparticles for Combination Cancer Immunotherapy

Jiayan Wu and Kanyi Pu*

Cancer immunotherapy has become a promising method for cancer treatment, bringing hope to advanced cancer patients. However, immune-related adverse events caused by immunotherapy also bring heavy burden to patients. Semiconducting polymer nanoparticles (SPNs) as an emerging nanomaterial with high biocompatibility, can eliminate tumors and induce tumor immunogenic cell death through different therapeutic modalities, including photothermal therapy, photodynamic therapy, and sonodynamic therapy.

In addition, SPNs can work as a functional nanocarrier to synergize with a variety of immunomodulators to amplify anti-tumor immune responses. In this review, SPNs-based combination cancer immunotherapy is comprehensively summarized according to the SPNs' therapeutic modalities and the type of loaded immunomodulators. The in-depth understanding of existing SPNs-based therapeutic modalities will hopefully inspire the design of more novel nanomaterials with potent anti-tumor immune effects, and ultimately promote their clinical translation.

# 1. Introduction

Over the past decade, cancer immunotherapy has undergone significant advancements, emerging as a crucial breakthrough in the field of cancer treatment.[1] Distinguished from traditional cancer treatment methods such as surgery, chemotherapy, and radiation therapy, cancer immunotherapy focuses on activating the patient's own immune system to combat tumors. Activated immune cells not only control the growth of the primary tumor but also differentiate into immune memory cells, inhibiting tumor metastasis and recurrence.[2] The current cancer immunotherapy revolves around establishing a robust cancer-immunity cycle centered on T lymphocytes.[3] The cancer-immunity cycle begins with cancer cells releasing tumor-associated antigens (TAAs). Subsequently, antigen-presenting

J. Wu, K. Pu  
School of Chemistry, Chemical Engineering and Biotechnology  
Nanyang Technological University  
70 Nanyang Drive, Singapore 637457, Singapore  
E-mail: kypu@ntu.edu.sg

K. Pu  
Lee Kong Chian School of Medicine  
Nanyang Technological University  
59 Nanyang Drive, Singapore 636921, Singapore

![](dt=2026-03-19/ht=13/7d9bc37e515f6f42422a64e1c35cc8aaecd0df05567722142472a5997d24b101.jpg)

The ORCID identification number(s) for the author(s) of this article can be found under https://doi.org/10.1002/adma.202308924

DOI: 10.1002/adma.202308924

cells (APCs) process these antigens and present them on the major histocompatibility complex (MHC) molecules to educate and activate naive T cells.[4] Activated T cells rapidly proliferate, generating a large population of tumor antigen-specific cytotoxic T cells (CTLs). Guided by chemokines and adhesion molecules, these CTLs migrate from the bloodstream to the tumor site.[5] Upon the pairing of the TCR receptor on CTL cells' surface with MHC-antigen peptide complexes on tumor cells' surface, T cells secrete perforin and granzyme B, inducing tumor cell apoptosis and the release of TAAs.

[6] Simultaneously, a portion of the T cells that have killed tumor cells can be differentiated into long-lived memory cells, participating in new cancer-immunity cycles.[7] Currently, various cancer immunotherapies, including immune checkpoint blockade (ICB),[8] cancer vaccines,[9] chimeric antigen

receptor (CAR) T cell therapy,[10] and cytokine therapy,[11] have been adopted as novel cancer treatment modalities, exhibiting encouraging clinical benefits. Unfortunately, existing cancer immunotherapies often face the challenges of insufficient response rate and are accompanied by immune-related adverse events (irAEs) that can harm multiple organs.[12] The combination of multiple cancer immunotherapies has shown significant improvement in tumor suppression, but it also imposes a greater burden on patients' health. Therefore, there is an urgent need to develop novel drug delivery strategies that are compatible with combination cancer immunotherapies while reducing the potential irAEs.

In recent years, targeted nano-delivery systems for cancer therapy have garnered large attention.[13] Utilizing nanocarriers to deliver immunomodulators has emerged as a promising approach to enhance the efficacy of cancer immunotherapy while reducing off-target toxicity.[14] By modulating the particle size, surface charge, and surface functional groups of nanocarriers, they can be selectively accumulated in tumor tissues.

[15] Upon reaching the tumor site, some rationally designed nano-delivery systems can also respond to overexpressed biomarkers in the tumor microenvironment (TME) to trigger drug release (e.g., glutathione, enzymes, reactive oxygen species, etc.).[16] These nanocarriers can effectively modulate local immunity within the TME and significantly reduce irAEs. Furthermore, targeted nano-delivery systems can be combined with imaging agents, enabling real-time monitoring of drug distribution and immune response.[17] This

REVIEW

图

Check for updates

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (1 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/792348278ee4512beb4ace579c36542ef68717fe9fed140179f9bacffc5d2209.jpg)

integration of treatment and diagnosis allows for personalized medical approaches, facilitating treatment optimization and early assessment of treatment outcomes.

In various nano-delivery systems, semiconductor polymer nanoparticles (SPNs) show great potential in combination cancer immunotherapy, leveraging their unique photophysical and biophysical properties (Figure 1).[18] Semiconductor polymers refer to a class of polymers with carbon-based $\pi$ -conjugated backbones.[19] Through assembly or modification with hydrophilic segments, semiconductor polymers can form nanoparticles in aqueous solutions. SPNs exhibit high biocompatibility, strong tumor accumulation ability, and fluorescence tracking capabilities.[20] According to tunable donor-acceptor (D-A) structures of the semiconducting polymer core, SPNs can eliminate tumor cells efficiently through multiple therapeutic modalities. Under the photothermal therapy (PTT) mode, SPNs gen

erate sufficient heat upon light irradiation, thereby promoting tumor apoptosis or necrosis.[21] In photodynamic therapy (PDT) and sonodynamic therapy (SDT) modes, SPNs can transfer the energy of excited states to oxygen, leading to the production of singlet oxygen and tumor eradication.[22] Importantly, the tumorkilling effect mediated by SPNs has been demonstrated to be highly immunogenic. The immunogenic cell death (ICD) of tumors facilitates the generation of high-quality tumor antigens, promotes antigen processing by APCs, and initiates the first step of the cancer-immunity cycle.

[23] Additionally, due to their high chemical flexibility, SPNs can serve as drug carriers in combination with a wide range of immune modulators and reinforce the immune response. Importantly, the rational design of the cleavable linker can allow the controllable release of immunomodulators in the tumor area, avoiding systemic toxicity. Differing from other conventional nanomedicines, SPNs can serve as a

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (2 of 22)

© 2023 Wiley-VCH GmbH

multifunctional theranostic platform, incorporating fluorescence imaging, tumor ablation, and immune activation.[24] In addition, the appropriate stiffness endows SPNs with higher tumor accumulation than other nanomedicines.

In this review, we comprehensively summarize the progress of cancer immunotherapy based on SPNs. We commence with a discussion on the intrinsic properties of SPNs for multimodal therapy and their suitability f
or cancer treatment. Notably, we highlight how SPNs function as therapeutic agents and drug carriers, synergizing with various immune modulators, including but not limited to checkpoint inhibitors, immune adjuvants, metabolic regulators, and TME modulators. The SPNs-based cancer immunotherapy effectively controls tumors, elicits powerful immune responses, and has no discernible adverse effects. We expect this review to inspire the design of new nanomaterials for cancer immunotherapy and clinical translation.

# 2. Intrinsic Properties of Semiconducting Polymer Nanoparticles (SPNs)

# 2.1. Photophysical Properties of SPNs

SPNs possess unique photophysical properties that render them highly valuable in combination cancer immunotherapy, playing diverse roles as imaging agents and therapeutic units. As imaging agents, SPNs offer advantages such as tunable emission wavelengths and good photostability, making them widely applicable for TME imaging.[25] By rational designing the donor-acceptor (D-A) units of the SPNs backbone, the absorption and emission characteristics of SPNs can be tuned from the visible to NIR regions. Under irradiation, electrons in the SPNs are excited from the $S_0$ state to the $S_1$ state.

During the process of electron relaxation back to the $S_0$ state, a portion of the energy is emitted in the form of photons, resulting in fluorescence (Figure 2a). Another portion of electrons returns to the $S_0$ state through vibrational relaxation, releasing thermal energy. The instantaneous thermal expansion of tumor tissue leads to a rapid increase in local pressure, generating ultrasound waves that can be detected by sensors and forming high-resolution photoacoustic images.

[26] Additionally, SPNs have been reported to respond to reactive oxygen species (ROS) in the tumor area and produce afterglow imaging.[27] The imaging capabilities of SPNs enable researchers to non-invasively monitor the tumor accumulation of SPNs in animal models, allowing them to select the optimal timing for treatment.

When working as therapeutic agents, SPNs commonly play a role in combination cancer immunotherapy involving PTT, PDT, or SDT. Briefly, during the PTT, SPNs can convert light energy into heat, leading to thermal damage within tumor tissues.[28] In the PDT pathway, some electrons excited under light irradiation can reach the $\mathrm{T}_{1}$ state via intersystem crossing (ISC). When the electrons return to the $\mathrm{S}_0$ state, they transfer energy to oxygen, generating singlet oxygen that effectively eradicates tumor cells.

The exact mechanism of SDT remains unclear, although some researchers propose it has similarities to PDT. Undisputedly, after ultrasound irradiation, SPNs can transfer a portion of energy to oxygen or water molecules in the TME, generating ROS and causing tumor cell death.[29] It is worth noting that SPN-mediated tu

mor cell death is highly immunogenic, indicating that alongside tumor ablation, it triggers the production of high-quality tumor antigens and damage-associated molecular patterns (DAMPs). After SPN treatment, the activation of dendritic cells (DCs) is upregulated, promoting efficient antigen presentation and T cell education.[30]

# 2.2. Biophysical Properties of SPNs

The excellent biophysical properties of SPNs are reflected in their long-term tumor accumulation ability and high biocompatibility. Over the past decade, SPNs have been used for the treatment of various tumor models in living animals, including rodents and rabbits. The long-term tumor accumulation of SPNs has been widely reported in previous studies.[31] When used as drug carriers, SPNs show excellent versatility, enabling efficient delivery of chemotherapeutic drugs, nucleic acids, antibodies, and small molecule inhibitors to the tumor area.

The tumor accumulation of SPNs is typically in the range of 15-20 injected dose per gram $(\% \mathrm{ID} \mathrm{g}^{-1})$ [32] which is significantly higher when compared with traditional polymeric NPs $(2 - 7\% \mathrm{ID} \mathrm{g}^{-1})$ [33] or inorganic NPs $(1 - 4\% \mathrm{ID} \mathrm{g}^{-1})$ [34] In addition, SPNs can remain in the tumor region for 3 days or longer. The particle size of SPNs is around $50 \mathrm{~nm}$ , which is similar to other types of NPs.

Therefore, the enhanced permeability and retention (EPR) effect is not a sufficient explanation for the long-term tumor accumulation behavior of SPNs.

We suppose that the appropriate stiffness of SPNs may play a critical role in long-term tumor accumulation. Several studies have reported the impact of NP stiffness on tumor penetration.[35] Hammond's group demonstrated a direct influence of NP stiffness on pharmacokinetics and tumor penetration.[36] Compared to stiff NPs (with an elastic modulus of $24\mathrm{kPa}$ ), compliant NPs (with an elastic modulus of $6\mathrm{kPa}$ ) could penetrate tumors more effectively.

Similarly, Moses and co-workers reported that tumor cells internalized soft NPs (Young's modulus $<1.6\mathrm{MPa}$ ) to a much higher extent than hard NPs (Young's modulus $>13.8\mathrm{MPa}$ ), which enabled the better penetration of soft NPs into tumors.[37] However, high interstitial fluid pressure (IFP) is one of the characteristics of tumor tissue.

In such cases, traditional polymer NPs formed by self-assembly through electrostatic or hydrophobic interactions disassembly easily due to the failure to reach the critical micelle concentration, making it difficult to stay in the tumor region. In contrast, inorganic NPs, although difficult to penetrate the tumor region, exhibit strong retention properties once they enter, as they are difficult to be "washed out" by the tumor.[38] It is worth noting that SPNs are likely to achieve a balance between tumor penetration and retention due to their appropriate stiffness (Figure 2b).

The polymer chains (usually poly(ethylene glycol), PEG) on the surface of SPNs provide them with a "soft" characteristic to penetrate the tumor tissue, while the internal $\pi-\pi$ interactions can maintain SPNs stable and slow down their degradation, thereby increasing SPNs retention during the treatment period (usually 2-3 days).

On the other hand, the monomers constituting semiconductor polymers are typically organic and biologically inert, enabling SPNs to avoid heavy metal induced toxicity.[39] Furthermore, the majority of PEG-modified SPNs are neutral or negatively

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (3 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/a09ee923ceea4db631f6cc499acb71e57602d3584ea52bd2978f8a9cb3c2b49a.jpg)

![](dt=2026-03-19/ht=13/dd2490f337db86bf2c343c7a984d55754651c11b4e8a65a4495f05c27de05df3.jpg)

![](dt=2026-03-19/ht=13/0595e23df286fb507a1c1a4923c5d1bea1412d13941b0cc3b9ff6b354c5fbfff.jpg)

charged, which reduces their binding to proteins during blood circulation, thus minimizing potential impacts on physiological functions. High biocompatibility of SPNs has been proved in various cell lines and animals in previous studies.[40] When incubated with blood from healthy volunteers, SPNs did not stimulate platelet activation or aggregation.[41] At the in vivo level, animals treated with SPNs showed no behavioral abnormalities and did not exhibit acute/chronic toxicity or alterations in blood cell counts throughout the treatment process.

After the treatment, blood biochemistry markers (blood urea nitrogen, aspartate aminotransferase, alanine aminotransferase, etc.) showed no significant changes.[42] Moreover, extensive research through H&E staining demonstrated that SPNs did not cause major organ damage.[22b,43]

# 3. SPNs-Based Combination Canc
er Immunotherapy

In combination cancer immunotherapy, SPNs can serve as functional carriers for spatiotemporal controllable drug release, synergizing with various immunomodulators. The diverse therapeutic modalities mentioned above can also provide insights for designing cleavable chemical linkers. For example, thermoresponsive and ROS-responsive linkers were widely used in controllable drug release. Moreover, when the drug itself is stimuli-responsive, SPNs can activate the drug upon triggering.[44] In the following sections, we will discuss in detail how SPNs function as a multi-functional nanomaterial, synergizing with various immunotherapies to achieve enhanced cancer treatment outcomes.

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (4 of 22)

© 2023 Wiley-VCH GmbH

# 3.1. Photothermal Immunotherapy

Photothermal therapy (PTT) is a treatment method that utilizes photosensitizers to absorb light energy and convert it into heat, with the aim of destroying cancer cells.[45] The mechanism of PTT for cancer treatment is determined by the heat temperature. When the temperature is controlled below $45^{\circ}\mathrm{C}$ , such low-temperature photothermal therapy is called mild hyperthermia, primarily assisting cancer immunotherapy by promoting immune cell infiltration and improving vascular permeability.

[46] When the temperature is between 45 and $50^{\circ}\mathrm{C}$ , the heat can cause significant damage to cancer cells while avoiding thermal injury to normal tissues. The cancer cell death in PTT is usually immunogenic, accompanied by the calreticulin expression on cell membrane and the release of DAMPs. Depending on the different wavelengths of the irradiating light, photosensitizers in PTT therapy can be excited by first near-infrared (NIR I, 750–1000 nm) or second near-infrared light (NIR II, 1000–1500 nm).

Compared to NIR-I PTT, NIR-II PTT offers advantages such as deeper penetration, less normal-tissue toxicity and less energy dissipation.[47] Due to the strong modularity of the D-A unit in semiconducting polymer materials, SPNs can meet the requirements of various excitation lights.[48] As a result, many photosensitizers based on SPNs have been developed and employed for PTT. However, single PTT is easy to induce tumor adaptive resistance.

Under the stimulation of photothermal effects, tumors could upregulate the expression of heat shock proteins (HSPs) to acquire resistance to high temperatures and avoid apoptosis.[49] Meanwhile, it has been reported that PTT can stimulate tumors to upregulate immune checkpoint proteins or immune inhibitory factors, leading to an immunosuppressive TME and inhibiting the activity of immune cells.

[50] To solve this problem, SPNs loaded with various immune modulators can effectively kill tumors through PTT while polarizing the TME into an immune-active phenotype, ultimately enhancing the efficacy of combination cancer immunotherapy.

# 3.1.1. PTT in Combination with Immune Agonists

Immune agonists are a type of substance that can regulate immunological responses by altering molecular signaling pathways or enhancing molecule activities. The synergistic effect of immune agonists and PTT in cancer treatment has been widely reported.[51] However, systemic administration of small molecule agonists can result in irAEs, increasing the burden on cancer patients.

To solve this problem, we reported an activatable semiconducting polymer nanoagonist (APNA) for NIR-II light-regulated photothermal immunotherapy of cancer.[52] APNA was composed of a NIRII light-absorbing semiconducting polymer backbone pBODO as the photothermal transducer, conjugated with a potent toll-like receptor (TLR) agonist R848 (Figure 3a). APNA could self-assemble into NPs in PBS solution, showing good photothermal property and photostability upon NIR-II $(1064\mathrm{nm})$ photoirradiation. It is worth noting that the semiconductor backbone and R848 were connected via a thermolabile cleavable linker VA-044, which could selectively release the immune ag

onist under the stimulation of photothermal effects and esterase, avoiding normal-tissue toxicity (Figure 3b,c). In the in vitro experiment, APNA + Laser group significantly enhanced DCs maturation (Figure 3d). When applied in vivo through intravenous $(i.v.)$ injection, APNA + Laser group showed strong tumor inhibition ability and achieved superior survival benefit (Figure 3e). In addition, APNA-mediated photothermal immunotherapy could inhibit lung and liver metastasis, which was not achievable for either monotherapy.

Mechanistic studies of photothermal immunoactivation in deep tumors demonstrated that as the tumor depth increases, the intratumoral photothermal temperature gradually decreased, which might reduce tumor ICD (Figure 3f). Importantly, at different photothermal depths, the DC maturation ratio in the ANPA + Laser group remained the same (Figure 3g). This reflected the excellent penetrability of the low-temperature $(\approx 44^{\circ}\mathrm{C})$ responsive release of immune agonist in the tumor.

The deep tumor immunostimulation achieved by APNA-mediated photothermal immunotherapy enhanced systemic antitumor immunity and suppressed the development of primary, distant, and metastatic tumors.

Besides, we reported a smart polymer nanoagonist (SPND) with NIR-II PTT and tumor-specific granzyme B (GrB) restimulation ability.[53] The inhibitor of serpinB9 (Sb9i) is grafted onto the semiconducting polymer backbone of SPND via a glutathione (GSH) responsive linker. Through $1064\mathrm{nm}$ laser irradiation, SPND achieved a photothermal conversion efficiency (PCE) of $80.2\%$ , which could effectively kill 4T1 tumor cells. In antitumor immunity, GrB secreted by T cells would be inhibited by Sb9 in tumor cells. Sb9i prodrug on SPND could be triggered by the elevated levels of GSH in the TME, restimulating the function of GrB by inhibiting the GrB-Sb9 axis, thereby enhancing the treatment efficiency of photothermal immunotherapy.

Except for linking immune agonists to SPNs through chemical bonds, they can also be incorporated into SPNs through the hydration-sonication process or encapsulated together within microneedle carriers. For example, He et al. developed a microneedle delivery system composed of PCPDTBT SPNs and TLR-3 agonist (Poly(I:C)).[54] Under 808 nm irradiation, the PCE of SPNs was $20.1\%$ . After piercing the B16F10 tumor-bearing mice's skin, the SPNs and Poly(I:C) could penetrate the tumor efficiently. After laser irradiation, SPN-based PTT and Poly(I:C) showed a synergetic antitumor effect. The highest population of $\mathrm{CD3^{+}CD8^{+}}$ T cells was observed in the combination group, which was 3.7-fold higher than the single PTT group.

# 3.1.2. PTT in Combination with Metabolic Modulators

Metabolic therapy is a therapeutic strategy that aims to regulate cellular metabolic processes. This approach can be applied to a variety of cell types and is efficient when combined with PTT. When the target is tumor cells, it usually regulates the glycolytic pathway in tumor mitochondria, amplifying the effect of PTT by causing cancer cell starvation. Huang and co-workers developed a semiconducting polymer DPQ with superior NIR-II PTT efficacy, together with a glycolysis inhibitor 2-deoxy-D-glucose (2DG).[55] The prepared SPNs could not only achieve NIR-II PTT but also induce tumor starvation by inhibiting anaerobic glycolysis in the

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (5 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/b5f37b7f957b234d636b8b8888d839e23eabac57097abaa89df44aae3b2b5327.jpg)

![](image)
9e1a76962a30126f6ddc6f8c1ee35.jpg)

![](dt=2026-03-19/ht=13/b489408d04343e240148991ac833cbbfdb2203e0fcf8034887d53e45cb53ce5c.jpg)

![](dt=2026-03-19/ht=13/1ce76831984d2252382e205de994086e25d0fd2261789af80c6da74480f41d83.jpg)

![](dt=2026-03-19/ht=13/c9865ee94eb013f13708079475fe7dcd40e6227334fcb937d4afa80676c72d1e.jpg)

![](dt=2026-03-19/ht=13/7851c6d4769daf594f76a12497f7766cfd9205d74257c80c1163343d4409f25a.jpg)

![](dt=2026-03-19/ht=13/6bae7516d228da5f9a1c55448ccf797d7dc0eb0181ed439d7f93c20bee6dfd12.jpg)

![](dt=2026-03-19/ht=13/8306db34a8d986024187bc1c6992dbdc38945803bba59514fd9ed162b8c8f5b8.jpg)

presence of 2DG, causing a $90.4\%$ tumor suppression rate on the 4T1 tumor-bearing mice model.

In addition, metabolic pathways in immune cells can also be modulated to enhance photothermal immunotherapy. For example, one research in our group reported a NIR-II activatable semiconducting polymeric nanoantagonist (ASPA) for synergistic photothermal immunometabolic therapy.[56] The metabolic regulator vipadenant (VIPA) was linked onto the semiconducting backbone through a thermolabile linker (Figure 4a). VIPA is an antagonist to the adenosine A2A receptor, which can enhance CTL activity while attenuating Tregs functions (Figure 4b).

ASPA had strong absorption in the NIR-II light region, and it exhibited high PCE $(85.4\%)$ and photostability. The photothermal release behavior of VIPA was confirmed by HPLC (Figure 4c). Under photoirradiation, the VIPA release ratio rapidly increased to $68.3\%$ in $10\mathrm{min}$ and ultimately reached $76.3\%$ after $20\mathrm{min}$ . The released VIPA from ASPA showed

similar adenosine binding affinity when compared with the free drug (Figure 4d). The therapeutic efficiency of ASPA was tested on the 4T1 tumor-bearing mice model. As a result, ASPA + Laser successfully eliminated tumors in mice without recurrence for 3 weeks. Following immune mechanism studies proved that ASPA + Laser could increase the number of mature DCs in tumors, which was due to tumor ICD induced by PTT (Figure 4e).

More importantly, ASPA + Laser group had the highest number of $\mathrm{GrB^{+}}$ CTLs in tumors, probably due to the dual function of NIR-II PTT and VIPA metabolic regulation (Figure 4f). In addition, ASPA + Laser group also showed the lowest level of intratumoral TGF- $\beta$ amount, proving that ASPA could dampen the immunosuppressive Treg cells through the adenosergic pathway (Figure 4g). The therapeutic benefits of ASPA were also tested in B16F10 tumor-bearing mice, which exhibited prolonged survival time and strong immune response.

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (6 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/19492dfd5d5f640a973b4d7f4f119c88c2d7a82010914267ff986c2dbd3450c4.jpg)

![](dt=2026-03-19/ht=13/8055a0b45927abc27d98dfd6383d0a0da6dd4911eac1fa4448acdaa9cd6722e7.jpg)

![](dt=2026-03-19/ht=13/6e921cd9cdfc830bc0a90bd1a5df0c14c831f87c757ec7ffb61de9d5908f50ee.jpg)

![](dt=2026-03-19/ht=13/01ba5f60b80d20b71a3dc6b5cc1b984787a65734bf5a64ccc96ee333cb5b38a8.jpg)

![](dt=2026-03-19/ht=13/bd39fafffdfa463fee6a1af6ca29a397702aa4ad7dd06584a920b44b7bb55799.jpg)

![](dt=2026-03-19/ht=13/68e43719968e12dc2f871e2fbda45a12baa8322e2c221e1b4e4a4cdbe02e696b.jpg)

![](dt=2026-03-19/ht=13/e0c02820211acf353a02033dd57834b53982aa331098ad1d8f6f15587a5aeb4f.jpg)

![](dt=2026-03-19/ht=13/6458cca7303ee9a20df27f010a97845c1dfe454d868cff638b7d98c44eaa8d4f.jpg)

![](dt=2026-03-19/ht=13/d17036fe295400277b3e2c7940c350b420a25aafb6a8b26816bf20efbb38f167.jpg)

![](dt=2026-03-19/ht=13/a5329495ec28b2d16a004615b3ae83cbd5a99f262cfd53bdd05329a05de58210.jpg)

# 3.1.3. PTT in Combination with TME Modulators

Immunosuppressive TME is a critical factor that mediates immune cell dysfunction or exhaustion. Remodulating the TME will enhance immune cell activity, thereby amplifying the therapeutic effects of photothermal immunotherapy. Zhen and coworkers reported a semiconducting polymer nanomanipulator (SPNm) that can regulate protein expression in the TME via NIR-II light irradiation.[57] SPNm comprised a NIR-II absorbing semiconducting polymer core, a lysine-specific histone demethylase 3 A (KDM3A) inhibitor IOX-1, and a thermo-responsive lipid shell (Figure 5a). After self-assembly in water, SPNm could form

uniform nanoparticles and showed broad absorption in the range of $950 - 1100\mathrm{nm}$ (Figure 5b). $10\mathrm{min}$ after $1064~\mathrm{nm}$ laser irradiation, IOX-1 release was confirmed through HPLC, proving that SPNm could achieve photothermally controlled drug release (Figure 5c). TME modulator IOX-1 could inhibit the demethylation function of KDM3A to upregulate the level of H3K9me2, resulting in HSP90 and metastasis-related proteins downregulation (Figure 5d). The anti-tumor ability of SPNm was tested in 4T1 tumor-bearing mice. The tumors in the $\mathrm{SPNm} +$ laser group were nearly completely suppressed with negligible lung metastases (Figure 5e). The immunofluorescence staining (IF) analysis showed that IOX-1 could down-regulate

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (7 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/711fa0ec315b19630d884e906b3bdb906a8858232ec4c201943da4dd4ec14894.jpg)

![](dt=2026-03-19/ht=13/4245c76830a6a15a853f1259fa8bc42ac69fee2efec2991d0aaaa1f325fd6a82.jpg)

![](dt=2026-03-19/ht=13/4f6df9777da560adbe06b48ccaeb6733104e2a9b3c8d090fb4a4a8504c436c61.jpg)

HSP90 expression, impeding the thermotolerance response of tumors (Figure 5f). In addition, decreased c-Jun and MMP-9 could inhibit tumor metastases, finally leading to the synergistic effect with PTT (Figure 5g).

Similarly, Xie's group reported SPNs contained in PDPP3T and gambogic acid (GA) via the nanoprecipitation method.[58] GA is a natural product and has an HSP90 inhibition potential. The achieved SPNs had strong absorption in the NIR-I region. By $808\mathrm{nm}$ laser irradiation, SPNs could efficiently induce HepG2 cell death, due to the dual action of photothermal effect and HSP90 inhibition. In addition to small molecule inhibitors, some SPNs carrying nucleic acids could also enhance photothermal immunotherapy by regulating the TME. Fu et al. developed a NIR-light-mediated optogenetic system by SPNs loaded interferon gamma (IFN- $\gamma$ ) plasmid with HSP
70 promoter.[59] Under $808~\mathrm{nm}$ laser irradiation, tumor cells would upregulate HSP70

expression and promote IFN- $\gamma$ production. When incubated with M2 macrophages, IFN- $\gamma$ generated by tumor cells could polarize M2 macrophages into M1 phenotype, enhancing the therapeutic effect of PTT.

# 3.1.4. PTT in Combination with Other ICD Inducers

Although PTT can induce tumor ICD, the tumor's adaptive heat tolerance would gradually weaken the treatment efficacy and immune response. Therefore, combining PTT with other ICD inducers can stimulate tumor cells to undergo ICD through different pathways, thus achieving effective tumor elimination and enhanced immune response.[60] Immunogenic chemotherapy is a therapeutic approach that utilizes specific chemotherapeutic agents to induce tumor ICD, including most anthracycline-based

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (8 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/e4b2f2a3d31765c51d8ede8c81dbcc0b79a61cad3aab5721bfcd419e6529dad7.jpg)

![](dt=2026-03-19/ht=13/4ebf46e775074b7dce4ff42884ef014ac8426f13f2d69eb58532ad5e9f00655c.jpg)

![](dt=2026-03-19/ht=13/ff70332cedb25f236852016c25f0dd346adafa789e981763ddf8cc39d182845a.jpg)

![](dt=2026-03-19/ht=13/b39cc4e6111aeb2e34614059fe58ac4d659d2c81d6ef70f3a38a020628adc24d.jpg)

![](dt=2026-03-19/ht=13/e448c17cdea5cd93ea4caf4a596a963284577ded7f0d7152e4f0eed64f4e2598.jpg)

![](dt=2026-03-19/ht=13/edda920727e5b954eaec4bd8eac555c2e0a088ef2a2babd445453ee0f06e65b3.jpg)

![](dt=2026-03-19/ht=13/3232e927bcb82c4fee5abbb7ddaddd9b3949afa572e9b0e113284c01540fc10b.jpg)

![](dt=2026-03-19/ht=13/65e46c14f32a55f22b980f95e3616bd62d4123f2bfd397ea6c7db2cd1f2a8d25.jpg)

![](dt=2026-03-19/ht=13/59f9625ded780245c84e9a2405111a92c8deef3f05a7bddd6fe6bb94cb414b80.jpg)

chemotherapeutic drugs, oxaliplatin and paclitaxel. Recently, Xiao's group reported SPNs called $\mathrm{NP}^{\mathrm{PSP - Pt}}$ consisted of NIR-II light absorption semiconducting polymer unit and oxaliplatin prodrug (Pt(IV)) for efficient photothermal immunotherapy.[61] Pt(IV) was conjugated into the polymer main chain through Stille polymerization. Under NIR-II light irradiation, the photoelectrons emitted from photosensitizers could be absorbed by

atoms in the $\mathrm{Pt(IV)}$ , thereby being reduced to free cytotoxic $\mathrm{Pt(II)}$ species. The controllable release of $\mathrm{Pt(II)}$ could penetrate tumor tissue deeply and induce strong tumor ICD, synergizing well with PTT (Figure 6a). Under light irradiation, a faster $\mathrm{Pt(II)}$ release was confirmed by ICP, indicating that the NIR-II irradiation could promote the $\mathrm{Pt(IV)}$ reduction and release (Figure 6b,c). $\mathrm{NP}^{\mathrm{PSP - Pt}}$ had wide-range absorption, especially in the NIR-II

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (9 of 22)

© 2023 Wiley-VCH GmbH

region. The PCE of $\mathrm{NP}^{\mathrm{PSP - Pt}}$ was $43.2\%$ along with good photostability (Figure 6d). $\mathrm{NP}^{\mathrm{PSP - Pt}} + \mathrm{Laser}$ (L) treatment induced strong CT26 cell ICD, which was confirmed by extracellular ATP evaluation, IF staining of HMGB1 and CRT. Furthermore, significant tumor growth inhibition of $\mathrm{NP}^{\mathrm{PSP - Pt}} + \mathrm{L}$ group was observed in the CT26 tumor-bearing mice. The immunogenic response inside the tumor was studied by flow cytometry.

$\mathrm{NP}^{\mathrm{PSP - Pt}} + \mathrm{L}$ treatment dramatically increased the mature DCs both in the tumor and lymph nodes, indicating a strong tumor ICD has happened. In addition, the $\mathrm{NP}^{\mathrm{PSP - Pt}} + \mathrm{L}$ group also had the highest amount of $\mathrm{CD8^{+}}$ T and $\mathrm{CD4^{+}}$ T cells, proving that the combination of PTT and immunogenic chemotherapy held great potential for cancer treatment by photothermal immunotherapy (Figure 6e,f).

Ferroptosis is an emerging mode of cell death. Due to its independence from the traditional apoptotic pathway and its high immunogenicity, the application of ferroptosis in tumor therapy has attracted much attention. Our group has reported a series of photothermal ferrotherapy for tumor treatment based on SPNs.[62] For example, Jiang et al. reported a novel transformable hybrid semiconducting polymer nanozyme called HSN.[63] The rich sulfur and nitrogen atoms in the semiconducting polymer backbone could bind $\mathrm{Fe}^{2+}$ efficiently and form uniform NPs (45 nm).

Upon NIR-II light irradiation, HSN generated heat for PTT and enhanced the Fenton reaction. The increased $\cdot\mathrm{OH}$ could not only induce cell ferroptosis but also transform HSN into tiny segments (1.7 nm) with increased tumor permeability. In the 4T1 tumor-bearing mice model, HSN-induced photothermal ferrotherapy achieved effective tumor killing at 9 mm depth. More importantly, the systemic immune activation effect generated by HSN could inhibit tumor lung and liver metastasis.

# 3.1.5. PTT in Combination with Multiple Immunotherapeutic Agents

Multiple immunotherapeutic agents work on different signal axes may elicit stronger immune responses and synergize with PTT. For example, Yasothamani et al. reported a tannic acid doping polyaniline NPs modified with estrogen receptor (ER) targeting agent and R837.[64] The achieved NPs could induce $\mathrm{ER^{+}}$ tumor cells ICD via NIR-I light-mediated PTT and amplify immune response through TLR-7 agonist R837. After combining with the ICB strategy (anti-PD-L1), the therapeutic tactics not only ablated the tumor by PTT, but also provided strong anti-cancer immunity, increasing the level of $\mathrm{CD8^{+}}$ T cells and pro-inflammatory cytokines in the tumor.

In addition, Li et al. developed a three-in-one immune modulation strategy based on NIR-II light-activated SPNs.[65] Such SPNs contained a NIR-II light-absorbing core, a PD-L1 inhibitor, a NO precursor (L-arginine) and tumor extracellular matrix (ECM) elimination agent bromelain. After NIR-II light irradiation, SPNs could trigger potent PTT and release multiple immunomodulatory drugs through the destruction of the thermo-responsive shell. When treating mice breast cancer, three-in-one SPNs could effectively control tumor growth, increase the number of intratumoral $\mathrm{CD8^{+}}$ T cells, and reduce tumor metastasis.

# 3.2. Photodynamic Immunotherapy

In PDT, photosensitizers upon photoexcitation can undergo ISC to reach the triplet state with a longer lifetime. Depending on the type of photosensitizer, the energy of the triplet state can be transferred to other substances, generating ROS through Type I or Type II PDT processes.[66] As a photo-controlled therapeutic strategy, PDT precisely generates ROS at the tumor site, effectively killing tumor cells and inducing tumor ICD. Currently, several photosensitizers based on SPNs have been developed and utilized for cancer treatment.[22a,67] Additionally, PDT is frequently combined with cancer immunotherapy to enhance the immune response.[68]

# 3.2.1. PDT in Combination with Immune Agonists

The controlled release of immune agonists in the tumor area can activate immune cell
s while avoiding systemic toxicity and amplifying the therapeutic effect of PDT. Wang and collaborators developed a smart semiconducting polymer nano-immunomodulator (SPNI) consisting of a NIR-absorbing semiconducting polymer to enable PDT and acid-responsive R837 prodrug to active DCs (Figure 7a).[69] The SPNI remained stable in blood circulation while achieving efficient R837 release in the tumor acid microenvironment (Figure 7b,c).

Simultaneously, SPNI under NIR light irradiation could generate ROS to trigger tumor ICD and form in situ cancer vaccine. The SPNI exhibited excellent tumor inhibition in both 4T1 and B16F10 tumor models, accompanied by the increased DCs activation ratio and T cell amounts in lymph nodes and tumors (Figure 7d). The combination of PDT and immune agonist based on SPNI not only eliminated the primary tumor but also elicited systemic immune response and immune memory effects.

Besides, Huang et al. reported a self-degradable SPN with TLR9 agonist CpG modification.[70] Superoxide radical $(\mathrm{O}_2\bullet)$ responsive degradable unit 1-methyl-1H-imidazole was inserted into the semiconducting polymer backbone to promote the CpG release during PDT. Under light irradiation, the SPN could generate $\mathrm{O}_2\bullet$ through the type-I process, killing tumor cells and releasing CpG for immune activation. When treating H22 tumor-bearing mice, the combination of PDT and CpG could eradicate the primary tumors and induce a systemic antitumor response.

# 3.2.2. PDT in Combination with Metabolic Modulators

Cellular metabolism plays a pivotal role in modulating the activity of immune cells. Among numerous metabolic signaling pathways, extensive research has been conducted on indoleamine 2,3-dioxygenase (IDO).[71] IDO exerts crucial functions in regulating immune responses and maintaining immune homeostasis. In tumors, IDO catalyzes the metabolism of tryptophan, converting it into kynurenine (Kyn), thereby inhibiting the proliferation and activity of $\mathrm{CD8^{+}}$ T cells.[72] Additionally, IDO can modulate the metabolism of DCs, dampening their antigen-presenting capacity.[73] Given IDO's multiple immunosuppressive effects, many IDO inhibitors have been developed to enhance the anti-tumor immune response. To achieve tumor-specific release of

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (10 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/1ccbd1e534570b76cab814add1d3ee49b4c0c66aca0b9d3c791d678298d4eadc.jpg)

![](dt=2026-03-19/ht=13/b60fa769f9dcef01f92fe70903fff9f324ecb82be8efd04ce2fb9e84effeb511.jpg)

![](dt=2026-03-19/ht=13/38addc88f0d2e1ed943e438cd11ea4b6778e5e9261c404d7c11961885d02ebd8.jpg)

![](dt=2026-03-19/ht=13/d2d199c21761d0bc1f2c83d2d3fb844722d9b3f224016b6b993cdd599f14fa0a.jpg)

IDO inhibitors, our group has developed a series of delivery systems based on SPNs.[32a,74] By conjugating various IDO inhibitors (KYNase, M-Trp) to the semiconducting polymer backbone via cleavable chemical linkages, we have successfully achieved ROS or enzyme-triggered inhibition of the IDO pathway, thereby enhancing the efficacy of photodynamic immunotherapy.

Of great significance, we introduced, for the first time, the concept of "semiconducting polymer Nano-PROTACs," which combined proteolysis targeting chimeras (PROTACs) with PDT (Figure 8a).[75] In our design, IDO-targeting PROTACs protein (IPP) are linked to semiconducting polymer backbones via a Cathepsin B (CatB) cleavable segment and PEG, forming $\mathrm{SPN}_{\mathrm{pro}}$ .

Under NIR light irradiation, $\mathrm{SPN}_{\mathrm{pro}}$ effectively generated ${}^{1}\mathrm{O}_{2}$ to eradicate tumors (Figure 8b). In the TME, IPP could be released efficiently due to the overexpressed CatB, followed by the degradation of IDO through classical PROTACs principles (Figure 8c). Notably, owing to PROTACs' protein-degrading characteristics, IPP exhibited a sustainable IDO degradation manner due to the reactivation and reuse of the active IPP (Figure 8d).

$\mathrm{SPN}_{\mathrm{pro}}$ mediated activatable photo-immunometabolic therapy was investigated in 4T1 tumor-bearing mice. The $\mathrm{SPN}_{\mathrm{pro}} +$ Laser treatment group significantly inhibited the primary and distant tumor growth. Through PDT-induced tumor ICD, the DC activation ratio was increased (Figure 8e). More importantly, the

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (11 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/93f5afac1e51368b6affd34ff9f18668b88e08e72aed35d59524c3a81223a3a0.jpg)

![](dt=2026-03-19/ht=13/c8a48b890cadaba2eb875203c4c21208ffe1c7b3da5f7f6d67121bc181bc7095.jpg)

![](dt=2026-03-19/ht=13/5237ec6f36253dd7a06cd93b38a49b340de988701928f3171546281b653461fe.jpg)

![](dt=2026-03-19/ht=13/060276e093aa9f0320cfbf585f49987e5a1562e0d9634081b5bfd0cafe6257a6.jpg)

![](dt=2026-03-19/ht=13/90e6ad8621cb56348f96ae9616a664e23688c44f72eb26c4c6a33c6b98faa917.jpg)

![](dt=2026-03-19/ht=13/a9d93e3073a6b6697ce7e22ed9bd041e51311a571a987fab0a1f80c360a5ba61.jpg)

![](dt=2026-03-19/ht=13/b056b901ec6a000951fb992747e89056135c1efc36988d538221750e05ec4383.jpg)

kynurenine content in tumors was dramatically decreased in the $\mathrm{SPN}_{\mathrm{pro}} +$ Laser group, which was attributed to the IPP-mediated IDO degradation (Figure 8f). As a result, $\mathrm{SPN}_{\mathrm{pro}} +$ Laser group showed the highest granzyme B amount in tumors, indicating a strong immune activation through photo-immunometabolic therapy (Figure 8g).

# 3.2.3. PDT in Combination with Immune Checkpoint Inhibitors

A class of proteins in the immune system that exert immunosuppressive functions are known as immune checkpoints. Through interaction with their ligands, checkpoints transmit signals and

modulate the activation state of immune cells and the extent of immune responses. The two most classical T cell-related immune checkpoints are CTLA-4 and PD-L1. In cancer immunotherapy, antibodies, small molecule inhibitors, or gene silencing techniques could effectively obstruct the function of immune checkpoints, resulting in stronger immune responses.[76] However, immune checkpoint blockade (ICB) often faces the challenge of insufficient response rates, attributed to the dearth of tumor-specific T cells. PDT can cause tumor ICD, producing high-quality tumor antigens, thus promoting antigen presentation by DCs and fostering the generation of tumor-specific T cells. Many therapies that combine PDT and ICB show increased efficacy in tumor treatment.[77]

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (12 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/b720edebc1bac5028c4ee21a00a0c125391b4efe17cf4f41ca5bb82dbbc2a081.jpg)

![](image)
8b3eabbc105ded3b1a0bd53f2647b3643598db7edab6.jpg)

![](dt=2026-03-19/ht=13/ac79555e140d167d4b4267084a971f03f8415750b500c1cd2ec72459b5abb849.jpg)

![](dt=2026-03-19/ht=13/b2100781ea992e2019689326accecac50381f24d391e38d38577116bb6d20e0b.jpg)

Ding et al. reported a SPN with surface-mimicking protein secondary structure as lysosome-targeting chimaeras (LYTACs) for PD-L1 degradation, rendering great promise for photodynamic synergistic immunotherapy.[78] Through rational design, the introduction of trifluoromethyl moieties into the semiconducting polymer reduced the bandgap of conjugated molecules, facilitating the process of ISC (Figure 9a). As a result, under white light irradiation, SP3 exhibited enhanced capability for ROS generation.

The photodynamic anti-tumor ability of SP3 NPs effectively elicited ICD in tumor cells, thereby increasing DCs activation and tumor-specific T cells. Furthermore, $\mathrm{^D F^D F G^D P P A}$ peptide sequence with $\beta$ -sheet secondary structures and PD-L1 peptide antagonist was conjugated with SP3 NPs to endow the NPs with ICB effect. SP3 NPs- $\mathrm{^D F^D F G^D P P A}$ had a stronger binding affinity for PD-L1 than other counterparts, indicating the importance of the secondary structure of the peptide and multivalent binding sites (Figure 9b).

When treating CT26 cells with SP3 NPs- $\mathrm{^D F^D F G^D P P A}$ , efficient PD-L1 degradation was observed, probably because of the multivalent PD-L1 cross-linking induced lysosome-targeting delivery and degradation (Figure 9c). In CT26 tumor-bearing mice model, the SP3 NPs- $\mathrm{^D F^D F G^D P P A}+$ Laser treatment exhibited the best tumor inhibition ability, lowest PD-L1+ tumor cells ratio, and highest active T cell ratio (Figure 9d).

In addition, the SP3 NPs- $\mathrm{^D F^D F G^D P P A}+$ Laser group could induce strong immune memory effects, providing a new strategy for designing photodynamic immunotherapy.

# 3.2.4. PDT in Combination with TME Modulators

The dense extracellular matrix (ECM) in the TME not only supports tumor cell growth and metastasis, but also acts as a physical barrier preventing infiltration of peripheral immune cells, helping the evasion of immune surveillance.[79] Therapies targeting ECM elimination have been demonstrated to generate synergistic effects with PDT, enhancing the efficacy of cancer immunotherapy.[80] Regretfully, it is challenging to properly control ECM-related signaling pathways using small-molecule inhibitors.

To solve this problem, we reported a SPN-based ECM nanoremodeler (SPNcb) for activatable cancer photoimmunotherapy.[81] SPNcb was synthesized by linking the lysyl oxidase (LOX) inhibitor onto the semiconducting polymer via a CatB-cleavable peptide sequence (Figure 10a). Under NIR light irradiation, SPNcb generated ROS to eliminate tumor cells and induce tumor ICD. In addition, CatB overexpressing in the TME could cleave the substrate linker and release active LOX inhibitors (Figure 10b,c).

The inhibition of the LOX pathway would facilitate ECM remodeling by reducing $\alpha$ -SMA and collagen I, thus promoting immune cell infiltration (Figure 10d). In the 4T1 tumor-bearing mice model, SPNcb + Laser treatment significantly inhibited the primary tumor growth and lung metastasis. Further studies on the immune mechanism proved that SPNcb + Laser treatment could increase the intratumoral $\mathrm{CD8^{+}}$ effector T cells and attenuate LOX-2-mediated ECM crosslinking in the lung.

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (13 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/8ca1ef85f1ba75a7a0e8080b027e8dd3978c3e68001fc584c36cdfbab9318ef8.jpg)

![](dt=2026-03-19/ht=13/ec6f87f9b4b599eb6c88e3740109505062ec620ac14a51d7a316a989a7375b3a.jpg)

![](dt=2026-03-19/ht=13/2d4acf0ee9bb2af85b47e138b004565f106948901872f75f37cc890d5d1b2e78.jpg)

![](dt=2026-03-19/ht=13/9a76c7d5522a1a7da7642b384bb745676b77f89d9decd9cf4fab2f6ca7550ad0.jpg)

In addition to the tumor ECM, immunosuppressive cells within the TME also constitute a major hindrance to the effectiveness of cancer immunotherapy. During the tumor proliferation process, the tumor recruits and activates tumor-associated immune cells through chemokines or chemical molecules. By secreting inhibitory cytokines, these cells prevent $\mathrm{CD8^{+}}$ T cells from activating and performing cytotoxic functions. To reprogram the TME, we presented a smart nano-PROTAC system based on SPNs, capable of PDT-induced tumor elimination and targeted degradation of tumor-specific proteins.

[82] The COX-1/2-targeting PROTAC peptide (CPP) is conjugated to the semiconducting polymer backbone via a CatB-cleavable segment, which could achieve tumor-specific release behavior. CPP ensured sustained degradation of COX-1/2 proteins in the TME, thereby reducing downstream PGE2 amounts. Depletion of PGE2 significantly diminished Tregs, M2 macrophages, and MDSCs within

the TME, polarizing the immunosuppressive TME into an active phenotype. This strategy synergized effectively with PDT, eliciting a potent immune response, restricting the growth of the primary tumor and preventing the occurrence of lung metastasis.

# 3.2.5. PDT in Combination with Multiple Immunotherapeutic Agents

Considering the complex immune escape mechanism of tumors, PDT combined with various immunomodulators can function on different axes, successfully overcome tumor drug resistance and immunological tolerance, and considerably improve anti-tumor immune responses through synergistic effects. In order to improve photodynamic immunotherapy efficiency, Li and colleagues created semiconducting polymer nanocomplexes

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (14 of 22)

© 2023 Wiley-VCH GmbH

(SPNCN) that could precisely suppress tumor autophagy and immunometabolism in TME upon NIR photoactivation.[83] Such SPNCN are made up of a PDT-enabled semiconducting polymer core and a shell that could react with ${}^{1}\mathrm{O}_{2}$ . The ${}^{1}\mathrm{O}_{2}$ responsive shell contained the autophagy inhibitor chloroquine (CQ) and IDO inhibitor NLG919. Tumor autophagy is highly related to the generation of drug resistance, and can selectively reduce the expression of MHC-I molecules to escape the recognition of T cells.

Meanwhile, the IDO pathway impairs T cell function by catalyzing the conversion of tryptophan to kynurenine. In the B16F10 tumor-bearing mice model, SPNCN + Laser induced strong tumor ICD through PDT. In addition, dual inhibition of autophagy and IDO pathways significantly increased the number of $\mathrm{CD8^{+}}$ T cells in primary and distant tumors and improved the survival of mice.

# 3.3. Sonodynamic Immunotherapy

Sonodynamic therapy (SDT) is an innovative non-invasive tumor treatment method. The mechanism of SDT is similar to that of PDT. By transmitting the energy of ultrasound irradiation to the sonosensitizer, its electrons can be excited to a high-energy state. When the electrons return to the ground state, they can transfer energy to oxygen or water molecules in the TME, generate ROS, and induce tumor cell ICD.

[29] Compared with NIR-light with a penetration depth of $< 1$ cm, ultrasound can effectively penetrate at least $4\mathrm{cm}$ of tissue, enabling ultrasound-based tumor diagnosis and treatment strategies with a high potential for clinical translation.[
84] In addition, it can also be combined with other cancer immunotherapy methods to achieve a synergistic effect and further improve the effect of cancer treatment. At present, many sonosensitizers based on SPNs have been developed and used in SDT.

# 3.3.1. SDT in Combination with Immune Agonists

Compared with other types of immune agonists, stimulator of interferon genes (STING) agonists have become an emerging cancer immunotherapy strategy due to their broad spectrum activity and potent efficiency.[85] STING is a protein with several transmembrane regions. Upon activation by agonists, STING phosphorylates interferon-regulated transcription factors and upregulates the expression of type I interferons and proinflammatory cytokines in a variety of immune cells, including DCs, macrophages and T cells, thereby activating the immune system.[86] However, most STING agonists face poor pharmacokinetics and tumor-targeting ability when applied in vivo, which limits their further applications.

To overcome this problem, Zhen and collaborators designed a semiconducting polymeric nanoagonist (SPNM) with ultrasound-controlled STING activation for precision sonoimmunotherapy (Figure 11a).[87] SPNM was made from a sonodynamic semiconducting polymer core conjugated to the STING agonist MSA-2 via ${}^{1}\mathrm{O}_{2}$ -responsive linkers. Under ultrasound irradiation, SPNM generated ${}^{1}\mathrm{O}_{2}$ , which not only eliminated tumor cells and triggered tumor ICD, but also released active MSA-2 by cleaving diphenoxyethylene bonds, activating the STING pathway in the TME (Figure 11b,c). MSA-2 could effectively bind and

activate STING, significantly increasing the phosphorylation of TBK1 and IRF3, thereby promoting the activation of immune cells (Figure 11d). Furthermore, SPNM + US treatment successfully suppressed the primary and distant tumor growth in SCC-7 tumor-bearing mice and induced potent immune memory effects (Figure 11e). The SPNM-based combination of SDT and STING activation could dramatically increase the effector T cells in tumors, accompanied by high serum IFN- $\beta$ levels (Figure 11f,g).

# 3.3.2. SDT in Combination with Immune Checkpoint Inhibitors

Immune checkpoint antibodies, the most extensively researched checkpoint inhibitor in clinical trials, have considerably increased patient survival in first-line therapy. However, checkpoint antibodies frequently induce irAEs when they activate the immune system. Controlled release of immune checkpoint antibodies locally in the tumor area can maximize the efficacy of immunotherapy while avoiding irAEs.

Our group reported SPNs-based sono-immunotherapeutic nanobodies $(\mathrm{SPN}_{\mathrm{Ab}})$ for sonodynamic activatable cancer immunotherapy (Figure 12a).[88] To achieve effective SDT, PFODBT was selected as the semiconducting polymer core, which could efficiently generate ${}^{1}\mathrm{O}_{2}$ and hydroxyl radicals under ultrasound irradiation (Figure 12b). In addition, the anti-CTLA-4 was conjugated to the semiconducting polymer backbone through the ${}^{1}\mathrm{O}_{2}$ -cleavable linker PSDA, which was gradually released during SDT (Figure 12c).

The anti-CTLA-4 released by $\mathrm{SPN}_{\mathrm{Ab}}$ had a similar CTLA-4 binding ability when compared with free antibodies, showing a potential to exert synergistic effects with SDT (Figure 12d). When treating 4T1 tumor-bearing mice, $\mathrm{SPN}_{\mathrm{Ab}} + \mathrm{US}$ treatment could ablate tumors and cause tumor ICD, and inhibit lung metastasis (Figure 12e).

Mechanism analysis revealed that $\mathrm{SPN}_{\mathrm{Ab}}$ under US irradiation significantly increased the ratio of mature DCs in tumor-draining lymph nodes and the number of $\mathrm{CD8^{+}}$ T cells in tumors (Figure 12f,g). Besides, the treatment generated high numbers of effector memory T cells, allowing mice to develop long-term antitumor effects and overcome tumor rechallenge.

# 3.3.3. SDT in Combination with TME Modulators

Myeloid-derived suppressor cells (MDSCs) are major immunosuppressive cells in tumors, which can be recruited by hypoxia or certain chemokines in the TME.[89] MDSCs would inhibit the activation and proliferation of $\mathrm{CD8^{+}}$ T cells by releasing immunosuppressive cytokines such as IL-10 and TGF- $\beta$ .[90] Considering the physiological role of MDSCs in maintaining immune homeostasis in normal organs, selectively modulating the MDSCs' function within the TME is important for cancer immunotherapy.

Li's group developed sonodynamic semiconducting polymer nanopartners $(\mathrm{SPN}_{\mathrm{Tl}})$ loaded with ROS-responsive tirapazamine (TPZ)-conjugate and MDSC inhibitor ibrutinib (Figure 13a).[91] TPZ was a tumor hypoxia-activated drug, which could amplify the SDT-induced tumor ICD, while ibrutinib could deplete MDSCs. $\mathrm{SPN}_{\mathrm{Tl}}$ could achieve efficient ${}^{1}\mathrm{O}_{2}$ generation under US irradiation because of its PFODBT core (Figure 13b). In addition, the

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (15 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/f9bda63d5e67f3a249c5292a3d0ed5e9e6668d881d6cb84c35fd32bebbbdcfd1.jpg)

![](dt=2026-03-19/ht=13/c71b0015aa9c91b431ec9c23f55f40f1193a1951103e8034c3cd197dbf9e391f.jpg)

![](dt=2026-03-19/ht=13/cdad3ea17917cf9d67a0b1a6dab4ba75919baced2ebdfa445e56a04cb248a2de.jpg)

![](dt=2026-03-19/ht=13/2bf52941096ba624fc364f4f2fc7dd9fb8114b4663efcede1c504a8477e74e7b.jpg)

![](dt=2026-03-19/ht=13/44de1ecd2923a5c6851c104d7e754fbc9fb500b44000067794aacfe65ec832ba.jpg)

![](dt=2026-03-19/ht=13/53000bb06449faa445093cecc7d442577b7b4cc888c3f7a7af0f7733b787310e.jpg)

![](dt=2026-03-19/ht=13/515473e860e9caf69fcc37ad632ac3197c06abcedbf477ac9b23245a58709201.jpg)

${}^{1}\mathrm{O}_{2}$ during the SDT process would break down the nanoparticle integrality to allow specific TPZ-conjugate and ibrutinib release into the TME (Figure 13c). In the 4T1 tumor-bearing mice model, $\mathrm{SPN}_{\mathrm{Ti}} + \mathrm{US}$ treatment effectively inhibited the primary and distant tumor growth with prolonged mice survival (Figure 13d). Flow cytometry results showed that $\mathrm{SPN}_{\mathrm{Ti}} + \mathrm{US}$ group exhibited the highest DCs maturation, which was attributed to the SDT and TPZ-induced tumor ICD (Figure 13e). Besides, ibrutinib released from $\mathrm{SPN}_{\mathrm{Ti}}$ could significantly decrease the MDSCs amounts in the TME (Figure 13f). Such combination of SDT and TME mod

ulator increased the intratumoral $\mathrm{CD8^{+}}$ T cells and facilitated a strong immune response against lung metastases.

# 3.3.4. SDT in Combination with Multiple Immunotherapeutic Agents

Accumulating evidence showed that the rational combination of SDT and multiple immunotherapeutic agents could further amplify the therapeutic effect of sonodynamic immunotherapy,

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (16 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/69a00fb27a6bca12d1b9db9f6aa380063d16a6f6683b15020be3fe2eee56a351.jpg)

![](image)
jpg)

![](dt=2026-03-19/ht=13/947a2a18552522305f98111c266a9961f62ab4f5c932154948517f2e4dd5042f.jpg)

![](dt=2026-03-19/ht=13/0ffdb6079013f7de48fc3e1976501c99068093ec0fa0aff810d814e455ac50ed.jpg)

![](dt=2026-03-19/ht=13/43a22a3c3d080681ec54bb67f8a8a379bcc3fba00980b906b9adf559af824bd4.jpg)

![](dt=2026-03-19/ht=13/f65792f0a3c9fe477389313333acc856db5ba84d1ab6cbe78082ced9db781daf.jpg)

![](dt=2026-03-19/ht=13/9bf1e20cee1842736ff8294460b46a61ebaa38c200692c91ee9835ff9103e7ab.jpg)

which was superior to monotherapy.[92] Our group reported novel semiconducting polymer immunomodulatory nanoparticles (SPINs) for deep-tissue activatable combination sonoimmunotherapy.[93] Through screening the library of sonosensitizers, SP7 was chosen as the semiconducting polymer backbone because of its highest ${}^{1}\mathrm{O}_{2}$ generation under US irradiation (Figure 14a). Then, checkpoint inhibitor $\alpha$ -PD-L1 and metabolic modulator NLG919 were conjugated to the backbone through a ${}^{1}\mathrm{O}_{2}$ -cleavable linker. While SDT killed tumors through ${}^{1}\mathrm{O}_{2}$ , it could also trigger the release of multiple immune modulators to regulate the immune response in situ (Figure 14b). Benefiting from the deep tissue penetration capability of ul

trasound, SPINs were able to be activated even under $10\mathrm{cm}$ pork tissues (Figure 14c). Both the ${}^{1}\mathrm{O}_{2}$ generation and the drug release behavior were similar with or without tissue barrier (Figure 14d,e). Next, SPIN-mediated sono-immunotherapy was tested in the Panc02 tumor-bearing mouse model. SPIN + US treatment efficiently inhibited the primary and distant tumor growth, prolonged mice survival and induced strong immune responses. Through multiplexed gene expression analysis, 261 genes were found to be differentially expressed in SPIN + US group (Figure 14f). The upregulated genes were highly associated with several key immune pathways, including ICD, DC activation, T cell priming, chemokines and pro-inflammatory

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (17 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/98e79ae882adc54c67af9cefc08aac511c144b551ec1c6146fc836fb0f9132ed.jpg)

![](dt=2026-03-19/ht=13/93a678cd4731c332687bef0b9e00c14ad67be26803f5c340725baa858ae2d06a.jpg)

![](dt=2026-03-19/ht=13/360fdb98edfd0be7efa89fd125ef1b5012cb34effe8d8d3d0343a7dc72fb4d86.jpg)

![](dt=2026-03-19/ht=13/5cacc654c5f6d4b758bd1bf3d3c783095cc8e3bc8afe777f1b75c473a18c5659.jpg)

![](dt=2026-03-19/ht=13/3ec4b85e6d6b2c9c05e06adcdc6ecdeb377ef9c27e0423bd244a293bbacd29c0.jpg)

![](dt=2026-03-19/ht=13/39cd69750a765a2cd21df26088fae770b87303760c56a8f2c4bcad7a2d52d5fb.jpg)

![](dt=2026-03-19/ht=13/4354c214f23b646cfb3b8485782c1bc31ce3121af95af50118d31a2eafe8a2d2.jpg)

![](dt=2026-03-19/ht=13/ef8a51691f69d1274d78ec89c2e08ddc3ccef3da8c9ef5d3ba47e2c962a1c2e0.jpg)

cytokines (Figure 14g). The strong immune memory effect was also observed in SPIN + US group, which help the mice against re-challenged tumors (Figure 14h). Furthermore, SPIN-mediated deep-tissue therapy was verified in the orthotopic pancreatic rabbit tumor model. Results from computed tomography (CT) imaging revealed that the SPIN + US group's tumor growth was obviously slower than that of the saline-injected group. SPINs exhibited a good paradigm of SDT in combination with multiple immunotherapeutic agents, providing a precision nanoplatform to temporospatial regulate cancer immunity. Besides, such a semiconducting polymer backbone was suitable for combining with other kinds of drugs, such as NLG919 and BMS-1166 (small molecular inhibitor of PD-1/PD-L1 pathway).[32b]

# 4. Summary and Perspective

SPNs, as a kind of emerging nanomaterial with high biocompatibility, have been widely used in combination cancer immunotherapy. The optimal size and stiffness of SPNs provide a distinctive advantage for long-term tumor accumulation compared with most traditional polymeric NPs and inorganic NPs, thereby minimizing off-target toxicity. SPNs can effectively kill tumors through PTT, PDT, or SDT under the irradiation of electromagnetic energy, inducing tumor ICD and initiating the crucial steps of the cancer-immunity cycle. Notably, SPNs can serve as a functional nanocarrier to encapsulate or conjugate various types of immune modulators, including but not limited to immune adjuvants, metabolic regulators, checkpoint inhibitors,

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (18 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/65ae21e0f2d5bc81eb5ed9579f7ca4710fdbc5c5cad97a23b52db9150d46578e.jpg)

![](dt=2026-03-19/ht=13/88d260564289c922afc54b8bee83f5b5472ee2e633cda045ca1ce9c6d56f93f3.jpg)

![](dt=2026-03-19/ht=13/d55b51fd1521556a1510a5886306a6401be56bcc0d51f476587e7f1d98e90033.jpg)

![](dt=2026-03-19/ht=13/aa56f1f64cfe09dec24153c9617568ac48d89d720ef7364a0c1cb91c05de4b8f.jpg)

![](dt=2026-03-19/ht=13/b37230c3ac848059769e6f870a5fcbe555693c8fb7a6b634905f13b90987b07c.jpg)

![](dt=2026-03-19/ht=13/b2a1a334131a18a5221e792da9426eceee73e1419e5fd30702236f74ec183ca8.jpg)

![](dt=2026-03-19/ht=13/a183cc129b61ce1dda1b972b3e6125353d691349b2ca9c5a577504d06be0bb12.jpg)

![](dt=2026-03-19/ht=13/2b17b9523869070ef8aa259a65852087ba0f2b162c1110850d65b2eb22427457.jpg)

![](dt=2026-03-19/ht=13/ec9ab8f4cd6b940a8d95b919741e092f783057308e950e2529625a8e6ca9b001.jpg)

![](dt=2026-03-19/ht=13/ed6a4da352268332d1dbd8ab6799e878bcb72231b6a861f9bd07fe3a356ac6a7.jpg)

TME modulators, and other ICD inducers. SPNs enable PTT, PDT, and SDT to synergize with a variety of drugs for combination cancer immunotherapy to amplify the immune response. In addition, immune modulators are often designed in prodrug formulations and combined with SPNs, enabling controlled release upon light, ultrasound, heat or tumor biomarker stimulation, effectively mitigating irAEs. However, when selecting therapeutic agents for combination therapy, researchers should avoid using compounds with low thermal stability or e
asily oxidizable structures. This precaution helps prevent potential interference with the thermal effects or ROS generated by PTT, PDT, or SDT.

Among various SPNs-based combination cancer immunotherapies, there are more studies on photothermal and photodynamic therapy, which may be because the research on optical-related therapy started earlier. By tuning the absorption wavelength of SPNs to the NIR-II window, the penetration of light can be significantly enhanced. In addition, some studies have engineered SPNs to exhibit both PTT and PDT ability, thereby further augmenting the efficacy of tumor eradication. Over the past three years, sono-based immunotherapy using SPNs has gained sub

stantial attention. Compared with light, ultrasound has a much stronger penetrating ability and can effectively treat tumors in deep organs. The tunable D-A structures in the SPN backbone determine whether it possesses one or more functions among PDT, PTT, or SDT. Although there is no precise theory to predict the function of a specific structure in a semiconducting polymer, we can still find some clues from reported examples.

For example, chemical structures that can enhance the vibrational relaxation of high-energy state electrons will be beneficial to PTT, while structures that can promote the ISC process and extend the lifetime of triplet state will be beneficial to PDT and SDT. In addition to the SPNs discussed in this review, there are some other organic materials that utilize the modulation of the D-A structure for efficient cancer therapy, such as aggregation-induced emission (AIE) materials.

[94] Recent studies have demonstrated that introducing AIE units into the backbone of semiconducting polymers can lead to superior optical properties.[95]

Despite the advantages discussed, there are still some issues that need to be addressed before SPNs can achieve large-scale clinical application. Although SPNs composed of bioinert

ADVANCED
SCIENCE NEWS
www.advancedsciencernews.com

ADVANCED MATERIALS www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (19 of 22)

© 2023 Wiley-VCH GmbH

components have demonstrated safety in vitro and in mice during treatment, their metabolic pathways in large animals have not been comprehensively studied. Additionally, a long-term toxicity evaluation is required before SPNs can be recognized as a clinically viable nanomedicine.

In conclusion, SPNs have demonstrated excellent anti-tumor ability in various animal models, inducing systemic immune activation and strong memory responses. In the future, the combination of different SPNs and novel immunomodulators is expected to increase the therapeutic potential of SPNs-based combination cancer immunotherapy and finally reach clinical translation.

# Acknowledgements

K.P. thanks Singapore National Research Foundation (NRF) (NRF-NRF107-2021-0005), Singapore Ministry of Education, Academic Research Fund Tier 2 (MOE-T2EP30220-0010 and MOE-T2EP30221-0004), and A*STAR SERC AME Programmatic Fund (SERC A18A8b0059) for the financial support.

# Conflict of Interest

The authors declare no conflict of interest.

# Keywords

cancer immunotherapy, semiconducting polymers, photothermal therapy, photodynamic therapy, sonodynamic therapy

Received: September 1, 2023

Revised: October 20, 2023

Published online: October 30, 2023

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (20 of 22)

© 2023 Wiley-VCH GmbH

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (21 of 22)

© 2023 Wiley-VCH GmbH

![](dt=2026-03-19/ht=13/c20e16d019f5b4887c5c7f5b24935d295e21b24675795d00aabfdca885515f39.jpg)

Jiayan Wu received his Ph.D. degree in the Changchun Institute of Applied Chemistry, Chinese Academy of Sciences in 2022. Then, he worked as a postdoctoral research fellow in Prof. Kanyi Pu's group at the School of Chemistry, Chemical Engineering and Biotechnology (CCEB), Nanyang Technological University (NTU). His current research focuses on designing nanomaterials for cancer immunotherapy.

![](dt=2026-03-19/ht=13/dbd7d6b8a91fbce81272c9282b140f1b5efe2ae2a7be3b3ece2c1f8bb1e7c805.jpg)

Kanyi Pu received his Ph.D. from the National University of Singapore in 2011 followed by a post-doctoral study at Stanford University School of Medicine. He is a Professor at School of Chemistry, Chemical Engineering and Biotechnology (CCEB) and Lee Kong Chain School of Medicine in Nanyang Technological University (NTU). His research encompasses chemistry, materials science, nanotechnology, and biophotonics with the focus on the development of molecular optical imaging probes and nanomedicines for early diagnosis and precision therapy.

ADVANCED SCIENCE NEWS

www.advancedsciencenews.com

ADVANCED MATERIALS

www.advmat.de

Adv.Mater.2024,36,2308924

2308924 (22 of 22)

© 2023 Wiley-VCH GmbH