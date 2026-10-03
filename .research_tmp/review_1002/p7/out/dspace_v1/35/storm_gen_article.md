## Related Work

**Generative Models for Protein Dynamics**
Recent advancements in generative modeling have significantly accelerated the study of protein dynamics, moving beyond single-structure prediction to conformational ensemble sampling. Early deep learning approaches focused on high-accuracy single structure prediction, such as AlphaFold [1], which laid the groundwork for subsequent generative methods. More recently, denoising diffusion models have emerged as a dominant backbone for this task, enabling efficient and accurate sampling of protein structure distributions. Notable examples include EigenFold, which utilizes a diffusion generative modeling framework [4], and Str2Str, a score-based framework for zero-shot conformation sampling [5]. Other diffusion-based approaches include ConfDiff, which employs force-guided SE(3) diffusion [7], and methods that investigate reinforcement learning strategies to optimize diffusion models for specific objectives [12]. While diffusion models are prevalent, alternative generative backbones have also been explored, including flow matching models like AlphaFlow and ESMFlow [8], variational autoencoders combined with language models in Structure Language Modeling (SLM) [9], and general deep generative models for learning molecular dynamics trajectories [10]. This paper builds upon the success of denoising diffusion models [3, 4, 5, 7, 12] but distinguishes itself by addressing the specific challenge of integrating physical feedback into this framework.

**Integration of Physical Feedback**
A critical challenge in data-driven protein modeling is the effective integration of physical supervision to ensure the physical plausibility of generated structures. Many recent generative approaches rely primarily on data-driven learning without explicit external physical constraints, such as EigenFold [4], Str2Str [5], and SLM [9], which learn distributions directly from crystallographic or simulation data. Other methods incorporate physical principles through different mechanisms: Boltzmann Generators use neural networks to learn coordinate transformations of equilibrium distributions [2], while Distributional Graphormer (DiG) transforms simple distributions to equilibrium distributions using deep learning [6]. ConfDiff integrates physical prior knowledge by using force fields to guide the diffusion process [7], and AlphaFlow leverages experimental data such as NMR and Cryo-EM [8]. Additionally, some works utilize molecular dynamics simulation data as a source of supervision [10] or employ reinforcement learning with human feedback or vision-language models [12]. Although the Rosetta all-atom energy function provides a robust physical baseline for macromolecular modeling [14], and force fields are used in coarse-grained dynamics [3], no prior cited work employs Energy-based Alignment (EBA) to efficiently calibrate conformational states based on energy differences from physical models, which is the core contribution of this paper.

**Optimization Strategies for Energy Integration**
The optimization strategy used to integrate physical energy terms into generative models is a key differentiator, as standard energy-based objectives often lead to intractable optimization problems. Most prior generative models for protein structures, including those based on diffusion [4, 5, 7, 12], flow matching [8], and VAEs [9], do not explicitly address the optimization of physical energy terms during the generation process, relying instead on the learned data distribution. Some approaches attempt to incorporate physical constraints through specific architectural designs, such as learning coordinate transformations for equilibrium states [2] or transforming distributions using graph neural networks [6]. In the context of optimization, reinforcement learning methods, such as policy gradient algorithms, have been applied to directly optimize diffusion models for downstream objectives [12]. However, these methods can be computationally expensive and unstable. This paper introduces an efficient calibration strategy that balances conformational states based on energy differences, offering a distinct and more tractable optimization approach compared to the policy gradient methods [12] or the lack of explicit physical optimization in purely data-driven models [4, 5, 8, 9, 10].

## References

[1] Highly accurate protein structure prediction with AlphaFold
[2] Boltzmann Generators -- Sampling Equilibrium States of Many-Body Systems
  with Deep Learning
[3] Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics
[4] EigenFold: Generative Protein Structure Prediction with Diffusion Models
[5] Str2Str: A Score-based Framework for Zero-shot Protein Conformation
  Sampling
[6] Towards Predicting Equilibrium Distributions for Molecular Systems with
  Deep Learning
[7] Protein Conformation Generation via Force-Guided SE(3) Diffusion Models
[8] AlphaFold Meets Flow Matching for Generating Protein Ensembles
[9] Structure Language Models for Protein Conformation Generation
[10] Generative Modeling of Molecular Dynamics Trajectories
[11] Learning to summarize from human feedback
[12] Training Diffusion Models with Reinforcement Learning
[13] Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model
[14] The Rosetta all-atom energy function for macromolecular modeling and design