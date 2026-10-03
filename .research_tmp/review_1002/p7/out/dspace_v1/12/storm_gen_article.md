## Related Work

**Sampling Strategy Mechanism**
Prior research on event downsampling has predominantly relied on fixed heuristics or threshold-based strategies to reduce data volume. Several approaches employ bioinspired or neuromorphic filtering algorithms to perform data reduction, such as the bioinspired filtering method [3], the robust EvDownsampling technique [2], and neuromorphic downsampling of camera output [6]. Other works focus on spatial or spatio-temporal reductions, including spatial and temporal downsampling for classification [1], insect-inspired spatio-temporal methods [5], and spatial downscaling evaluated via Spiking Neural Networks [4]. Additionally, empirical analyses have been conducted to study the effects of subsampling rates on CNN performance [7]. In contrast to these rigid, non-adaptive mechanisms, our work introduces a probabilistic framework that models event importance through an event-importance probability density function (ePDF), allowing for arbitrary definition and adaptation to specific applications.

**Temporal Adaptability**
Existing downsampling methods generally operate using static or pre-defined temporal rules that do not dynamically adjust to real-time scene dynamics. For instance, the bioinspired [3], neuromorphic [6], and insect-inspired [5] approaches apply fixed filtering or downsampling logic, while spatial and temporal downsampling techniques [1] and robust reduction methods [2] typically rely on consistent, non-adaptive processing pipelines. Similarly, empirical subsampling studies [7] and SNN-based downscaling comparisons [4] evaluate performance under fixed sampling conditions rather than adapting on-the-fly. Our approach differs by operating in a purely online setting, estimating event importance on-the-fly from raw event streams to enable scene-specific adaptation in real-time.

**Task Compatibility Paradigm**
A significant portion of prior work on event subsampling requires task-specific adaptation, where downstream models must be retrained or fine-tuned to accommodate the reduced data. Notably, studies investigating the boundaries of event subsampling for video classification have focused on the impact of subsampling on CNN training stability and accuracy, implying a dependency on model-specific optimization [7]. While other methods propose general data reduction techniques [1, 2, 3, 4, 5, 6], they do not explicitly guarantee compatibility with models trained on original, full-resolution event streams without modification. In contrast, we introduce zero-shot event downsampling, ensuring that downsampled events remain directly usable for models trained on the original event stream without requiring any task-specific adaptation or retraining.

**Importance Prioritization Criterion**
Previous downsampling strategies often prioritize data reduction based on spatial density, temporal frequency, or generic bioinspired principles without explicitly targeting structural geometric features. Methods such as spatial downscaling [1, 4], robust data reduction [2], and bioinspired or neuromorphic filtering [3, 5, 6] generally aim to preserve overall information content or mimic biological systems, rather than specifically isolating structurally critical events. Empirical subsampling analyses [7] similarly evaluate global performance metrics without defining a specific structural importance criterion. Our work distinguishes itself by designing a contour-preserving ePDF that explicitly prioritizes structurally important events, ensuring that the most informative geometric features are retained under strict event-budget constraints.

## References

[1] Spatial and temporal downsampling in event-based visual classification
[2] EvDownsampling: a robust method for downsampling event camera data
[3] Less data same information for event-based sensors: A bioinspired filtering and data reduction algorithm
[4] Performance Comparison of DVS Data Spatial Downscaling Methods Using Spiking Neural Networks
[5] Insect-inspired Spatio-temporal Downsampling of Event-based Input
[6] Neuromorphic downsampling of event-based camera output
[7] Pushing the boundaries of event subsampling in event-based video
  classification using CNNs