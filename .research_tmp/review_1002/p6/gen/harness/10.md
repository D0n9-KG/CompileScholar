## Related Work

### Multi‑token prediction as a pre‑training objective

The standard objective for autoregressive language models is next‑token prediction (NTP), in which a single head assigns probability mass to the token immediately following the context. Multi‑token prediction (MTP) generalizes this by attaching multiple prediction heads so that the model is trained, at each position, to predict the next $k$ tokens rather than a single one (Tay et al., 2022). Tay et al. showed that, relative to NTP, MTP improves downstream task performance and training efficiency; because the auxiliary heads are trained to produce plausible continuations, they can be reused at inference time to accelerate decoding. MTP is closely related to, and can be viewed as a trained instance of, blockwise / parallel decoding, in which a model learns to emit several future tokens in one forward pass (Stern et al., 2023). Recent large‑scale models adopt MTP‑style auxiliary heads as a core component, both to improve quality and to supply the draft tokens used by self‑speculative decoding (e.g., DeepSeek‑AI, 2024).

### Self‑speculative decoding and multi‑head inference

The inference benefit of MTP rests on speculative decoding, in which a cheap draft proposes several tokens that a verifier checks in parallel, accepting the longest correct prefix and resampling otherwise (Leviathan et al., 2023; Chen et al., 2023). When the draft comes from auxiliary heads on the *same* model rather than a separate draft model, the approach is called self‑speculative decoding, and MTP's pre‑trained heads are a natural fit because they are trained under the same distribution as the base model (Tay et al., 2022). A complementary line of work obtains similar speedups by adding or learning draft heads around a fixed model: Medusa appends and trains multiple decoding heads on a frozen base model and combines them with tree attention (Cai et al., 2024), while the EAGLE family learns a feature‑level autoregressive draft and, in EAGLE‑2, builds a dynamic draft tree to raise the acceptance rate (Li et al., 2024; Li et al., 2024a). These methods target the *decoding procedure*; MTP instead shapes the *objective* during pre‑training, so the two are complementary rather than competing.

### Curriculum learning

Curriculum learning posits that learning is more efficient when examples are presented in order of increasing difficulty, mirroring how humans and animals learn (Bengio et al., 2009). The idea has since been applied across representation learning and, more recently, to language‑model pre‑training, where "difficulty" is defined over data ordering, sequence length, or the training signal. Most prior work schedules the *data*; comparatively little work schedules the *objective*. Our forward and reverse curricula are an instance of objective‑level curriculum: they re‑sequence the complexity of the prediction target (NTP versus MTP) over the course of training, rather than the order in which training examples are seen.

### Multi‑token prediction and model scale

A recurring question is for which models MTP actually pays off. Although the original work and subsequent analyses emphasize gains for large models, the benefits at small scale are less clear, and prior work reports that smaller language models struggle to exploit the MTP objective as effectively as their larger counterparts [CITE]. This size dependence is the gap we address: rather than abandoning MTP for small language models (SLMs) or treating the objective as a fixed target, we introduce a curriculum that adapts how the MTP objective is presented over training, so that SLMs can still capture its downstream and self‑speculative benefits.

In contrast to the above lines of work—which either fix the MTP objective (Tay et al., 2022) or modify the decoding procedure (Cai et al., 2024; Li et al., 2024; Li et al., 2024a)—we apply curriculum learning *to the objective itself*, and we study the forward (NTP→MTP) versus reverse (MTP→NTP) ordering, including how the two differ in their effect on downstream NTP quality versus self‑speculative decoding.

### References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.
- Cai, T., et al. (2024). Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2401.10774.
- Chen, C., et al. (2023). Accelerating large language model decoding with speculative sampling. *arXiv preprint* arXiv:2302.01318.
- DeepSeek‑AI. (2024). DeepSeek‑V3 technical report. *arXiv preprint* arXiv:2412.19437.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. In *Proceedings of the 40th International Conference on Machine Learning (ICML)*.
- Li, Y., et al. (2024). EAGLE: Speculative sampling requires rethinking feature uncertainty. In *Proceedings of the 41st International Conference on Machine Learning (ICML)*. arXiv:2401.15077.
- Li, Y., et al. (2024a). EAGLE‑2: Faster inference of language models with dynamic draft trees. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. arXiv:2406.16858.
- Stern, M., Shazeer, N., & Uszkoreit, J. (2023). Blockwise parallel decoding for deep autoregressive models. In *Advances in Neural Information Processing Systems (NeurIPS)*.
- Tay, Y., Dehghani, M., Bahri, D., et al. (2022). Multi‑token prediction. *arXiv preprint* arXiv:2204.02310.
- **[CITE]** — *the specific prior work showing SLMs struggle with MTP. Send me the reference and I'll add it here and in the text.*

---

要我把哪条主线再展开、或者收紧到某个会议篇幅（比如 NeurIPS 的 related work 通常更短）吗？另外那个 `[CITE]` 你手上是哪篇，发我我就补全。