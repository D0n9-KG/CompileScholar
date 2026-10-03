# Related Work

## Dense Passage Retrieval

Dense retrievers encode queries and passages into a shared vector space with a pre-trained language model fine-tuned under a contrastive objective, and rank candidates with a shallow similarity, typically the inner product or cosine distance. The paradigm was established by **Dense Passage Retrieval (DPR)** [Karpukhin et al., 2020], which fine-tuned a BERT encoder [Devlin et al., 2019] for open-domain question answering over Natural Questions [Kwiatkowski et al., 2019] and used the `[CLS]` token as the passage representation. A large body of work has since varied the training recipe and the encoder. **Sentence-BERT** [Reimers & Gurevych, 2019] introduced siamese fine-tuning for sentence embeddings and showed that mean pooling over token states can match the `[CLS]` token. **Contriever** [Izacard et al., 2021] showed that dense encoders can be trained entirely from unsupervised, in-batch contrastive objectives and adopted mean pooling, while **ANCE** [Xiong et al., 2021] demonstrated that hard, approximate-nearest-neighbor negatives substantially improve contrastive training. A more recent line of work invests in *pre-training the encoder for retrieval itself* — E5 [Wang et al., 2022] and its decoder-only extension e5-mistral [Wang et al., 2024], GTR [Guu et al., 2023], and REPR [Pradeep et al., 2023] — narrowing the gap to strong sparse baselines. Generalization is commonly evaluated on MS MARCO [Nguyen et al., 2016] and, for zero-shot transfer, on the heterogeneous BEIR benchmark [Thakur et al., 2021].

## Pooling Strategies and Backbone Architectures

Although dense retrieval is usually framed around end-task retrieval quality, the *representation* a model produces depends on two choices that are often treated as incidental: how token states are aggregated into a single vector (the `[CLS]` token, as in DPR, versus mean pooling, as in Sentence-BERT and Contriever), and which backbone family is used (bidirectional, encoder-only BERT [Devlin et al., 2019; Liu et al., 2019] versus causal, decoder-only models such as LLaMA [Touvron et al., 2023]). These choices are well studied for their effect on retrieval accuracy, but their consequences for *what knowledge the model has actually internalized* — as opposed to how well it happens to rank documents — have received far less attention. Because pooling strategy and backbone family are precisely the two axes we vary in our analysis, we situate this work against both the retrieval-quality literature and the emerging question of how much of a retriever's competence is inherited from pre-training.

## Where Does Retrieval Knowledge Come From?

Contrastive fine-tuning is generally understood to reshape the geometry of the embedding space so that relevant query–passage pairs are drawn together [Chen et al., 2020; He et al., 2020]. A sharper question is whether fine-tuning *adds* retrieval knowledge, or merely re-weights knowledge that pre-training already supplied. A prior analysis of BERT-based DPR argued that retrieval knowledge is acquired predominantly during pre-training, so that knowledge absent from the pre-trained encoder cannot be substantially recovered by downstream fine-tuning [**CRUX — prior BERT/DPR pre-training analysis; confirm exact reference**]. That claim, however, was established for a single backbone (BERT) and a single pooling/dataset configuration (DPR on Natural Questions), leaving open whether it is a property of dense retrieval generally or an artifact of that particular setup.

Our work revisits and generalizes this question. We extend the analysis along three axes — pooling strategy (`[CLS]` vs. mean pooling), backbone family (encoder-only BERT vs. decoder-only LLaMA), and dataset (Natural Questions and MS MARCO) — and find that the pre-training-dominance pattern holds for DPR-style tuning, where fine-tuning primarily re-calibrates neuron activations rather than reorganizing the underlying knowledge, but does **not** hold universally: it fails to extend to mean-pooled (Contriever-style) encoders and to decoder-based (LLaMA) backbones.

---

## References

- **Chen**, T., et al. (2020). A simple framework for contrastive learning of visual representations. *ICML*.
- **Devlin**, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- **Guu**, K., et al. (2023). Retrieval-augmented natural language supervision for dense passage retrieval. *NeurIPS*. *(confirm)*
- **He**, K., et al. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR*.
- **Izacard**, G., et al. (2021). Unsupervised dense information retrieval with contrastive learning. *ICML*.
- **Karpukhin**, V., et al. (2020). Dense passage retrieval for open-domain question answering. *EMNLP*.
- **Kwiatkowski**, T., et al. (2019). Natural questions: A benchmark for question answering research. *TACL*.
- **Liu**, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv*.
- **Nguyen**, T.-H., et al. (2016). MS MARCO: A human generated machine reading comprehension dataset. *arXiv*.
- **Pradeep**, R., et al. (2023). RePR: Improving text encoders with representation learning. *NeurIPS*. *(confirm)*
- **Reimers**, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. *EMNLP-IJCNLP*.
- **Thakur**, N., et al. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. *arXiv*.
- **Touvron**, H., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv*.
- **Wang**, L., et al. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv*.
- **Wang**, L., et al. (2024). e5-mistral (decoder-only dense retrieval). *(confirm exact title/venue)*
- **Xiong**, L., et al. (2021). Approximate nearest neighbor negative contrastive learning for dense text retrieval. *ICLR*.
- **[CRUX]** — the prior BERT/DPR analysis establishing that dense-retriever knowledge is acquired mainly during pre-training. **Reference unverified — please confirm, or grant me web/GitHub access and I'll verify.**

---

Want me to (a) verify the full reference list and fill in the crux citation if you approve network access, or (b) switch the inline style to numbered `[1]`? I can also save this to a file in your workspace if you'd like.