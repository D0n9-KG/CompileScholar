Neural Machine Translation: A Review of Methods, Resources, and Tools 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2012.15515v1 [cs.CL] 31 Dec 2020 
 
 

# Neural Machine Translation: A Review of Methods, Resources, and Tools

 
 
 Zhixing Tan
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
    
 Shuo Wang
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
    
 Zonghan Yang
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
    
 Gang Chen
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
    
 Xuancheng Huang
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
    
 
 Maosong Sun
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
 Address: Beijing Academy of Artificial Intelligence
 
    
 Yang Liu
 
 Email: liuyang2011@tsinghua.edu.cn 
 
 Address: Department of Computer Science and Technology, Tsinghua University
 
 Address: Institute for AI Industry Research, Tsinghua University
 
 Address: Institute for Artificial Intelligence, Tsinghua University
 
 Address: Beijing National Research Center for Information Science and Technology
 
 Address: Beijing Academy of Artificial Intelligence
 
 Corresponding author: Corresponding author.
 

 Abstract 
 
 Machine translation (MT) is an important sub-field of natural language processing that aims to translate natural languages using computers. In recent years, end-to-end neural machine translation (NMT) has achieved great success and has become the new mainstream method in practical MT systems. In this article, we first provide a broad review of the methods for NMT and focus on methods relating to architectures, decoding, and data augmentation. Then we summarize the resources and tools that are useful for researchers. Finally, we conclude with a discussion of possible future research directions.

 
 
 
 Keywords: 
Neural machine translation , Attention mechanism , Deep learning , Natural language processing

 
 

## 1 Introduction

 
 Figure 1 : An overview of the NMT architecture, which consists of embedding layers, a classification layer, an encoder network, and a decoder network. We use different colors to distinguish different languages. 
 
 
 Machine Translation (MT) is an important task that aims to translate natural language sentences using computers. The early approach to machine translation relies heavily on hand-crafted translation rules and linguistic knowledge. As natural languages are inherently complex, it is difficult to cover all language irregularities with manual translation rules. With the availability of large-scale parallel corpora, data-driven approaches that learn linguistic information from data have gained increasing attention. Unlike rule-based machine translation, Statistical Machine Translation (SMT) [ 1 , 2 ] learns latent structures such as word alignments or phrases directly from parallel corpora. Incapable of modeling long-distance dependencies between words, the translation quality of SMT is far from satisfactory. With the breakthrough of deep learning, Neural Machine Translation (NMT) [ 3 , 4 , 5 , 6 ] has emerged as a new paradigm and quickly replaced SMT as the mainstream approach to MT.

 
 
 Neural machine translation is a radical departure from previous machine translation approaches. On the one hand, NMT employs continuous representations instead of discrete symbolic representations in SMT. On the other hand, NMT uses a single large neural network to model the entire translation process, freeing the need for excessive feature engineering. The training of NMT is end-to-end as opposed to separately tuned components in SMT. Besides its simplicity, NMT has achieved state-of-the-art performance on various language pairs [ 7 ] . In practice, NMT also becomes the key technology behind many commercial MT systems [ 8 , 9 ] .

 
 
 As neural machine translation attracts much research interest and grows into an area with many research directions, we believe it is necessary to conduct a comprehensive review of NMT. In this work, we will give an overview of the key ideas and innovations behind NMT. We also summarize the resources and tools that are useful and easily accessible. We hope that by tracing the origins and evolution of NMT, we can stand on the shoulder of past studies, and gain insights into the future of NMT.

 
 
 The remainder of this article is organized as follows: Section 2 will review the methods of NMT. We first introduce the basics of NMT, and then we selectively describe the recent progress of NMT. We focus on methods related to architectures, decoding, and data augmentation. Section 3 will summarize the resources such as parallel or monolingual corpora that are publicly available to researchers. Section 4 will describe tools that are useful for training and evaluating NMT models. Finally, we conclude and discuss future directions in Section 5 .

 
 
 

## 2 Methods

 
 As a data-driven approach to machine translation, NMT also embraces the probabilistic framework. Mathematically speaking, the goal of NMT is to estimate an unknown conditional distribution P ⁡ ( 𝒚 | 𝒙 ) P(\bm{y}|\bm{x}) given the dataset 𝒟 \mathcal{D} , where 𝒙 \bm{x} and 𝒚 \bm{y} are random variables representing source input and target output, respectively. We strive to answer the three basic questions of NMT:

 
 1. 
 
 Modeling. How to design neural networks to model the conditional distribution?

 

 2. 
 
 Inference. Given a source input, how to generate a translation sentence from the NMT model?

 

 3. 
 
 Learning. How to effectively learn the parameters of NMT from data?

 

 
 
 
 In 2.1 , we first describe the basic methods of NMT for addressing the above three questions. We then dive into the details of NMT architectures in 2.2 and introduce bidirectional inference and non-autoregressive NMTs in 2.3 . We discuss alternative training objectives and using monolingual data in 2.4 and 2.5 , respectively.

 
 
 Despite the great success, NMT is far from perfect. There are several theoretical and practical challenges faced by NMT. We survey the research progress of some important directions. We describe methods for open vocabulary in 2.6 , prior knowledge integration in 2.7 , and interpretability and robustness in 2.8 .

 
 
 Figure 2 : A running example of the beam-search algorithm. 
 
 

### 2.1 Overview of NMT

 

#### 2.1.1 Modeling

 
 Translation can be modeled at different levels, such as document-, paragraph-, and sentence-level. In this article, we focus on sentence-level translation. Besides, we also assume the input and output sentences are sequences. Thus the NMT model can be viewed as a sequence-to-sequence model. Assuming we are given a source sentence 𝐱 = { x 1 , … , x S } \mathbf{x}=\{x_{1},\ldots,x_{S}\} and a target sentence 𝐲 = { y 1 , … , y T } \mathbf{y}=\{y_{1},\dots,y_{T}\} . By using the chain rule, the conditional distribution can be factorized from left-to-right (L2R) as

 

 
 | 
 P ⁡ ( 𝒚 = 𝐲 | 𝒙 = 𝐱 ) = ∏ t = 1 T P ⁡ ( y t | y 0 , … , y t − 1 , x 1 , … , x S ) . P(\bm{y}=\mathbf{y}|\bm{x}=\mathbf{x})=\prod_{t=1}^{T}P(y_{t}|y_{0},\dots,y_{t-1},x_{1},\ldots,x_{S}). | 
 | 
 (1) | 
 

 NMT models which conform the Eq. ( 1 ) is referred to as L2R autoregressive NMT [ 3 , 4 , 5 , 6 ] , for the prediction at time-step t t is taken as a input at time-step t + 1 t+1 .

 
 
 Almost all neural machine translation models employ the encoder-decoder framework [ 4 ] . The encoder-decoder framework consists of four basic components: the embedding layers, the encoder and decoder networks, and the classification layer. Figure 1 shows a typical autoregressive NMT model using the encoder-decoder framework, which we shall use as an example. “ bos ” and “ eos ” are special symbols that mark the beginning and ending of a sentence, respectively.

 
 
 The embedding layer embodies the concept of continuous representation . It maps a discrete symbols x t x_{t} into a continuous vector 𝐱 t ∈ ℝ d \mathbf{x}_{t}\in\mathbb{R}^{d} , where d d denotes the dimension of the vector. The embeddings are then fed into later layers for more finer-grained feature extraction.

 
 
 The encoder network maps the source embeddings into hidden continuous representations. To learn expressive representations, the encoder must be able to model the ordering and complex dependencies that existed in the source language. Recurrent neural networks (RNN) are suitable choice for modeling variable-length sequences. With RNNs, the computation involves in encoder can be described as

 

 
 | 
 𝐡 t = RNN ENC ​ ( 𝐱 t , 𝐡 t − 1 ) . \mathbf{h}_{t}=\mathrm{RNN}_{\textrm{ENC}}(\mathbf{x}_{t},\mathbf{h}_{t-1}). | 
 | 
 (2) | 
 

 By iteratively applying the state transition function RNN ENC \mathrm{RNN}_{\textrm{ENC}} over the input sequence, we can use the final state 𝐡 S \mathbf{h}_{S} as the representation for the entire source sentence, and then feed it to the decoder.

 
 
 The decoder can be viewed as a language model conditioned on 𝐡 S \mathbf{h}_{S} . The decoder network extracts necessary information from the encoder output, and also models the long-distance dependencies between target words. Given the start symbol y 0 = bos y_{0}=\texttt{ bos } and the initial state 𝐬 0 = 𝐡 S \mathbf{s}_{0}=\mathbf{h}_{S} , the RNN decoder compresses the decoding history { y 0 , … , y t − 1 } \{y_{0},\ldots,y_{t-1}\} into a state vector 𝐬 t ∈ ℝ d \mathbf{s}_{t}\in\mathbb{R}^{d} :

 

 
 | 
 𝐬 t = RNN DEC ​ ( 𝐲 t − 1 , 𝐬 t − 1 ) . \displaystyle\mathbf{s}_{t}=\mathrm{RNN}_{\textrm{DEC}}(\mathbf{y}_{t-1},\mathbf{s}_{t-1}). | 
 | 
 (3) | 
 

 
 
 The classification layer predicts the distribution of target tokens. The classification layer is typically a linear layer with softmax activation function. Assuming the vocabulary of target language is 𝒱 \mathcal{V} , and | V | |V| is the size of the vocabulary. Given an decoder output 𝐬 t ∈ ℝ d \mathbf{s}_{t}\in\mathbb{R}^{d} , the classificaition layer first maps 𝐡 \mathbf{h} to a vector 𝐳 \mathbf{z} in the vocabulary space ℝ | V | \mathbb{R}^{|V|} with the linear map. Then the softmax function is used to ensure the output vector is a valid probability:

 

 
 | 
 softmax ⁡ ( 𝐳 ) = exp ⁡ ( 𝐳 ) ∑ i = 1 | V | exp ⁡ ( 𝐳 [ i ] ) , \displaystyle\mathrm{softmax}(\mathbf{z})=\frac{\exp(\mathbf{z})}{\sum_{i=1}^{|V|}\exp(\mathbf{z}_{[i]})}, | 
 | 
 (4) | 
 

 where we use 𝐳 [ i ] \mathbf{z}_{[i]} to denote the i i -th component in 𝐳 \mathbf{z} .

 
 
 

#### 2.1.2 Inference

 
 Given an NMT model and a source sentence 𝐱 \mathbf{x} , how to generate a translation from the model is an important problem. Ideally, we would like to find the target sentence 𝐲 \mathbf{y} which maximizes the model prediction P ⁡ ( 𝐲 | 𝒙 = 𝐱 ; 𝜽 ) P(\mathbf{y}|\bm{x}=\mathbf{x};\bm{\theta}) as the translation. However, due to the intractably large search space, it is impractical to find the translation with the highest probability. Therefore, NMT typically uses local search algorithms such as greedy search or beam search to find a local best translation.

 
 
 Beam search is a classic local search algorithm which have been widely used in NMT. Previously, beam search have been successfully applied in SMT.
The beam search algorithm keeps track of k k states during the inference stage. Each state is a tuple ⟨ y 0 ​ … ​ y t , v ⟩ \langle y_{0}\ldots y_{t},v\rangle , where y 0 ​ … ​ y t y_{0}\ldots y_{t} is a candidate translation, and v v is the log-probability of the candidate. At each step, all the successors of all k k states are generated, but only the top- k k successors are selected. The algorithm usually terminates when the step exceed a pre-defined value or k k full translation are found. It should be noted that the beam search will degrade into the greedy search if k = 1 k=1 .

 
 
 Algorithm 1 The beam search algorithm 
 
 
 1 
 t ← 1 t\leftarrow 1 ;
 
 
 2 
 𝒜 = { ⟨ 𝚋𝚘𝚜 , 0 ⟩ } \mathcal{A}=\{\langle\mathtt{ }\mathtt{bos}\mathtt{ },0\rangle\} ; ⊳ \triangleright The set of alive candidates 
 
 
 3 
 ℱ = { } \mathcal{F}=\{\} ; ⊳ \triangleright The set of finished candidates 
 
 
 4 
 while t 𝚖𝚊𝚡 ​ _ ​ 𝚕𝚎𝚗𝚐𝚝𝚑 t \mathtt{max\_length} do 
 
      
 5 
 𝒞 = { } \mathcal{C}=\{\} ;
 
      
 6 
 for ⟨ y 0 ​ … ​ y t − 1 , v ⟩ ∈ 𝒜 \langle y_{0}\ldots y_{t-1},v\rangle\in\mathcal{A} do 
 
           
 7 
 𝒑 ← NMT ⁡ ( y 0 ​ … ​ y t − 1 , 𝐱 ) \bm{p}\leftarrow\mathrm{NMT}(y_{0}\ldots y_{t-1},\mathbf{x}) ; 
 
           
 8 
 for w ∈ 𝒱 w\in\mathcal{V} do 
 
                
 9 
 y t ← w y_{t}\leftarrow w ;
 
                
 10 
 l ← log ⁡ ( 𝒑 ⁡ [ w ] ) l\leftarrow\log(\bm{p}[w]) ;
 
                
 11 
 𝒞 ← 𝒞 ∪ { ⟨ y 0 ​ … ​ y t , v + l ⟩ } \mathcal{C}\leftarrow\mathcal{C}\cup\{\langle y_{0}\ldots y_{t},v+l\rangle\} ;
 
           
 12 
 end for
 
      
 13 
 end for
 
      
 14 
 𝒞 ← TopK ⁡ ( 𝒞 , k ) \mathcal{C}\leftarrow\mathrm{TopK}(\mathcal{C},k) ; 
 
      
 15 
 for ⟨ y 0 ​ … ​ y t , v ⟩ ∈ 𝒞 \langle y_{0}\ldots y_{t},v\rangle\in\mathcal{C} do 
 
           
 16 
 if y t = = 𝚎𝚘𝚜 y_{t}==\mathtt{ }\mathtt{eos}\mathtt{ } then 
 
                
 17 
 ℱ ← ℱ ∪ { ⟨ y 0 ​ … ​ y t , v ⟩ } \mathcal{F}\leftarrow\mathcal{F}\cup\{\langle y_{0}\ldots y_{t},v\rangle\} ;
 
           
 18 
 else 
 
                
 19 
 𝒜 ← 𝒜 ∪ { ⟨ y 0 ​ … ​ y t , v ⟩ } \mathcal{A}\leftarrow\mathcal{A}\cup\{\langle y_{0}\ldots y_{t},v\rangle\} ;
 
           
 20 
 end if
 
      
 21 
 end for
 
      
 22 
 𝒜 ← TopK ⁡ ( 𝒜 , k ) \mathcal{A}\leftarrow\mathrm{TopK}(\mathcal{A},k) ; 
 
      
 23 
 ℱ ← TopK ⁡ ( ℱ , k ) \mathcal{F}\leftarrow\mathrm{TopK}(\mathcal{F},k) ; 
 
      
 24 
 t ← t + 1 t\leftarrow t+1 ;
 
 
 25 
 end while
 
 
 26 
 ⟨ y 0 ​ … ​ y t , v ⟩ ← Top ⁡ ( ℱ ) \langle y_{0}\ldots y_{t},v\rangle\leftarrow\mathrm{Top}(\mathcal{F}) ;
 
 
 27 
 return y 1 ​ … ​ y t y_{1}\ldots y_{t} 

 
 
 
 
 The pseudo-codes of the beam search algorithm are given in 1 . We also give a running example of the algorithm in Figure 2 .

 
 
 

#### 2.1.3 Training of NMT Models

 
 NMT typically uses maximum log-likelihood (MLE) as the training objective function, which is a commonly used method of estimating the parameters of a probability distribution. Formally, given the training set 𝒟 = { ⟨ 𝐱 ( s ) , 𝐲 ( s ) ⟩ } s = 1 S \mathcal{D}=\{\langle\mathbf{x}^{(s)},\mathbf{y}^{(s)}\rangle\}_{s=1}^{S} , the goal of training is to find a set of model parameters that maximize the log-likelihood on the training set:

 

 
 | 
 𝜽 ^ MLE = argmax 𝜽 { ℒ ⁡ ( 𝜽 ) } , \displaystyle\hat{\bm{\theta}}_{\mathrm{MLE}}=\mathop{\rm argmax}_{\bm{\theta}}\Big\{\mathcal{L}(\bm{\theta})\Big\}, | 
 | 
 (5) | 
 

 where the log-likelihood is defined as

 

 
 | 
 ℒ ⁡ ( 𝜽 ) = ∑ s = 1 S log ⁡ P ⁡ ( 𝐲 ( s ) | 𝐱 ( s ) ; 𝜽 ) . \displaystyle\mathcal{L}(\bm{\theta})=\sum_{s=1}^{S}\log P(\mathbf{y}^{(s)}|\mathbf{x}^{(s)};\bm{\theta}). | 
 | 
 (6) | 
 

 
 
 By the virtue of back-propagation algorithm, we can efficiently compute the gradient of ℒ \mathcal{L} with respect to 𝜽 \bm{\theta} . The training of NMT models usually adopts stochastic gradient search (SGD) algorithm. Instead of computing gradients on the full training set, SGD computes the loss function and graidents on a minibatch of the training set. The plain SGD optimzier updates the parameters of an NMT model with the following rule:

 
 
 

 
 | 
 𝜽 ← 𝜽 − α ​ ∇ 𝜽 ℒ ​ ( 𝜽 ) , \displaystyle\bm{\theta}\leftarrow\bm{\theta}-\alpha\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta}), | 
 | 
 (7) | 
 

 where α \alpha is the learning rate . With well-chosen learning rate, the parameters of NMT are guaranteed to converge into a local optima. In practice, instead of plain SGD optimizer, adaptive learning rate optimizers such as Adam [ 10 ] are found to greatly reduce the training time.

 
 
 
 

### 2.2 Architectures

 

#### 2.2.1 Evolution of NMT Architectures

 
 Since 2013, there are attempts to build a pure neural MT. Early NMT architectures such as RCTM [ 3 ] , RNNEncdec [ 4 ] , and Seq2Seq [ 5 ] adopt a fixed-length approach, where the size of source representation is fixed regardless the length of source sentences. These works typically use recurrent neural networks (RNN) as the decoder network for generating variable-length translation. However, it is found that the performance of this approach degrades as the length of the input sentence increases [ 11 ] . Two explanations can account for this phenomenon:

 
 1. 
 
 The fixed-length representations have become the bottleneck during the encoding process for long sentences [ 4 ] . As the encoder is forced to compress the entire source sentence into a set of fixed-length vectors, some important information may be lost in this process.

 

 2. 
 
 The longest path between the source words and target words is O ⁡ ( S + T ) O(S+T) , and it is challenging for neural networks to learn long-term dependencies [ 12 ] . Sutskever et al. [5] found that reverse the source sentence can significantly improve the performance of the fixed-length approach. By reversing the source sentence, the paths between the beginning words of source and target sentences are reduced, thus the optimization problem becomes easier.

 

 
 
 
 Figure 3 : At each decoding step, the attention mechanism dynamically generates a context vector based on the most relevant source representations for predicting the next target word. 
 
 
 Due to these limitations, later NMT architectures switch to variable-length source representations, where the length of source representations depends on the length of the source sentence. The RNNsearch architecture [ 6 ] introduces attention mechanism , which is an important approach to implementing variable-length representations. Figure 3 shows the comparison between fixed-length and variable-length approaches. By using the attention mechanism, the paths between any source and target words are within a constant length. As a result, the attention mechanism has eased optimization difficulty.

 
 
 With the breakthrough of deep learning, NMT with deep neural networks have attracted much research interest. Seq2Seq [ 5 ] is the first architecture demonstrate the potential of deep NMT. Later architectures such as GNMT [ 8 ] , ByteNet [ 13 ] , ConvSeq-2Seq [ 14 ] , and Transformer [ 15 ] all use multi-layered neural networks. ByteNet and ConvSeq2Seq have replaced RNNs with convolutional neural networks (CNN) in their architectures while Transformer relies entirely on self-attention networks (SAN). Both CNNs and SANs can reduce the sequential operations invovled in RNNs, and benefit from the parallel computation provided by modern devices such as GPU or TPU. Importantly, SAN can further reduce the longest path between two target tokens. We shall later describe the techniques for building these architectures.

 
 
 

#### 2.2.2 Attention Mechanism

 
 The introduction of attention mechanism [ 6 ] is a milestone in NMT architecture research. The attention network computes the relevance of each value vector based on queries and keys. This can also be interpreted as a content-based addressing sche-me [ 16 ] . Formally, given a set of m m query vectors 𝐐 ∈ ℝ m × d \mathbf{Q}\in\mathbb{R}^{m\times d} , a set of n n key vectors 𝐊 ∈ ℝ n × d \mathbf{K}\in\mathbb{R}^{n\times d} and associated value vectors 𝐕 ∈ ℝ n × d \mathbf{V}\in\mathbb{R}^{n\times d} , the computation of attention network involves two steps. The first step is to compute the relevance between keys and values, which is formally described as

 

 
 | 
 𝐑 = score ⁡ ( 𝐐 , 𝐊 ) , \mathbf{R}=\mathrm{score}(\mathbf{Q},\mathbf{K}), | 
 | 
 (8) | 
 

 where score ⁡ ( ⋅ ) \mathrm{score}(\cdot) is a scoring function which have several alternatives. 𝐑 ∈ ℝ m × n \mathbf{R}\in\mathbb{R}^{m\times n} is a matrix storing the relevance score between each keys and values. The next step is compute the output vectors. For each query vector, the corresponding output vector is expressed as a weighted sum of value vectors:

 

 
 | 
 Attention ⁡ ( 𝐐 , 𝐊 , 𝐕 ) \displaystyle\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) | 
 = softmax ⁡ ( 𝐑 ) ⋅ 𝐕 . \displaystyle=\mathrm{softmax}(\mathbf{R})\cdot\mathbf{V}. | 
 | 
 (9) | 
 

 Figure 4 depicts the two steps involved in the computation of attention mechanism.

 
 
 
 
 
 (a) Given a query vector and key vectors, the attention network first computes a weight vector through the scoring function. 
 
 
 (b) Each output vector is computed as a weighted sum of value vectors. 
 
 Figure 4 : Detailed computations involved in the attention mechanism. 
 
 
 Considering on the scoring function, the attention networks can be roughly classified into two categories: additive attention [ 6 ] and dot-product attention [ 17 ] . The additive attention models score \mathrm{score} through a feed-forward neural network:

 

 
 | 
 𝐑 [ i , j ] = 𝐯 ⊤ ​ tanh ⁡ ( 𝐖 s ​ 𝐐 [ i ] + 𝐔 s ​ 𝐊 [ j ] ) , \mathbf{R}_{[i,j]}=\mathbf{v}^{\top}\tanh(\mathbf{W}_{s}\mathbf{Q}_{[i]}+\mathbf{U}_{s}\mathbf{K}_{[j]}), | 
 | 
 (10) | 
 

 where 𝐖 s ∈ ℝ d × d , 𝐔 s ∈ ℝ d × d \mathbf{W}_{s}\in\mathbb{R}^{d\times d},\mathbf{U}_{s}\in\mathbb{R}^{d\times d} , and 𝐯 ∈ ℝ d × 1 \mathbf{v}\in\mathbb{R}^{d\times 1} are learnable parameters. On the other hand, the dot-product attention uses dot production to compute the matching score:

 

 
 | 
 𝐑 [ i , j ] = 𝐐 [ i ] ⊤ ​ 𝐊 [ j ] . \mathbf{R}_{[i,j]}=\mathbf{Q}_{[i]}^{\top}\mathbf{K}_{[j]}. | 
 | 
 (11) | 
 

 
 
 In practice, the dot-product attention is much faster than the additive attention. However, the dot-product attention is found to be less stable than the additive attention when d d is large [ 15 ] . Vaswani et al. [15] suspect that the dot-products grow large in magnitude for large values of d d , which may resulting extremely small gradients caused by the softmax function. To remedy this issue, they propose to scale the dot-products by 1 d \frac{1}{\sqrt{d}} .

 
 
 The attention mechanism is usually used as a part of the decoder network. Another type of attention network called self-attention network, is widely used in both the encoder and decoder of NMT. We shall describe self-attention and other variants of attention network later.

 
 
 

#### 2.2.3 RNNs, CNNs, and SANs

 
 There are many methods of building powerful encoders and decoders, which can roughly divide into three categories: the recurrent neural network (RNN) based methods, convolutional neural network (CNN) based methods, and self-attention network (SAN) based methods. There are several aspects we need to take into considerations for building an encoder and decoder:

 
 1. 
 
 Receptive field . We hope each output produced by the encoder and decoder can potentially encode arbitrary information in the input sequence.

 

 2. 
 
 Computational complexity . It is desirable to a use network with lower computational complexity.

 

 3. 
 
 Sequential operations . Too many sequential operations preclude the parallel computation within the sequence.

 

 4. 
 
 Position awareness . The network should distinguish the ordering presents in the sequence.

 

 
 Table 1 summarizes the computation as well as the above-mentioned aspects of typical RNN, CNN, and SAN.

 
 
 
 
 
 Layer | 
 Computation | 
 R.F. | 
 Complexity | 
 S.O. | 
 P.A. | 

 
 RNN | 
 𝐡 l , t = 𝐖𝐡 l − 1 , t + 𝐔𝐡 l , t − 1 \mathbf{h}_{l,t}=\mathbf{W}\mathbf{h}_{l-1,t}+\mathbf{U}\mathbf{h}_{l,t-1} | 
 ∞ \infty | 
 O ⁡ ( n ⋅ d 2 ) O(n\cdot d^{2}) | 
 O ⁡ ( n ) O(n) | 
 Yes | 

 
 CNN | 
 𝐡 l , t = ∑ i = 1 k 𝐖 ( i ) ​ 𝐡 l − 1 , t + i − ⌈ k + 1 2 ⌉ \mathbf{h}_{l,t}=\sum_{i=1}^{k}\mathbf{W}^{(i)}\mathbf{h}_{l-1,t+i-\lceil\frac{k+1}{2}\rceil} | 
 k k | 
 O ⁡ ( k ⋅ n ⋅ d 2 ) O(k\cdot n\cdot d^{2}) | 
 O ⁡ ( 1 ) O(1) | 
 Yes | 

 
 SAN | 
 𝐡 l , t = ∑ i = 1 n α l , i ​ 𝐡 l − 1 , i \mathbf{h}_{l,t}=\sum_{i=1}^{n}\alpha_{l,i}\mathbf{h}_{l-1,i} | 
 ∞ \infty | 
 O ⁡ ( n 2 ⋅ d ) O(n^{2}\cdot d) | 
 O ⁡ ( 1 ) O(1) | 
 No | 

 
 Table 1 : Comparisons between different neural network layers. We use R.F. to denote the receptive field, S.O. to denote the number of sequential operations, and P.A. to denote the position awareness of the layer. t t is the position in the sequence, l l is the layer number. For CNN, k k is the filter width and 𝐖 ( i ) \mathbf{W}^{(i)} is the weight of the i i -th filter. 
 
 
 
 
 
 (a) RNN. 
 
 
 (b) CNN with kernel width k = 3 k=3 . 
 
 
 (c) SAN. 
 
 Figure 5 : Overview of the computation diagram of RNN, CNN, and SAN. To be clarity, we use a node to denote the input or output vector of a specific layer. 
 
 
 
 
 
 (a) CNN with additional zero paddings. 
 
 
 (b) SAN with causal masking. 
 
 Figure 6 : The computation of CNN and SAN during decoding. 
 
 
 Figure 5 gives an overview of the ways of RNN, CNN, and SAN to encode sequences, respectively. In order to keep the auto-regressive property of NMT decoder during training, CNN and SAN furthur needs additional padding and masking to prevent the network from seeing future words. Figure 6 shows padding and masking used in CNN and SAN.

 
 
 As we can see in Figure 5 (a), RNNs are a family of sequential models that repeatedly apply the same state transition function to sequences. In theory, RNNs are among the most powerful family of neural networks [ 18 ] . However, it suffers from severe vanishing and exploding gradient problem [ 12 ] in practice. RNNs with gates, such as long short-term memory (LSTM) [ 19 ] and gated recurrent unit (GRU) [ 4 ] have been proposed to alleviate this problem. Another way to stabilize the training is to incorporate normalization layers, such as layer normalization [ 20 ] .

 
 
 
 
 
 (a) Deep RNN. 
 
 
 (b) Bidirectional RNN. 
 
 
 (c) Alternating RNN. 
 
 Figure 7 : Three extensions to RNNs. 
 
 
 Figure 7 shows three extensions to RNNs that are widely used in NMT literature. Deep RNNs is one important way to increase the expressive power of RNNs. However, training deep neural networks is challenging because it also faces the vanishing and exploding gradient problem. There are many ways to construct deep RNNs, and the most popular one is by stacking multiple RNNs with residual connections [ 21 ] . The residual connection is an important method to construct deep neural networks. Residual connections use the identity mappings as the skip connections, which is formally described as

 

 
 | 
 𝐲 = 𝐱 + f ⁡ ( 𝐱 ) , \mathbf{y}=\mathbf{x}+f(\mathbf{x}), | 
 | 
 (12) | 
 

 where 𝐱 \mathbf{x} , and 𝐲 \mathbf{y} are input and output, respectively. f f is the neural network. By using identity mappings, the gradient signal can directly propagate into lower layers. Bidirectional RNNs [ 6 ] use two RNNs to process the same sequence in opposite directions, and concatenating the results of both RNNs to be the final output. In this way, each output of bidirectional RNNs encodes all the tokens in the sequence. An alternative to bidirectional RNNs is alternating RNNs [ 22 ] , which consists of RNNs in opposite directions in adjacent layers.

 
 
 Besides the difficulty training of RNNs, another major drawback of RNNs is that RNNs are sequential models in nature, which cannot benefit from the parallel computations provided by modern GPUs. CNNs and SANs, however, which fully exploit the parallel computation within sequences, are widely used in newer NMT architectures.

 
 
 Convolutional neural network (CNN) was first introduced into NMT in 2013 [ 3 ] . However, it was not as successful as RNNs until 2017 [ 14 ] . The main obstacle for applying CNNs is its limited receptive field. Stacking L L CNNs with kernel width k k can increase the receptive field from k k to L ⋅ ( k − 1 ) + 1 L\cdot(k-1)+1 . The network needs to go deeper with large L L and adopt large kernel size k k to model long sentences. However, learning deep CNNs is challenging, and using large kernel size k k may significantly increase the complexity and parameters involved in CNNs.

 
 
 
 
 
 (a) 
 
 
 (b) 
 
 Figure 8 : Comparison between CNN and dilated CNN. (a) Two layers CNN with filter width k = 3 k=3 for each layer. (b) Dilated CNN with filter width k = 3 k=3 for all layers, dilation rate r = 1 r=1 in layer 1 and r = 2 r=2 in layer 2. 
 
 
 One solution to increase the receptive field without using a large k k is through dilation [ 13 ] . Figure 8 shows the comparison between plain CNN and dilated CNN. Plain CNN can be viewed as a special case of dilated CNN with a dilation rate r = 1 r=1 . The computation of dilated CNN is mathematically formulated as

 

 
 | 
 𝐡 l , t = ∑ i = 1 k 𝐖 ( i ) ​ 𝐡 l − 1 , t + ( i − ⌈ k + 1 2 ⌉ ) × r . \mathbf{h}_{l,t}=\sum_{i=1}^{k}\mathbf{W}^{(i)}\mathbf{h}_{l-1,t+(i-\lceil\frac{k+1}{2}\rceil)\times r}. | 
 | 
 (13) | 
 

 Stacking L L dilated CNNs whereby the dilation rates are doubled every layer, the receptive field increases to ( 2 L − 1 ) ⋅ ( k − 1 ) + 1 (2^{L}-1)\cdot(k-1)+1 . As a result, the receptive field grows exponentially with L L , as opposed to linearly with L L in plain CNN.

 
 
 
 
 
 (a) 
 
 
 (b) 
 
 Figure 9 : Comparison between CNN and depthwise CNN. Each node in the graph represents a neuron instead of a vector. (a) Plain CNN. We highlight the computation of the first neuron in the output vector. (b) Depthwise CNN. Note that the connections are significantly reduced compared with plain CNN. 
 
 
 Another solution is to reduce the computations involved in CNN. Depthwise convolution [ 23 ] reduces the complexity from O ⁡ ( k ​ d 2 ) O(kd^{2}) to O ⁡ ( k ​ d ) O(kd) by performing convolution independently over channels. Figure 9 depicts the comparison between CNN and depthwise CNN. The output of the depthwise convolution layer is defined as

 

 
 | 
 𝐡 l , t = ∑ i = 1 k 𝐰 ( i ) ⊙ 𝐡 l − 1 , t + i − ⌈ k + 1 2 ⌉ , \mathbf{h}_{l,t}=\sum_{i=1}^{k}\mathbf{w}^{(i)}\odot\mathbf{h}_{l-1,t+i-\lceil\frac{k+1}{2}\rceil}, | 
 | 
 (14) | 
 

 where 𝐰 ( i ) \mathbf{w}^{(i)} is the i i -th column of weight matrix 𝐖 ∈ ℝ k × d \mathbf{W}\in\mathbb{R}^{k\times d} . Lightweight convolution [ 24 ] further reduces the number of parameters of depthwise convolution through weight sharing.

 
 
 Figure 10 : An illustration of multi-head attention. 
 
 
 Self-attention network (SAN) [ 15 ] is a special case of attention network where the queries, keys, and values come from the same sequence. Similar to CNN, SAN is trivial to parallelize. Furthermore, Each output in SAN also has infinite receptive fields, which is the same with RNN. In SAN, the queries, keys, and values are typically obtained through a linear map of the input representations. The scaled dot-product self-attention mechanism can be formally described as

 

 
 | 
 Attention ⁡ ( 𝐐 , 𝐊 , 𝐕 ) = softmax ⁡ ( 𝐐𝐊 ⊤ d ) ​ 𝐕 . \mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d}})\mathbf{V}. | 
 | 
 (15) | 
 

 
 
 Multi-head attention [ 15 ] is an extended attention network with multiple parallel heads. Each head attends information from different subspace across value vectors. As a result, multi-head attention can perform more flexible transformations than the single-head attention. We give an illustration of multi-head attention in Figure 10 .

 
 
 The major disadvantage of SAN network is that it ignores the ordering of words in the sequence. To remedy this, SAN needs additional position encoding to differentiate orders. Vaswani et al. [15] proposed a sinusoid style position encoding, which is formulated as

 

 
 | 
 timing ​ ( t , 2 ​ i ) \displaystyle\textrm{timing}(t,2i) | 
 = sin ⁡ ( t / 10000 2 ​ i / d ) , \displaystyle=\sin(t/10000^{2i/d}), | 
 | 
 (16) | 
 
 
 | 
 timing ​ ( t , 2 ​ i + 1 ) \displaystyle\textrm{timing}(t,2i+1) | 
 = cos ⁡ ( t / 10000 2 ​ i / d ) , \displaystyle=\cos(t/10000^{2i/d}), | 
 | 
 (17) | 
 

 where t t is the position and i i is the dimension index. Another popular way of position encoding is to learn an additional position embedding. Finally, the position encoding is added to each word representation, so the same words with different positions can have different representations.

 
 
 Figure 11 : The Transformer architecture. 
 
 
 

#### 2.2.4 Comparison of Fundamental Architectures

 
 We take the state-of-the-art Transformer architecture [ 15 ] as an example to put all things together. Figure 11 shows the architecture of Transformer. The Transformer model relies solely on attention networks, with additional sinusoid-style position encoding added to input embedding. The Transformer network consists of a stack of 6 encoder layers and 6 decoder layers. Each encoder layer contains two sub-layers whereas each decoder layer contains three sub-layers. To stabilize optimization, Transformer uses residual connection and layer normalization in each sub-layer.

 
 
 
 
 
 Model | 
 Encoder | 
 Decoder | 
 Complexity | 
 V.R. | 
 Path E {}_{\textrm{E}} | 
 Path D {}_{\textrm{D}} | 

 
 RCTM 1 [ 3 ] | 
 CNN | 
 RNN | 
 O ⁡ ( S 2 + T ) O(S^{2}+T) | 
 No | 
 S S | 
 T T | 

 
 RCTM 2 [ 3 ] | 
 CNN | 
 RNN | 
 O ⁡ ( S 2 + T ) O(S^{2}+T) | 
 Yes | 
 S S | 
 T T | 

 
 RNNEncdec/Seq2Seq [ 4 , 5 ] | 
 RNN | 
 RNN | 
 O ⁡ ( S + T ) O(S+T) | 
 No | 
 S + T S+T | 
 T T | 

 
 RNNsearch [ 6 ] | 
 RNN | 
 RNN | 
 O ⁡ ( S ​ T ) O(ST) | 
 Yes | 
 1 1 | 
 T T | 

 
 ByteNet [ 13 ] | 
 CNN | 
 CNN | 
 O ⁡ ( S + T ) O(S+T) | 
 Yes | 
 c c | 
 c c | 

 
 ConvSeq2Seq [ 14 ] | 
 CNN | 
 CNN | 
 O ⁡ ( S ​ T ) O(ST) | 
 Yes | 
 1 1 | 
 c c | 

 
 Transformer [ 15 ] | 
 SAN | 
 SAN | 
 O ⁡ ( S 2 + S ​ T + T 2 ) O(S^{2}+ST+T^{2}) | 
 Yes | 
 1 1 | 
 1 1 | 

 
 Table 2 : Comparison of fundamental architectures. V.R. denotes whether the architecture employs variable representation. Path E {}_{\textrm{E}} denotes the longest path between the source and target tokens. Path D {}_{\textrm{D}} denotes the longest path between two target tokens. 
 
 
 We summarize the comparison of fundamental NMT architectures in Table 2 . We highlight several important aspects of these fundamental architectures.

 
 
 
 
 
 (a) Bidirectional inference. 
 
 
 (b) Non-autoregressive decoding. 
 
 Figure 12 : Comparisons of different decoding strategies. (a) Bidirectional decoding: generates a sentence in both left-to-right (L2R) and right-to-left (R2L) directions; (b) Non-autoregressive (NAR) decoding: generates a sentence at one time. 
 
 
 
 

### 2.3 Bidirectional Inference and Non-autoregressive NMT

 
 The dominate approach to NMT factorizes the conditional probability P ⁡ ( 𝐲 | 𝐱 ) P(\mathbf{y}|\mathbf{x}) from left to right (L2R) auto-repressively. However, the factorization of the distribution is not unique. Researchers [ 25 , 26 , 27 , 28 ] have found that models with right-to-left (R2L) factorization are complementary to L2R models. The bidirectional inference is an approach to simultaneously generating translation with both L2R and R2L decoders. In addition to auto-regressive approaches where each output word on previously generated outputs, non-autoregressive NMTs [ 29 ] avoids this auto-regressive property and produces outputs in parallel, allowing much lower latency during inference.

 
 

#### 2.3.1 Bidirectional Inference

 
 Ignoring the future context is another obvious weakness of AR decoding. Thus, a natural idea is that the quality of translation will be improved if autoregressive models can “know” the future information. From this perspective, many approaches have been proposed to improve translation performance by exploring the future context. Some researchers proposed to model both past and future context [ 30 , 31 , 32 ] and some others also found that L2R and R2L autoregressive models can generate complementary translations [ 25 , 26 , 27 , 28 ] . For instance, Zhou et al. [28] analyzed the translation accuracy of the first and last 4 tokens for L2R and R2L models, respectively. The statistical results show that, in Chinese-English translation, L2R performs better in the first 4 tokens while R2L translates better in the last 4 tokens.

 
 
 Based on the findings mentioned above, a number of methods have been proposed to combine the advantages of L2R and R2L decoding. These approaches are collectively referred to as bidirectional decoding. Bidirectional decoding based methods can be mainly fall into four categories [ 33 ] : (1) agreement between L2R and R2L [ 25 , 34 , 35 ] , (2) rescore with bidirectional decoding [ 25 , 36 ] , (3) asynchronous bidirectional decoding [ 27 , 37 ] , and (4) synchronous bidirectional decoding [ 28 , 38 , 39 ] .

 
 
 Mathematically, the L2R translation order is rather arbitrary, and other arrangements such as R2L factorization are equally correct:

 

 
 | 
 P ⁡ ( 𝐲 | 𝐱 ) = ∏ t = 1 T P ⁡ ( y t | 𝐲 t , 𝐱 ) ⏟ L2R model = ∏ t = 1 T P ⁡ ( y t | 𝐲 t , 𝐱 ) ⏟ R2L model . P(\mathbf{y}|\mathbf{x})=\underbrace{\prod\limits_{t=1}^{T}P(y_{t}|\mathbf{y}_{ t},\mathbf{x})}_{\text{L2R model}}=\underbrace{\prod\limits_{t=1}^{T}P(y_{t}|\mathbf{y}_{ t},\mathbf{x})}_{\text{R2L model}}. | 
 | 
 (18) | 
 

 Based on this theoretical assumption, Liu et al. [25] , Yang et al. [34] , and Zhang et al. [35] proposed joint training schemes in which each direction is used as a regularizer for the other direction. Empirical results show that these methods can lead to significant improvements compared with standard L2R and R2L models.

 
 
 Another common scheme to combine L2R and R2L translations is rescoring (also known as reranking). A strong L2R model firstly produces an n n -best list of translations, and then an R2L model rescores each translation in the n n -best list [ 25 , 36 , 40 ] . As the scores from L2R and R2L directions are based on complementary models, the quality of translation can be improved by rescoring.
Recently, Zhang et al. [27] introduced a new strategy to exploit both L2R and R2L models. They named this method asynchronous bidirectional decoding (ASBD), which first produces outputs (hidden states) by an R2L model and then uses these outputs to optimize the L2R model. ASBD can be done in three steps: The first step is to train a R2L model with bilingual corpora. The second step is to obtain outputs for each given source sentence using the trained R2L model. Finally, the output of R2L model is used as the additional context with the training data to train the L2R model. Thanks to incorporating the future information from the R2L model, the performance of L2R model can be substantially improved.

 
 
 Although ASBD improves the quality of translation, it also incurs other problems. The L2R and R2L models are trained separately so that they have no chance to interact with each other. Besides, the L2R model translates source sentences based on the outputs of an R2L model, this degrades the efficiency of inference. To address these problems, Zhou et al. [28] further proposed a synchronous bidirectional decoding (SBD) method which generates translations using both L2R and R2L inference synchronously and interactively. Specifically, SBD uses a new synchronous attention model to allow both L2R and R2L models “communicating” with each other. As shown in Figure 12(a) , the dotted arrows illustrate interactions between L2R and R2L decoding. Zhou et al. [28] also designed a variant of the standard beam search algorithm to hold L2R and R2L decoding concurrently. The idea behind this algorithm is to maintain that each half beam contains L2R and R2L predictions, respectively. Empirical results show that SBD can significantly improve performance with a slight cost to decoding speed.

 
 
 Mehri and Sigal [41] proposed a novel middle-out decoder architecture that begins from an initial middle-word and simultaneously expands the sequence in both L2R and R2L directions. Zhou et al. [38] also proposed a similar method that allows L2R and R2L inferences to start concurrently from the left and right sides, respectively. Both L2R and R2L inferences terminate at the middle position. Extensive experiments demonstrate that this method can improve not only the accuracy of translation but also decoding efficiency.

 
 
 

#### 2.3.2 Non-autoregressive NMTs

 
 To reduce the latency during inference, Gu et al. [29] first proposed the non-autoregressive NMT (NAT) to generate the target words in parallel. Formally, given the source sentence 𝐱 \mathbf{x} , the probability of the target sentence 𝐲 \mathbf{y} is modeled as follows:

 

 
 | 
 P 𝒩 ​ 𝒜 ​ ( 𝐲 | 𝐱 ; 𝜽 ) = P L ​ ( T | 𝐱 ; 𝜽 ) ⋅ ∏ t = 1 T P ⁡ ( y t | 𝐱 ; 𝜽 ) , P_{\mathcal{NA}}(\mathbf{y}|\mathbf{x};\bm{\theta})=P_{L}(T|\mathbf{x};\bm{\theta})\cdot\prod\limits_{t=1}^{T}P(y_{t}|\mathbf{x};\bm{\theta}), | 
 | 
 (19) | 
 

 where P 𝒩 ​ 𝒜 ​ ( 𝐲 | 𝐱 ; 𝜽 ) P_{\mathcal{NA}}(\mathbf{y}|\mathbf{x};\bm{\theta}) is the NAT model, P L ​ ( T | 𝐱 ; 𝜽 ) P_{L}(T|\mathbf{x};\bm{\theta}) is a length sub-model to determine the length of target sentence, and 𝜽 \bm{\theta} denotes the set of model parameters.

 
 
 How to predict the length of target sentence (i.e., P L ​ ( T | 𝐱 ; 𝜽 ) P_{L}(T|\mathbf{x};\bm{\theta}) in Eq. ( 19 )) is critical for NAT. Gu et al. [29] proposed a fertility predictor to predict the length of translation. The fertility of a word in the source side determines how many target words it is aligned to. The fertility predictor can be denoted as

 

 
 | 
 P F ​ ( 𝐟 | 𝐱 ; 𝜽 ) = ∏ s = 1 S P ⁡ ( f s | 𝐱 ; 𝜽 ) , P_{F}(\mathbf{f}|\mathbf{x};\bm{\theta})=\prod\limits_{s=1}^{S}P(f_{s}|\mathbf{x};\bm{\theta}), | 
 | 
 (20) | 
 

 where 𝐟 = { f 1 , ⋯ , f S } \mathbf{f}=\{f_{1},\cdots,f_{S}\} is the fertility of the source sentence that consists of S S words, and 𝜽 \bm{\theta} is the set of parameters. At the training phase, the gold fertility of each sentence pair in the training data can be obtained by a word alignment system. At the inference phase, the length of the target sentence can be determined by the fertility predictor:

 

 
 | 
 T ^ \displaystyle\hat{T} | 
 = ∑ s = 1 S f s ^ , \displaystyle=\sum\limits_{s=1}^{S}\hat{f_{s}}, | 
 | 
 (21) | 
 
 
 | 
 f s ^ \displaystyle\hat{f_{s}} | 
 = argmax f s P ⁡ ( f s | 𝐱 ; 𝜽 ^ ) , \displaystyle=\mathop{\rm argmax}\limits_{f_{s}}P(f_{s}|\mathbf{x};\hat{\bm{\theta}}), | 
 | 
 (22) | 
 

 where T ^ \hat{T} is the number of words in the translation of the source sentence 𝐱 \mathbf{x} , and 𝜽 ^ \hat{\bm{\theta}} is the set of learned parameters.

 
 
 Different from autoregressive NMT models that take the previous words (ie., 𝐲 t \mathbf{y}_{ t} ) as the input to predict the next target word y t y_{t} , NAT lacks such history information. Gu et al. [29] also noticed that missing the input of the decoder can greatly impair translation quality. Thus, the authors proposed to copy each source token to the decoder, and the times each input token to be copied is its “feritility”. Gu et al. [29] also used knowledge distillation [ 42 ] , which employs strong autoregressive models as the “teachers” to improve the performance. Knowledge distillation has proven necessary for non-autoregressive translation [ 43 , 29 , 44 , 45 , 46 ] .

 
 
 Despite the promising success of NAT, which can boost the decoding efficiency by about 15 times speedup compared with vanilla Transformer, NAT suffers from considerable quality degradation. Recently, many methods have been proposed to narrow the performance gap between non-autoregressive NMT and autoregressive NMT [ 44 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 46 , 54 ] .

 
 
 To take advantage of both autoregressive NMT and non-autoregressive NMT, Wang et al. [47] designed a semi-autoregressive Transformer (SAT) model. SAT keeps the autoregressive property in global but performs parallel translation in local. Specifically, SAT produces K K sequential words per time-step independently to others. Consequently, SAT can balance autoregressive NMT ( K = 1 K=1 ) and non-autoregressive NMT ( K = T K=T ) by adjusting the value of K K . Akoury et al. [53] moved a further step to propose a syntactically supervised Transformer (SynST), which first autoregressively predicts a chunked parse tree and then generates all words in one shot conditioned on the predicted parse.

 
 
 A critical issue of NAT is that NAT copies the source words as the input of the decoder while ignores the difference between the source and target semantics. To address this problem, Guo et al. [48] proposed to use a phrase table to covert source words to target words. They adopt a maximum match algorithm to greedily segment the source sentence into several phrases and then map these source phrases into target phrases by retrieving a pre-defined phrase table. Thanks to the enhanced decoder input, translation quality is significantly improved.

 
 
 Inspired by the mask-predict task proposed by Devlin et al. [55] , Ghazvininejad et al. [46] introduced a conditioned masked language model (CMLM) to generate translation by iterative refinement. CMLM trains the conditioned language model using a mask-predict manner and produces target sentences by iterative decoding during inference. Specifically, in the training phase, CMLM first randomly masks the words in the target sentence and then predicts these masked words. In the inference, CMLM generates the entire target sentence in a preset number of decoding iteration N N . At iteration n ∈ [ 1 , N ] n\in[1,N] , the decoder input is the entire target sentence with T − T ⁡ ( N − t + 1 ) N T-\frac{T(N-t+1)}{N} words masked. The decoding process starts with a fully-masked target sentence and the words with the lowest prediction probabilities will be masked. With a proper number of decoding iteration, CMLM can effectively close the gap with fully autoregressive models and maintain the decoding efficiency.

 
 
 
 

### 2.4 Alternative Training Objectives

 
 NMT trained with maximum likelihood estimation or MLE have achieved state-of-the-art results on various language pairs [ 7 ] . Despite the remarkable success, Ranzato et al. [56] indicate two drawbacks of MLE for NMT. First, NMT models are not exposed to their errors during training, which is referred to as the exposure bias problem. Second, MLE is defined at word-level rather than sentence-level. Due to these limitations, researchers have investigated several alternative objectives.

 
 
 Ranzato et al. [57] introduce Mixed Incremental Cross-Entropy Reinforce (MIXER) for sequence-level training. The MIXER algorithm borrows ideas from reinforcement learning for backpropagating gradients from non-differentiable metrics such as BLEU. Wu et al. [58] give a study of reinforcement learning for NMT. Shen et al. [59] proposed minimum risk training (MRT) to alleviate the problem. In MRT, the risk is defined as the expected loss with respect to the posterior distribution:

 

 
 | 
 ℒ ⁡ ( 𝜽 ) \displaystyle\mathcal{L}(\bm{\theta}) | 
 = ∑ s = 1 S 𝔼 𝐲 | 𝐱 ( s ) ; 𝜽 ​ [ Δ ⁡ ( 𝐲 , 𝐲 ( s ) ) ] \displaystyle=\sum_{s=1}^{S}\mathbb{E}_{\mathbf{y}|\mathbf{x}^{(s)};\bm{\theta}}\left[\Delta(\mathbf{y},\mathbf{y}^{(s)})\right] | 
 | 
 (23) | 
 
 
 | 
 | 
 = ∑ s = 1 S ∑ 𝐲 ∈ 𝒴 ⁡ ( 𝐱 ( s ) ) P ⁡ ( 𝐲 | 𝐱 ( s ) ; 𝜽 ) ​ Δ ​ ( 𝐲 , 𝐲 ( s ) ) , \displaystyle=\sum_{s=1}^{S}\sum_{\mathbf{y}\in\mathcal{Y}(\mathbf{x}^{(s)})}P(\mathbf{y}|\mathbf{x}^{(s)};\bm{\theta})\Delta(\mathbf{y},\mathbf{y}^{(s)}), | 
 | 
 (24) | 
 

 where 𝒴 ⁡ ( 𝐱 ( s ) ) \mathcal{Y}(\mathbf{x}^{(s)}) is a set of all possible candidate translations for 𝐱 ( s ) \mathbf{x}^{(s)} ; Δ ⁡ ( 𝐲 , 𝐲 ( s ) ) \Delta(\mathbf{y},\mathbf{y}^{(s)}) measures the difference between model prediction and gold-standard. Shen et al. [59] indicate three advantages for MRT over MLE. Firstly, MRT direct optimize NMT with respect to evaluation metrics. Secondly, MRT can incorporate with arbitrary loss functions. Finally, MRT is transparent to architectures and can be applied to any end-to-end NMT systems. MRT achieves significant performance improvements than MLE training for RNNSearch. However, recent literature [ 60 ] has also pointed out the weakness of reinforcement learning for NMT, including discussion about optimization goals and difficulty in convergence.

 
 
 Efforts on improving training objectives reveal the art of translating motivation into functions and rewrite the conventional loss function with them or integrating them into it as regularizers. A collection of classical structured prediction losses are reviewed and compared in Edunov et al. [61] , including MLE, sequence-level MLE, MRT, and max-margin learning. Yang et al. [62] leveraged the idea of max-margin learning in reducing word omission errors in NMT. They artificially constructed negative examples by omitting words in target reference sentences, forcing the NMT model to assign a higher probability to a ground-truth translation and a lower probability to an erroneous translation. Wieting et al. [63] aimed at improving the semantic similarity between ground-truth references and translation outputs from NMT systems. They proposed to use a margin-based loss as an alternative reward function, encouraging NMT models to output semantically correct hypotheses even if they mismatch with the reference in the lexicon. Chen et al. [64] aimed at improving model capability of capturing long-range semantic structure. They proposed to explicitly model the source-target alignment with optimal transport (OT), and couple the OT loss with the MLE loss function. Kumar and Tsvetkov [65] aimed at improving model efficiency and reducing the memory footprint of NMT models. Observing that the softmax layer usually takes considerable memory usage and the longest computation time, they proposed to replace the softmax layer with a continuous embedding layer, using Von Mises-Fisher distribution to implement soft ranking as softmax layer functions. As a result, the novel probabilistic loss enables NMT models to train much faster and handle very large vocabularies.

 
 
 

### 2.5 Using Monolingual Data and Unsupervised NMT

 
 The amount of parallel data significantly affects the training of parameters as NMT is found to be data-hungry [ 66 ] . Unfortunately, large-scale parallel corpora are not available for the vast majority of language pairs. In contrast, monolingual corpora are abundant and much easy to obtain. As a result, it is important to augment the training set with monolingual data.

 
 

#### 2.5.1 Using Monolingual Data

 
 As NMT is trained in an end-to-end way, it raises the difficulties in taking advantage of monolingual data. In the past few years researchers have proposed various methods to make use of the source- and target-side monolingual data in neural machine translation.

 
 
 For target-side monolingual data, early attempts try to incorporate a language model trained on large-scale monolingual data into NMT. Gulcehre et al. [67] proposed two ways to integrate a language model. One way is called shallow fusion , which uses a language model during decoding to rescore the n n -base list. Another way is called deep fusion , which combines the decoder and language model with a controller mechanism. However, the improvements of these approaches are limited.

 
 
 Another way to use target-side monolingual data is called Back-translation (BT) [ 68 ] . BT can make use of target-side monolingual data without changing the architecture of NMT. In Sennrich et al. [68] , they first trained a target-to-source translation model using the parallel corpus. Then, the target-side monolingual data are used to build a synthetic parallel corpus, whose source sides are generated by the target-to-source translation model. Finally, the concatenation of parallel corpus and synthetic parallel corpus is used to learn a source-to-target translation model. Although the architecture and decoding algorithm is kept unchanged, the monolingual data is fully utilized to improve the translation quality. The authors attributed the effectiveness of using monolingual data to domain adaptation effects, reductions of overfitting, and improved fluency.
BT has shown to be the most simple and effective method to leverage target-side monolingual data [ 68 , 69 ] . It is especially useful when only a small number of parallel data is available [ 70 ] . Imamura et al. [71] found that the diversities of source sentences affect the performance of BT. In the meantime, Edunov et al. [72] analyzed BT extensively and showed that noised-BT, which builds a synthetic corpus by sampled source sentences or noised output of beam-search, leads to higher accuracy. Caswell et al. [73] investigated the role of noise in noised-BT. They revealed that the noises work in a way of making the model be able to distinguish the synthetic data and genuine data. The model can further take advantages of helpful signal and ignore harmful signal. As a result, they proposed a simple method called tagged-BT, which appends a preceding tag (e.g., BT ) to every synthetic source sentence. Wang et al. [74] proposed to consider uncertainty-based confidence to help NMT models distinguish synthetic data from authentic data.

 
 
 Besides target-side monolingual data, source-side monolingual data are also important resources to improve the translation quality of semi-supervised neural machine translation. Zhang and Zong [75] explored two ways to leverage source-side monolingual data. The former one is knowledge distillation (also called self-training), which utilizes the source-to-target translation model to build a synthetic parallel corpus. The latter is multi-task learning that simultaneously learns translation and source sentence reordering tasks.

 
 
 There are many works to make use of both source- and target-side monolingual data. Hoang et al. [76] found that the translation quality of the target-to-source model in BT matters and then proposed iterative back-translation, making the source-to-target and target-to-source to enhance each other iteratively. Cheng et al. [77] presented an approach to train a bidirectional neural machine translation model, which introduced autoencoders on the monolingual corpora with source-to-target and target-to-source translation models as encoders and decoders by appending a reconstruction term to the training objective. He et al. [78] proposed a dual-learning mechanism, which utilized reinforcement learning to make the source-to-target and target-to-source model to teach each other with the help of source- and target-side language models. Zheng et al. [79] proposed a mirror-generative NMT model to integrate source-to-target and target-to-source NMT models and both-side language models, which can learn from monolingual data naturally.

 
 
 
 
 
 (a) Unidirectional language model pre-training. 
 
 
 (b) Bi-directional language model pre-training. 
 
 
 (c) Sequence-to-sequence model pre-training. 
 
 Figure 13 : Three commonly used ways for pre-training. 
 
 
 Pre-training is an alternative way to utilize monolingual data for NMT, which is shown to be beneficial by further combining with back-translation in the supervised and unsupervised NMT scenario [ 80 , 81 ] . Recently, pre-training has attracted tremendous attention because of its effectiveness on low-resource language understanding and language generation tasks [ 82 , 83 , 55 ] . Researchers found that models trained on large-scale monolingual data can learn linguistics knowledge [ 84 ] . These knowledge can be transferred into downstream tasks by initializing the task-oriented models with the pre-trained weights. Language modeling is a commonly used pre-training method. The drawback of standard language modeling is that it is unidirectional, which may be sub-optimal as a pre-training technique. Devlin et al. [55] proposed a masked pre-training language model (MLM) objective, which allows the model to make full use of context at the price of losing the ability to generate sequences. Combining language modeling and masking with sequence-to-sequence models, however, do not suffer from these limitations [ 80 , 85 , 81 ] .

 
 
 Edunov et al. [86] fed the output representations of ELMO [ 82 ] to the encoder of NMT. Zhu et al. [87] proposed to fuse extracted representations into each layer of encoder and decoder through attention mechanism. Song et al. [80] proposed to pre-train a sequence-to-sequence model first, and then finetune the pre-trained model on translation task directly. BART [ 85 ] took various noising method to pre-train a denoising sequence-to-sequence model and then finetune the model with an additional encoder that replaces the word embeddings of the pre-trained encoder. Liu et al. [88] proposed mBART which is trained by applying BART to large-scale monolingual data across many languages.

 
 
 

#### 2.5.2 Unsupervised NMT

 
 Due to insufficient parallel corpus, it is not feasible to use supervised methods to train an NMT model on many language pairs. Unsupervised neural machine translation aims to obtain a translation model without using parallel data. Apparently, unsupervised machine translation is much more difficult than the supervised and semi-supervised settings.

 
 
 Unsupervised neural machine translation is composed of three parts. First, by the virtue of recent advances on unsupervised cross-lingual embeddings [ 89 , 90 ] and word-by-word translation systems [ 91 ] , the unsupervised translation models can be initialized by weak translation models with fundamental cross-lingual information. Second, denoising autoencoders [ 92 ] are used to embed the sentences into dense latent representations. The sentences of different languages are assumed to be embedded into the same latent space so that the latent representations of source sentences can be decoded into the target language. Third, iterative back-translation is used to strengthen the source-to-target and target-to-source translation models. Lample et al. [93] and Artetxe et al. [94] first successfully built an unsupervised NMT system as described above. Specifically, Lample et al. [93] utilizes a discriminator to force the encoder to embed sentences of each language to the same latent space.

 
 
 While Lample et al. [93] used a shared encoder and a shared decoder, Artetxe et al. [94] adopt a shared encoder but two separate decoder approach. Yang et al. [95] conjectured that sharing of the encoder and decoder between two languages may lose their language characteristics. Therefore they proposed leveraging two separate encoders with some shared layers and using two different GANs to restrict the latent representations. Artetxe et al. [96] and Lample et al. [97] found that an unsupervised statistical machine translation system with iterative back-translation can easily outperform the unsupervised NMT counterpart. Lample et al. [97] summarized that initialization, language modeling, and iterative back-translation are three principles in fully unsupervised MT and they further found that combining unsupervised SMT and unsupervised NMT can reach better performances.

 
 
 Ren et al. [98] suggested that the noises and errors existed in pseudo-data can be accumulated and hinder the improvements during iterative back-translations. Therefore, they proposed to use SMT which is less sensitive to noises as posterior regularizations to unsupervised NMT. As the unsupervised NMT is usually initialized by unsupervised bilingual word embeddings (UBWE), Sun et al. [99] proposed to utilize UBWE agreement to enhance unsupervised NMT. Wu et al. [100] considered that pseudo sentences predicted by weak unsupervised MT systems are usually of low quality. To alleviate this issue, they proposed an extract-edit approach, which is an alternative to back-translation. First, they extracted the most relevant target sentences from target monolingual data given the source sentence. Then, extracted target sentences were edited to be aligned with the source sentences. This method makes it possible to use real sentence pairs to train the unsupervised NMT system. Ren et al. [101] also proposed a similar retrieve-and-rewrite method to initialize an unsupervised SMT system. Artetxe et al. [102] improved unsupervised SMT by exploiting subword information, developing a theoretically well-founded unsupervised tuning method, and incorporating a joint refinement procedure. Finally, they utilized the improved unsupervised SMT to initialize NMT model and get state-of-the-art results. As a unique method to utilize monolingual data, cross-lingual pre-trained models are used by Lample and Conneau [103] to initialize unsupervised MT systems.

 
 
 
 

### 2.6 Open Vocabulary

 
 NMT typically operates with a fixed vocabulary. Due to practical reasons such as computational concerns and memory constraints, the vocabulary size of NMT models often ranges from 30k to 50k. For word-level NMT, the limited size of vocabulary results in a large number of unknown words. Therefore, word-level NMT is unable to translate these words and performs poorly in open-vocabulary settings [ 5 , 6 ] .

 
 
 Although word-level NMT is unable to translate out-of-vocabulary words, character-level NMT do not have this problem. By splitting words into characters, the vocabulary size is much smaller and every rare word can be represented.
 Chung et al. [104] found that the NMT model with subword-level encoder and character-level decoder can also work well.
 Lee et al. [105] introduced a fully character-level NMT with convolutional network and found that character-to-character NMT is suitable in many-to-one multilingual setting.
 Luong and Manning [106] built hybrid systems that translate mostly at the word level and consult the character components for rare words.
 Passban et al. [107] proposed an extension to the model of Chung et al. [104] , which works at the character level and boosts the decoder with target-side morphological information.
 Chen et al. [108] proposed an NMT model at different levels of granularity with a multi-level attention.
 Gao et al. [109] found that self-attention performs very well on character-level translation.

 
 
 Character-level NMT also has its imperfection, splitting words into characters results in longer sequences in which each symbol contains less information, creating both modeling and computational challenges [ 110 ] . Other than word-level and character-level methods, subword-level method is another choice to model input and output sentences. Sennrich et al. [111] first adapted byte-pair-encoding (BPE) to word segmentation task, which is a simple but effective method. BPE making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This method reaches a compromise between vocabulary size and sequence length with stabilized better performance over word- and character-level methods. Moreover, it is an unsupervised method with few hyper-parameters, making it the most commonly used method for word segmentation for neural machine translation and text generation. Kudo [112] presented a simple regularization method, namely subword regularization, to improve the robustness of subword-level NMT. Provilkov et al. [113] introduced BPE-dropout to regularize the subword segmentation algorithm BPE, which is more compatible with conventional BPE than the method proposed by Kudo [112] . Wang et al. [114] investigated byte-level subwords, specifically byte-level BPE (BBPE), which is more efficient than using pure bytes only.

 
 
 

### 2.7 Prior Knowledge Integration

 
 As NMT modeling the entire translation process with a neural network, it is hard to integrate knowledge into NMT. On the one hand, existing linguistic knowledge such as dictionaries is potentially useful for NMT. On the other hand, NMT often leads to over-translation and under-translation [ 115 ] , which raises the need for adding prior knowledge to NMT.

 
 
 One line of studies focus on inducing lexical knowledge into NMT models. Zhang et al. [116] proposed a general framework that can integrate prior knowledge into NMT models through posterior regularization and found that bilingual dictionary is useful to improve NMT models.
 Morishita et al. [117] found that feeding hierarchical subword units to different modules of NMT models can also improve the translation quality.
 Liu et al. [118] proposed a novel shared-private word embedding to capture the relationship of different words for NMT models.
 Chen et al. [119] distinguished content words and functional words depending on the term frequency inverse document frequency (i.e., TF-IDF ) and then added an additional encoder and an additional loss for content words. Weller-Di Marco and Fraser [120] studied strategies to model word formation in NMT to explicitly model fusional morphology.

 
 
 Modeling the source-side syntactic structure has also drawn a lot of attention.
 Eriguchi et al. [121] extended NMT models to an end-to-end syntactic model, where the decoder is softly aligned with phrases at the source side when generating a target word.
 Sennrich and Haddow [122] explored external linguistic information such as lemmas, morphological features, POS tags and dependency labels to improve translation quality.
 Hao et al. [123] presented a multi-granularity self-attention mechanism to model phrases which are extracted by syntactic trees.
 Bugliarello and Okazaki [124] proposed the Parent-Scaled Self-Attention to incorporate dependency tree to capture the syntactic knowledge of the source sentence. There are also some works that use multi-task training to learn source-side syntactic knowledge, in which the encoder of a NMT model is trained to perform POS tagging or syntactic parsing [ 125 , 126 ] .

 
 
 Another line of studies directly model the target-side syntactic structures [ 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 ] . Aharoni and Goldberg [130] trained a end-to-end model to directly translate source sentences into constituency trees. Similar approaches are proposed to use two neural models to generate the target sentence and its corresponding tree structure [ 128 , 129 ] . Gū et al. [127] proposed to use a single model to perform translation and parsing at the same time. Yang et al. [133] introduced a latent variable model to capture the co-dependence between syntax and semantics.
 Yang et al. [134] trained a neural model to predict the soft template of the target sentence conditioning only on the source sentence and then incorporated the predicted template into the NMT model via a separate template encoder.

 
 
 

### 2.8 Interpretability and Robustness

 
 Despite the remarkable progress, it is hard to interpret the internal workings of NMT models. All internal information in NMT is represented as high-dimensional real-valued vectors or matrices. Therefore, it is challenging to associate these hidden states with language structures. The lack of interpretability has made it very difficult for researchers to understand the translation process of NMT models.

 
 
 In addition to interpretability, the lack of robustness is a severe challenge for NMT systems as well. With small perturbations in source inputs (also referred to as adversarial examples ), the translations of NMT models may lead to significant erroneous changes [ 135 , 136 ] . The lack of robustness of NMT limits its application on tasks that require robust performance on noisy inputs. Therefore, improving the robustness of NMT has gained increasing attention in the NMT community.

 
 

#### 2.8.1 Interpretability

 
 Efforts have been devoted to improving the interpretability of NMT systems in recent works. Ding et al. [137] proposed to visualize the internal workings of the RNNSearch [ 6 ] architecture. With layer-wise relevance propagation [ 138 ] , they computed and visualized the contribution of each contextual word to arbitrary hidden states in RNNSearch. Bau et al. [139] share similar motivations with Ding et al. [137] . Their basic assumption is that the same neuron in different NMT models captures similar syntactic and semantic information. They proposed to use several types of correlation coefficients to measure the importance of each neuron. As a result, by identifying important neurons and controlling their activation, the translation process of NMT systems can be controlled. Strobelt et al. [140] also put effort into visualizing the working process of RNNSearch. The highlights of their work lie in the utilization of training data. When an NMT system decodes some words, their visualization system provides the most relevant training corpora by using the nearest neighbor search. In case of translation errors, the system can locate the erroneous outputs directly in the training set by showing its origin cause. As a result, this function provides better assistants and makes it easy for developers to adjust the model and the data.

 
 
 With the tremendous success of the Transformer architecture [ 15 ] , the NMT community have shown increasing interest in understanding and interpreting Transformer. He et al. [141] generalized the idea of layer-wise relevance to word importance by attributing the NMT output to every input word through a gradient-based method. The calculated word importance illustrates the influence of each source words, which also serves as an implication of under-translation errors. Raganato and Tiedemann [142] analyzed the internal representations of Transformer encoder. Utilizing the attention weights in each layer, they extract relation among each word in the source sentence. They designed four types of probing tasks to analyze the syntactic and semantic information encoded by each layer representation and test their transferability. Voita et al. [143] also proposed to analyze the bottom-up evolution of representations in Transformer with canonical correlation analysis (CCA). By estimating mutual information, they studied how information flows in Transformer. Stahlberg et al. [144] proposed an operation sequence model to interpret NMT. Based on the translation outputted by the Transformer system, they proposed explicit modeling of the word reordering process and provided explicit word alignment between the reordered target-side sentence and the source sentence. As a result, one can track the reordering process of each word’s information as they are explicitly aligned with the source side. Recent work [ 145 ] also provided a theoretical understanding of Transformer by proving that Transformer networks are universal approximators of sequence-to-sequence functions.

 
 
 

#### 2.8.2 Robustness

 
 Belinkov and Bisk [135] first investigated the robustness of NMT. They pointed out that both synthetic and natural noise can severely harm the performance of NMT models. They experimented with four types of synthetic noise and leveraged structure-invariant representation and adversarial training to improve the robustness of NMT. Similarly, Zhao et al. [146] proposed to map the input sentence to a latent space with generative adversarial networks (GAN) and search for adversarial examples in that space. Their approach can produce semantically and syntactically coherent sentences that have negative impacts on the performance of NMT models.

 
 
 Ribeiro et al. [147] proposed semantic-preserving adversarial rules to explicitly induce adversarial examples. This approach provides a better guarantee for the adversarial examples to satisfy semantically equivalence property. Cheng et al. [148] proposed two types of approaches to generating adversarial examples by perturbing the source sentence or the internal representation of the encoder. By integrating the effect of adversarial examples into the loss function, the robustness of neural machine translation is improved by adversarial training.

 
 
 Ebrahimi et al. [149] proposed a character-level white-box attack for character-level NMT. They proposed to model the operations of character insertion, deletion, and swapping with vector computations so that the generation of adversarial examples can be formulated with differentiable string-edit operations. Liu et al. [88] proposed to jointly utilize textual and phonetic embedding in NMT to improve robustness. They found that to train a more robust model, more weights should be put on the phonetic rather than textual information. Cheng et al. [136] proposed doubly adversarial inputs to improve the robustness of NMT. Concretely, they proposed to both attack the translation model with adversarial source examples and defend the translation model with adversarial target inputs for model robustness. Zou et al. [150] utilized reinforcement learning to generate adversarial examples, producing stable attacks with semantic-preserving adversarial examples. Cheng et al. [151] proposed a novel adversarial augmentation method that minimizes the vicinal risk over virtual sentences sampled from a smoothly interpolated embedding space around the observed training sentence pairs. The adversarial data augmentation method substantially outperforms other data augmentation methods and achieves significant improvements in translation quality and robustness. For the better exploration of robust NMT, Michel and Neubig [152] proposed an MTNT dataset, source sentences of which are collected from Reddit discussion, and contain several types of noise. Target referenced translations for each source sentence, in contrast, are clear from noise. Experiments showed that current NMT models perform badly on the MTNT dataset. As a result, this dataset can serve as a testbed for NMT robustness analysis.

 
 
 
 
 

## 3 Resources

 

### 3.1 Parallel Data

 
 
 
 Workshop | 
 Domain | 
 Language Pair | 

 
 WMT20 | 
 News | 
 zh-en, cz-en, fr-de, de-en, iu-en, km-en, ja-en, ps-en, pl-en, ru-en, ta-en | 

 
 Biomedical | 
 en-eu, en-zh, en-fr, en-de, en-it, en-pt, en-ru, en-es | 

 
 Chat | 
 en-de | 

 
 IWSLT20 | 
 TED Talks | 
 en-de | 

 
 e-Commerce | 
 zh-en, en-ru | 

 
 Open Domain | 
 zh-ja | 

 
 WAT20 | 
 Scientific Paper | 
 en-ja, zh-ja | 

 
 Business Scene Dialogue | 
 en-ja | 

 
 Patent | 
 zh-ja, ko-ja, en-ja | 

 
 News | 
 ja-en, ja-ru | 

 
 IT and Wikinews | 
 hi-en, th-en, ms-en, id-en | 

 Table 3 : Domain and language pairs provided by WMT20, IWSLT20, WAT20. 
 
 
 
 
 Source | 
 Fr-En | 
 Es-En | 
 De-En | 
 Pt-En | 
 Ru-En | 
 Ar-En | 
 Zh-En | 
 Ja-En | 
 Hi-En | 

 
 OPUS [ 153 ] | 
 200.6M | 
 172.0M | 
 93.3M | 
 77.7M | 
 75.5M | 
 69.2M | 
 31.2M | 
 6.2M | 
 1.7M | 

 Table 4 : Number of sentences that available at OPUS for major languages to English. 
 
 
 Bilingual parallel corpora are the most important resources for NMT. There are several publicly available corpora, such as the datasets provided by WMT 1 1 
 1 
 
 
 
 http://www.statmt.org/wmt20/index.html , IWSLT 2 2 
 2 
 
 
 
 http://iwslt.org/doku.php , and WAT 3 3 
 3 
 
 
 
 http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2020/index.html . Table 3 lists the available domains and language pairs in these workshops.

 
 
 Besides the aforementioned machine translation workshops, we also recommend OPUS 4 4 
 4 
 
 
 
 http://opus.nlpl.eu to search resources for training NMT models, which gathers parallel data for a large number of language pairs. We list the number of sentence pairs that are available for major languages to Enligsh in Table 4 . OPUS also provides the OPUS-100 corpus for multilingual machine translation research [ 154 ] , which is an English-centric multilingual corpus covering over 100 languages.

 
 
 

### 3.2 Monolingual Data

 
 Monolingual data are also valuable resources for NMT. The Common Crawl Foundation 5 5 
 5 
 
 
 
 https://commoncrawl.org/ provides open access to high quality crawled data for over 40 languages. The CCNet toolkit 6 6 
 6 
 
 
 
 https://github.com/facebookresearch/cc_net [ 155 ] can be used to download and clean Common Crawl texts. Wikipedia provides database dump 7 7 
 7 
 
 
 
 https://dumps.wikimedia.org that can be used to extract monolingual data, which can be download using WikiExtractor 8 8 
 8 
 
 
 
 https://github.com/attardi/wikiextractor . WMT 2020 also provides several monolingual training data, which consists of data collected from NewsCrawl, NewsDicussions, Europarl, NewsCommentary, CommonCrawl, and WikiDumps.

 
 
 
 

## 4 Tools

 
 With the rapid advances of deep learning, many open-source deep learning frameworks have emerged, with TensorFlow [ 156 ] and PyTorch [ 157 ] as representative examples. At the same time, we have also witnessed the rapid development of open-source NMT toolkits, which significantly boosted the research progress of NMT. In this section, we will give a summarization of popular open-source NMT toolkits. Besides, we also introduce tools that are useful for evaluation, analysis, and data pre-processing.

 
 

### 4.1 Open-source NMT Toolkits

 
 
 
 Name | 
 Language | 
 Framework | 
 Status | 

 
 Tensor2Tensor | 
 Python | 
 TensorFlow | 
 Deprecated | 

 
 FairSeq | 
 Python | 
 PyTorch | 
 Active | 

 
 Nmt | 
 Python | 
 TensorFlow | 
 Deprecated | 

 
 OpenNMT | 
 Python/C++ | 
 PyTorch/TensorFlow | 
 Active | 

 
 Sockeye | 
 Python | 
 MXNet | 
 Active | 

 
 Nematus | 
 Python | 
 Tensorflow | 
 Active | 

 
 Marian | 
 C++ | 
 - | 
 Active | 

 
 THUMT | 
 Python | 
 PyTorch/TensorFlow | 
 Active | 

 
 NMT-Keras | 
 Python | 
 Keras | 
 Active | 

 
 Neural Monkey | 
 Python | 
 TensorFlow | 
 Active | 

 Table 5 : Popular Open-source NMT toolkits on GitHub, the ordering is determined by the number of stars as the date of December 2020. 
 
 
 We summarize some popular open-source NMT toolkits on GitHub in Table 5 .
The users can get the source codes of these toolkits directly from GitHub. We shall give a brief description of these projects.

 
 
 Tensor2Tensor. Tensor2Tensor [ 158 ] is a library of deep learning models and datasets based on TensorFlow [ 156 ] . The library was mainly developed by the Google Brain team. Tensor2Tensor provides implementation of several NMT architectures (e.g., Transformer) for the translation task. The users can run Tensor2Tensor easily on CPU, GPU, and TPU, either locally or on Cloud.

 
 
 FairSeq. FairSeq [ 159 ] is a sequence modeling toolkit developed by Facebook AI Research. The toolkit is based on Pytorch [ 157 ] and allows the users to train custom models for the translation task. FairSeq implements traditional RNN-based models and Transformer models. Besides, it also includes CNN-based translation models (e.g., LightConv and DynamicConv).

 
 
 Nmt. Nmt [ 160 ] is a toolkit developed by Google Research. The toolkit implements the GNMT architecture [ 8 ] . Besides, the NMT project also provides a nice tutorial for building a competitive NMT model from scratch. The codebase of NMT is high-quality and lightweight, which is friendly for users to add customized models.

 
 
 OpenNMT. OpenNMT is an open-source NMT toolkit developed by the collaboration of Harvard University and SYSTRAN. The toolkit currently maintains two implementations: OpenNMT-py and OpenNMT-tf . OpenNMT is proven to be research-friendly and production-ready. The OpenNMT project also provides CTranslate2 as a fast inference engine that supports both CPU and GPU.

 
 
 Sockeye. Sockeye [ 161 ] is a versatile sequence-to-sequence toolkit that is based on MXNet [ 162 ] . Sockeye is maintained by Amazon and powers machine translation services such as Amazon Translate. The toolkit features state-of-the-art machine translation models and fast CPU inference, which is useful for both research and production.

 
 
 Nematus. Nematus is an NMT toolkit developed by the NLP Group at the University of Edinburgh. The toolkit is based on TensorFlow and supports RNN-based NMT architectures as well as the Transformer architecture. In addition to the toolkits, Nematus also released high-performing NMT models covering 13 translation directions.

 
 
 Marian. Marian [ 163 ] is an efficient and self-contained NMT framework currently being developed by the Microsoft Translator team. The framework is written entirely in C++ with minimal dependencies. Marian is widely deployed by many companies and organizations. For example, Microsoft Translator currently adopts Marian as its neural machine translation engine.

 
 
 THUMT. THUMT [ 164 ] is an open-source toolkit for neural machine translation developed by the NLP Group at Tsinghua University. The toolkit includes TensorFlow, and Pytorch implementations. It supports vanilla RNN-based and Transformer models and is easy for users to build new models. Furthermore, THUMT provides visualization analysis using layer-wise relevance propagation [ 137 ] .

 
 
 NMT-Keras. NMT-Keras [ 165 ] is a flexible toolkit for neural machine translation developed by the Pattern Recognition and Human Language Technology Research Center at Polytechnic University of Valencia. The toolkit is based on Keras which uses Theano or TensorFlow as the backend. NMT-Keras emphasizes the development of advanced applications for NMT systems, such as interactive NMT and online learning. It also has been extended to other tasks including image and video captioning, sentence classification, and visual question answering.

 
 
 Neural Monkey Neural Monkey is an open-source neural machine translation and general sequence-to-sequence learning system. The toolkit is built on the TensorFlow library and provides a high-level API tailored for fast prototyping of complex architectures.

 
 
 

### 4.2 Tools for Evaluation and Analysis

 
 Manual evaluation of MT outputs is not only expensive but also impractical to scaling for large language pairs. On the contrary, automatic MT evaluation is inexpensive and language-independent, with BLEU [ 166 ] as the representative automatic evaluation metric. Besides evaluation, there is also a need for analyzing MT outputs. We recommend the following tools for evaluating and analyzing MT output.

 
 
 Sacre BLEU. SacreBLEU 9 9 
 9 
 
 
 
 https://github.com/mjpost/sacrebleu [ 167 ] is a toolkit to compute shareable, comparable, and reproducible BLEU scores. SacreBLEU computes BLEU scores on detokenized outputs, using WMT standard tokenization. As a result, the scores are not affected by different processing tools. Besides, it can produce a short version string that facilitates cross-paper comparisons.

 
 
 Compare-mt . Compare-mt 10 10 
 10 
 
 
 
 https://github.com/neulab/compare-mt [ 168 ] is a program to compare the outputs of multiple systems for language generation. In order to provide high-level analysis of outputs, it enables analysis of accuracy of generation of particular types of words, bucketed histograms of sentence accuracies or counts based on salient characteristics, and so on.

 
 
 MT-ComparEval . MT-ComparEval 11 11 
 11 
 
 
 
 https://github.com/ondrejklejch/MT-ComparEval is also a tool for comparison and evaluation of machine translations. It allows users to compare translations according to automatic metrics or quality comparison from the aspects of n-grams.

 
 
 

### 4.3 Other Tools

 
 Asides from the above mentioned tools, we found the following toolkits are very useful for NMT research and deployment.

 
 
 Moses . Moses 12 12 
 12 
 
 
 
 https://github.com/moses-smt/mosesdecoder [ 169 ] is a self-contained statistical machine translation toolkit. Besides SMT-related components, Moses provides a large number of tools to clean and pre-process texts, which are also useful for training NMT models. Moses also contains several easy-to-use scripts to analyze and evaluate MT outputs.

 
 
 Subword-NMT . 
 Subword-NMT 13 13 
 13 
 
 
 
 https://github.com/rsennrich/subword-nmt is an open-source toolkit for unsupervised word segmentation for neural machine translation and text generation. It adopts the Byte-Pair Encoding (BPE) algorithm proposed by [ 111 ] and BPE dropout proposed by [ 113 ] . It is the most commonly used toolkit to alleviate the out-of-vocabulary problem in NMT.

 
 
 SentencePiece . SentencePiece 14 14 
 14 
 
 
 
 https://github.com/google/sentencepiece is a powerful unsupervised text segmentation toolkit. SentencePiece is written in C++ and provides APIs for other languages such as Python. SentencePiece implements the BPE algorithm [ 111 ] and unigram language model [ 112 ] . Unlike Subword-NMT , SentencePiece can learn to segment raw texts without additional pre-processing. As a result, SentencePiece is a suitable choice to segment multilingual texts.

 
 
 
 

## 5 Conclusion

 
 
Neural machine translation has become the dominant approach to machine translation in both research and practice. This article reviewed the widely used methods in NMT, including modeling, decoding, data augmentation, interpretation, as well as evaluation. We then summarize the resources and tools that are useful for NMT research.

 
 
 Despite the great success achieved by NMT, there are still many problems to be explored. We list some important and challenging problems for NMT as follows:

 
 1. 
 
 Understanding NMT. Although there are many attempts to analyze and interpret NMT, our understandings about NMT are still limited. Understanding how and why NMT produces its translation result is important to figure out the bottleneck and weakness of NMT models.

 

 2. 
 
 Designing better architectures . Designing a new architecture that better than Transformer is beneficial for both NMT research and production. Furthermore, designing a new architecture that balances translation performance and computational complexity is also important.

 

 3. 
 
 Making full use of monolingual data. Monolingual data are valuable resources. Despite the remarkable progress, we believe that there is still much room for NMT to make use of abundant monolingual data.

 

 4. 
 
 Integrating prior knowledge. Incorporating human knowledge into NMT is also an important problem. Although there is some progress, the results are far from satisfactory. How to convert discrete and continuous representations into each other is a problem of NMT that needs further exploration.

 

 
 
 
 

## Acknowledgements

 
 This work was supported by the National Key R D Program of China (No. 2017YFB0 202204), National Natural Science Foundation of China (No. 61925601, No. 61761166 008, No. 61772302), Beijing Academy of Artificial Intelligence, Huawei Noah’s Ark Lab, and the NExT++ project supported by the National Research Foundation, Prime Ministers Office, Singapore under its IRC@Singapore Funding Initiative. We thank Kehai Chen and Xiangwen Zhang for their kindly suggestions.

 
 
 

## References

 
 
 [1] 
 
P. F. Brown, J. Cocke,
S. A. Della Pietra, V. J. Della Pietra,
F. Jelinek, J. Lafferty,
R. L. Mercer, P. S. Roossin,

 
 A statistical approach to machine translation,

 
 Computational linguistics 16
(1990) 79–85.

 

 
 [2] 
 
P. Koehn, F. J. Och,
D. Marcu, Statistical phrase-based
translation, Technical Report, UNIVERSITY OF SOUTHERN
CALIFORNIA MARINA DEL REY INFORMATION SCIENCES INST, 2003.

 

 
 [3] 
 
N. Kalchbrenner, P. Blunsom,

 
 Recurrent continuous translation models,

 
 in: Proceedings of EMNLP, 2013,
pp. 1700–1709.

 

 
 [4] 
 
K. Cho, B. van Merrienboer,
C. Gulcehre, D. Bahdanau,
F. Bougares, H. Schwenk,
Y. Bengio,

 
 Learning phrase representations using rnn
encoder–decoder for statistical machine translation,

 
 in: Proceedings of EMNLP, 2014,
pp. 1724–1734.

 

 
 [5] 
 
I. Sutskever, O. Vinyals,
Q. V. Le,

 
 Sequence to sequence learning with neural networks,

 
 in: Proceedings of NeurIPS,
2014, pp. 3104–3112.

 

 
 [6] 
 
D. Bahdanau, K. Cho,
Y. Bengio,

 
 Neural machine translation by jointly learning to
align and translate,

 
 in: Proceedings of ICLR, 2015.

 

 
 [7] 
 
M. Junczys-Dowmunt, T. Dwojak,
H. Hoang,

 
 Is neural machine translation ready for deployment? a
case study on 30 translation directions,

 
 arXiv preprint arXiv:1610.01108
(2016).

 

 
 [8] 
 
Y. Wu, M. Schuster,
Z. Chen, Q. V. Le, et al.,

 
 Google’s neural machine translation system: Bridging
the gap between human and machine translation,

 
 arXiv preprint arXiv:1609.08144
(2016).

 

 
 [9] 
 
H. Hassan, A. Aue,
C. Chen, V. Chowdhary,
J. Clark, C. Federmann,
X. Huang, M. Junczys-Dowmunt,
W. Lewis, M. Li, et al.,

 
 Achieving human parity on automatic chinese to
english news translation,

 
 arXiv preprint arXiv:1803.05567
(2018).

 

 
 [10] 
 
D. Kingma, J. Ba,

 
 Adam: A method for stochastic optimization,

 
 arXiv preprint arXiv:1412.6980
(2014).

 

 
 [11] 
 
K. Cho, B. Van Merriënboer,
D. Bahdanau, Y. Bengio,

 
 On the properties of neural machine translation:
Encoder-decoder approaches,

 
 arXiv preprint arXiv:1409.1259
(2014).

 

 
 [12] 
 
Y. Bengio, P. Simard,
P. Frasconi,

 
 Learning long-term dependencies with gradient descent
is difficult,

 
 IEEE Trans. Neural Netw. (1994).

 

 
 [13] 
 
N. Kalchbrenner, L. Espeholt,
K. Simonyan, A. v. d. Oord,
A. Graves, K. Kavukcuoglu,

 
 Neural machine translation in linear time,

 
 arXiv preprint arXiv:1610.10099
(2016).

 

 
 [14] 
 
J. Gehring, M. Auli,
D. Grangier, D. Yarats,
Y. N. Dauphin,

 
 Convolutional sequence to sequence learning,

 
 arXiv preprint arXiv:1705.03122
(2017).

 

 
 [15] 
 
A. Vaswani, N. Shazeer,
N. Parmar, J. Uszkoreit,
L. Jones, A. N. Gomez,
Ł. Kaiser, I. Polosukhin,

 
 Attention is all you need,

 
 in: Proceedings of NeurIPS,
2017, pp. 5998–6008.

 

 
 [16] 
 
A. Graves, G. Wayne,
I. Danihelka,

 
 Neural turing machines,

 
 arXiv preprint arXiv:1410.5401
(2014).

 

 
 [17] 
 
M.-T. Luong, H. Pham,
C. D. Manning,

 
 Effective approaches to attention-based neural
machine translation,

 
 arXiv preprint arXiv:1508.04025
(2015).

 

 
 [18] 
 
H. T. Siegelmann, E. D. Sontag,

 
 On the computational power of neural nets,

 
 Journal of computer and system sciences
(1995).

 

 
 [19] 
 
S. Hochreiter, J. Schmidhuber,

 
 Long short-term memory,

 
 Neural computation (1997).

 

 
 [20] 
 
J. L. Ba, J. R. Kiros,
G. E. Hinton,

 
 Layer normalization,

 
 arXiv preprint arXiv:1607.06450
(2016).

 

 
 [21] 
 
K. He, X. Zhang, S. Ren,
J. Sun,

 
 Deep residual learning for image recognition,

 
 in: Proceedings of CVPR, 2016,
pp. 770–778.

 

 
 [22] 
 
J. Zhou, W. Xu,

 
 End-to-end learning of semantic role labeling using
recurrent neural networks,

 
 in: Proceedings of ACL, 2015,
pp. 1127–1137.

 

 
 [23] 
 
L. Kaiser, A. N. Gomez,
F. Chollet,

 
 Depthwise separable convolutions for neural machine
translation,

 
 arXiv preprint arXiv:1706.03059
(2017).

 

 
 [24] 
 
F. Wu, A. Fan,
A. Baevski, Y. N. Dauphin,
M. Auli,

 
 Pay less attention with lightweight and dynamic
convolutions,

 
 arXiv preprint arXiv:1901.10430
(2019).

 

 
 [25] 
 
L. Liu, M. Utiyama,
A. Finch, E. Sumita,

 
 Agreement on target-bidirectional neural machine
translation,

 
 in: Proceedings of NAACL-HLT,
2016, pp. 411–416.

 

 
 [26] 
 
C. D. V. Hoang, G. Haffari,
T. Cohn,

 
 Towards decoding as continuous optimisation in neural
machine translation,

 
 in: Proceedings of EMNLP, 2017,
pp. 146–156.

 

 
 [27] 
 
X. Zhang, J. Su, Y. Qin,
Y. Liu, R. Ji, H. Wang,

 
 Asynchronous bidirectional decoding for neural
machine translation,

 
 in: Proceedings of AAAI, 2018,
pp. 5698–5705.

 

 
 [28] 
 
L. Zhou, J. Zhang,
C. Zong,

 
 Synchronous bidirectional neural machine
translation,

 
 TACL (2019).

 

 
 [29] 
 
J. Gu, J. Bradbury,
C. Xiong, V. O. Li,
R. Socher,

 
 Non-autoregressive neural machine translation,

 
 in: Proceedings of ICLR, 2018.

 

 
 [30] 
 
Z. Zheng, H. Zhou,
S. Huang, L. Mou,
X. Dai, J. Chen, Z. Tu,

 
 Modeling past and future for neural machine
translation,

 
 Transactions of the Association for Computational
Linguistics 6 (2018)
145–157.

 

 
 [31] 
 
Z. Zheng, S. Huang,
Z. Tu, X.-Y. Dai,
C. Jiajun,

 
 Dynamic past and future for neural machine
translation,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 930–940.

 

 
 [32] 
 
B. Zhang, D. Xiong,
J. Su, J. Luo,

 
 Future-aware knowledge distillation for neural
machine translation,

 
 IEEE/ACM Transactions on Audio, Speech, and
Language Processing 27 (2019)
2278–2287.

 

 
 [33] 
 
J. Zhang, C. Zong,

 
 Neural machine translation: Challenges, progress and
future,

 
 arXiv preprint arXiv:2004.05809
(2020).

 

 
 [34] 
 
Z. Yang, L. Chen,
M. Le Nguyen,

 
 Regularizing forward and backward decoding to improve
neural machine translation,

 
 in: Proceedings of International Conference on
Knowledge and Systems Engineering (KSE), 2018, pp.
73–78.

 

 
 [35] 
 
Z. Zhang, S. Wu, S. Liu,
M. Li, M. Zhou, T. Xu,

 
 Regularizing neural machine translation by
target-bidirectional agreement,

 
 in: Proceedings of AAAI,
volume 33, 2019, pp.
443–450.

 

 
 [36] 
 
R. Sennrich, B. Haddow,
A. Birch,

 
 Edinburgh neural machine translation systems for wmt
16,

 
 in: Proceedings of WMT, 2016,
pp. 371–376.

 

 
 [37] 
 
J. Su, X. Zhang, Q. Lin,
Y. Qin, J. Yao, Y. Liu,

 
 Exploiting reverse target-side contexts for neural
machine translation via asynchronous bidirectional decoding,

 
 Artificial Intelligence 277
(2019) 103168.

 

 
 [38] 
 
L. Zhou, J. Zhang,
C. Zong, H. Yu,

 
 Sequence generation: from both sides to the middle,

 
 in: Proceedings of IJCAI, 2019,
pp. 5471–5477.

 

 
 [39] 
 
J. Zhang, L. Zhou,
Y. Zhao, C. Zong,

 
 Synchronous bidirectional inference for neural
sequence generation,

 
 Artificial Intelligence 281
(2020) 103234.

 

 
 [40] 
 
R. Sennrich, A. Birch,
A. Currey, U. Germann,
B. Haddow, K. Heafield,
A. V. M. Barone, P. Williams,

 
 The university of edinburgh’s neural mt systems for
wmt17,

 
 in: Proceedings of WMT, 2017,
pp. 389–399.

 

 
 [41] 
 
S. Mehri, L. Sigal,

 
 Middle-out decoding,

 
 in: Advances in NeurIPS, 2018,
pp. 5518–5529.

 

 
 [42] 
 
Y. Kim, A. M. Rush,

 
 Sequence-level knowledge distillation,

 
 in: Proceedings of EMNLP, 2016,
pp. 1317–1327.

 

 
 [43] 
 
C. Zhou, J. Gu,
G. Neubig,

 
 Understanding knowledge distillation in
non-autoregressive machine translation,

 
 in: Proceedings of ICLR, 2019.

 

 
 [44] 
 
J. Lee, E. Mansimov,
K. Cho,

 
 Deterministic non-autoregressive neural sequence
modeling by iterative refinement,

 
 in: Proceedings of EMNLP, 2018,
pp. 1173–1182.

 

 
 [45] 
 
J. Libovickỳ, J. Helcl,

 
 End-to-end non-autoregressive neural machine
translation with connectionist temporal classification,

 
 in: Proceedings of EMNLP, 2018,
pp. 3016–3021.

 

 
 [46] 
 
M. Ghazvininejad, O. Levy,
Y. Liu, L. Zettlemoyer,

 
 Mask-predict: Parallel decoding of conditional masked
language models,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 6114–6123.

 

 
 [47] 
 
C. Wang, J. Zhang,
H. Chen,

 
 Semi-autoregressive neural machine translation,

 
 in: Proceedings of EMNLP, 2018,
pp. 479–488.

 

 
 [48] 
 
J. Guo, X. Tan, D. He,
T. Qin, L. Xu, T.-Y.
Liu,

 
 Non-autoregressive neural machine translation with
enhanced decoder input,

 
 in: Proceedings of AAAI,
volume 33, 2019, pp.
3723–3730.

 

 
 [49] 
 
C. Shao, Y. Feng,
J. Zhang, F. Meng,
X. Chen, J. Zhou,

 
 Retrieving sequential information for
non-autoregressive neural machine translation,

 
 in: Proceedings of ACL, 2019,
pp. 3013–3024.

 

 
 [50] 
 
Y. Wang, F. Tian, D. He,
T. Qin, C. Zhai, T.-Y.
Liu,

 
 Non-autoregressive machine translation with auxiliary
regularization,

 
 in: Proceedings of AAAI,
volume 33, 2019, pp.
5377–5384.

 

 
 [51] 
 
M. Stern, W. Chan,
J. Kiros, J. Uszkoreit,

 
 Insertion transformer: Flexible sequence generation
via insertion operations,

 
 in: Proceedings of ICML, 2019,
pp. 5976–5985.

 

 
 [52] 
 
B. Wei, M. Wang, H. Zhou,
J. Lin, X. Sun,

 
 Imitation learning for non-autoregressive neural
machine translation,

 
 in: Proceedings of ACL, 2019,
pp. 1304–1312.

 

 
 [53] 
 
N. Akoury, K. Krishna,
M. Iyyer,

 
 Syntactically supervised transformers for faster
neural machine translation,

 
 in: Proceedings of ACL, 2019,
pp. 1269–1281.

 

 
 [54] 
 
J. Gu, Q. Liu, K. Cho,

 
 Insertion-based decoding with automatically inferred
generation order,

 
 TACL 7 (2019)
661–676.

 

 
 [55] 
 
J. Devlin, M.-W. Chang,
K. Lee, K. Toutanova,

 
 Bert: Pre-training of deep bidirectional transformers
for language understanding,

 
 in: Proceedings of NAACL-HLT,
2019, pp. 4171–4186.

 

 
 [56] 
 
M. Ranzato, S. Chopra,
M. Auli, W. Zaremba,

 
 Sequence level training with recurrent neural
networks,

 
 arXiv preprint arXiv:1511.06732
(2015).

 

 
 [57] 
 
M. Ranzato, S. Chopra,
M. Auli, W. Zaremba,

 
 Sequence level training with recurrent neural
networks,

 
 in: Proceedings of ICLR, 2016.

 

 
 [58] 
 
L. Wu, F. Tian, T. Qin,
J. Lai, T.-Y. Liu,

 
 A study of reinforcement learning for neural machine
translation,

 
 in: Proceedings of EMNLP, 2018,
pp. 3612–3621.

 

 
 [59] 
 
S. Shen, Y. Cheng, Z. He,
W. He, H. Wu, M. Sun,
Y. Liu,

 
 Minimum risk training for neural machine
translation,

 
 in: Proceedings of ACL, 2016,
pp. 1683–1692.

 

 
 [60] 
 
L. Choshen, L. Fox,
Z. Aizenbud, O. Abend,

 
 On the weaknesses of reinforcement learning for
neural machine translation,

 
 in: Proceedings of ICLR, 2020.

 

 
 [61] 
 
S. Edunov, M. Ott,
M. Auli, D. Grangier,
M. Ranzato,

 
 Classical structured prediction losses for sequence
to sequence learning,

 
 in: Proceedings of NAACL-HLT,
2018, pp. 355–364.

 

 
 [62] 
 
Z. Yang, Y. Cheng,
Y. Liu, M. Sun,

 
 Reducing word omission errors in neural machine
translation: A contrastive learning approach,

 
 in: Proceedings of ACL, 2019,
pp. 6191–6196.

 

 
 [63] 
 
J. Wieting, T. Berg-Kirkpatrick,
K. Gimpel, G. Neubig,

 
 Beyond BLEU:training neural machine translation
with semantic similarity,

 
 in: Proceedings of ACL, 2019,
pp. 4344–4355.

 

 
 [64] 
 
L. Chen, Y. Zhang,
R. Zhang, C. Tao,
Z. Gan, H. Zhang,
B. Li, D. Shen,
C. Chen, L. Carin,

 
 Improving sequence-to-sequence learning via optimal
transport,

 
 in: Proceedings of ICLR, 2019.

 

 
 [65] 
 
S. Kumar, Y. Tsvetkov,

 
 Von mises-fisher loss for training sequence to
sequence models with continuous outputs,

 
 in: Proceedings of ICLR, 2019.

 

 
 [66] 
 
B. Zoph, D. Yuret,
J. May, K. Knight,

 
 Transfer learning for low-resource neural machine
translation,

 
 arXiv preprint arXiv:1604.02201
(2016).

 

 
 [67] 
 
C. Gulcehre, O. Firat,
K. Xu, K. Cho,
Y. Bengio,

 
 On integrating a language model into neural machine
translation,

 
 Computer Speech Language 45
(2017) 137–148.

 

 
 [68] 
 
R. Sennrich, B. Haddow,
A. Birch,

 
 Improving neural machine translation models with
monolingual data,

 
 in: Proceedings of ACL, 2016,
pp. 86–96.

 

 
 [69] 
 
A. Poncelas, D. Shterionov,
A. Way, G. de Buy Wenniger,
P. Passban,

 
 Investigating backtranslation in neural machine
translation,

 
 arXiv preprint arXiv:1804.06189
(2018).

 

 
 [70] 
 
A. Karakanta, J. Dehdari,
J. van Genabith,

 
 Neural machine translation for low-resource languages
without parallel corpora,

 
 Machine Translation 32
(2018) 167–189.

 

 
 [71] 
 
K. Imamura, A. Fujita,
E. Sumita,

 
 Enhancement of encoder and attention using target
monolingual corpora in neural machine translation,

 
 in: Proceedings of the 2nd Workshop on Neural
Machine Translation and Generation, 2018, pp.
55–63.

 

 
 [72] 
 
S. Edunov, M. Ott,
M. Auli, D. Grangier,

 
 Understanding back-translation at scale,

 
 arXiv preprint arXiv:1808.09381
(2018).

 

 
 [73] 
 
I. Caswell, C. Chelba,
D. Grangier,

 
 Tagged back-translation,

 
 WMT 2019 (2019)
53.

 

 
 [74] 
 
S. Wang, Y. Liu, C. Wang,
H. Luan, M. Sun,

 
 Improving back-translation with uncertainty-based
confidence estimation,

 
 arXiv preprint arXiv:1909.00157
(2019).

 

 
 [75] 
 
J. Zhang, C. Zong,

 
 Exploiting source-side monolingual data in neural
machine translation,

 
 in: Proceedings of EMNLP, 2016,
pp. 1535–1545.

 

 
 [76] 
 
V. C. D. Hoang, P. Koehn,
G. Haffari, T. Cohn,

 
 Iterative back-translation for neural machine
translation,

 
 in: Proceedings of the 2nd Workshop on Neural
Machine Translation and Generation, 2018, pp.
18–24.

 

 
 [77] 
 
Y. Cheng, W. Xu, Z. He,
W. He, H. Wu, M. Sun,
Y. Liu,

 
 Semi-supervised learning for neural machine
translation,

 
 in: Proceedings of ACL, 2016,
pp. 1965–1974.

 

 
 [78] 
 
D. He, Y. Xia, T. Qin,
L. Wang, N. Yu, T.-Y.
Liu, W.-Y. Ma,

 
 Dual learning for machine translation,

 
 in: Advances in NeurIPS, 2016,
pp. 820–828.

 

 
 [79] 
 
Z. Zheng, H. Zhou,
S. Huang, L. Li, X.-Y.
Dai, J. Chen,

 
 Mirror-generative neural machine translation,

 
 in: Proceedings of ICLR, 2020.

 

 
 [80] 
 
K. Song, X. Tan, T. Qin,
J. Lu, T.-Y. Liu,

 
 Mass: Masked sequence to sequence pre-training for
language generation,

 
 arXiv preprint arXiv:1905.02450
(2019).

 

 
 [81] 
 
Y. Liu, J. Gu, N. Goyal,
X. Li, S. Edunov,
M. Ghazvininejad, M. Lewis,
L. Zettlemoyer,

 
 Multilingual denoising pre-training for neural
machine translation,

 
 arXiv preprint arXiv:2001.08210
(2020).

 

 
 [82] 
 
M. E. Peters, M. Neumann,
M. Iyyer, M. Gardner,
C. Clark, K. Lee,
L. Zettlemoyer,

 
 Deep contextualized word representations,

 
 arXiv preprint arXiv:1802.05365
(2018).

 

 
 [83] 
 
A. Radford, J. Wu,
R. Child, D. Luan,
D. Amodei, I. Sutskever,

 
 Language models are unsupervised multitask learners,

 
 OpenAI Blog 1
(2019) 9.

 

 
 [84] 
 
K. Clark, U. Khandelwal,
O. Levy, C. D. Manning,

 
 What does bert look at? an analysis of bert’s
attention,

 
 in: Proceedings of the 2019 ACL Workshop
BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP,
2019, pp. 276–286.

 

 
 [85] 
 
M. Lewis, Y. Liu,
N. Goyal, M. Ghazvininejad,
A. Mohamed, O. Levy,
V. Stoyanov, L. Zettlemoyer,

 
 Bart: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehension,

 
 arXiv preprint arXiv:1910.13461
(2019).

 

 
 [86] 
 
S. Edunov, A. Baevski,
M. Auli,

 
 Pre-trained language model representations for
language generation,

 
 arXiv preprint arXiv:1903.09722
(2019).

 

 
 [87] 
 
J. Zhu, Y. Xia, L. Wu,
D. He, T. Qin, W. Zhou,
H. Li, T.-Y. Liu,

 
 Incorporating bert into neural machine translation,

 
 arXiv preprint arXiv:2002.06823
(2020).

 

 
 [88] 
 
H. Liu, M. Ma, L. Huang,
H. Xiong, Z. He,

 
 Robust neural machine translation with joint textual
and phonetic embedding,

 
 in: Proceedings of ACL, 2019,
pp. 3044–3049.

 

 
 [89] 
 
M. Zhang, Y. Liu,
H. Luan, M. Sun,

 
 Adversarial training for unsupervised bilingual
lexicon induction,

 
 in: Proceedings of ACL, 2017,
pp. 1959–1970.

 

 
 [90] 
 
M. Artetxe, G. Labaka,
E. Agirre,

 
 Learning bilingual word embeddings with (almost) no
bilingual data,

 
 in: Proceedings of ACL, 2017,
pp. 451–462.

 

 
 [91] 
 
A. Conneau, G. Lample,
M. Ranzato, L. Denoyer,
H. Jégou,

 
 Word translation without parallel data,

 
 arXiv preprint arXiv:1710.04087
(2017).

 

 
 [92] 
 
P. Vincent, H. Larochelle,
Y. Bengio, P.-A. Manzagol,

 
 Extracting and composing robust features with
denoising autoencoders,

 
 in: Proceedings of ICML, 2008,
pp. 1096–1103.

 

 
 [93] 
 
G. Lample, A. Conneau,
L. Denoyer, M. Ranzato,

 
 Unsupervised machine translation using monolingual
corpora only,

 
 arXiv preprint arXiv:1711.00043
(2017).

 

 
 [94] 
 
M. Artetxe, G. Labaka,
E. Agirre, K. Cho,

 
 Unsupervised neural machine translation,

 
 arXiv preprint arXiv:1710.11041
(2017).

 

 
 [95] 
 
Z. Yang, W. Chen,
F. Wang, B. Xu,

 
 Unsupervised neural machine translation with weight
sharing,

 
 arXiv preprint arXiv:1804.09057
(2018).

 

 
 [96] 
 
M. Artetxe, G. Labaka,
E. Agirre,

 
 Unsupervised statistical machine translation,

 
 arXiv preprint arXiv:1809.01272
(2018).

 

 
 [97] 
 
G. Lample, M. Ott,
A. Conneau, L. Denoyer,
M. Ranzato,

 
 Phrase-based neural unsupervised machine
translation,

 
 arXiv preprint arXiv:1804.07755
(2018).

 

 
 [98] 
 
S. Ren, Z. Zhang, S. Liu,
M. Zhou, S. Ma,

 
 Unsupervised neural machine translation with smt as
posterior regularization,

 
 in: Proceedings of the AAAI,
volume 33, 2019, pp.
241–248.

 

 
 [99] 
 
H. Sun, R. Wang, K. Chen,
M. Utiyama, E. Sumita,
T. Zhao,

 
 Unsupervised bilingual word embedding agreement for
unsupervised neural machine translation,

 
 in: Proceedings of ACL, 2019,
pp. 1235–1245.

 

 
 [100] 
 
J. Wu, X. Wang, W. Y.
Wang,

 
 Extract and edit: An alternative to back-translation
for unsupervised neural machine translation,

 
 arXiv preprint arXiv:1904.02331
(2019).

 

 
 [101] 
 
S. Ren, Y. Wu, S. Liu,
M. Zhou, S. Ma,

 
 A retrieve-and-rewrite initialization method for
unsupervised machine translation,

 
 in: Proceedings of ACL, 2020,
pp. 3498–3504.

 

 
 [102] 
 
M. Artetxe, G. Labaka,
E. Agirre,

 
 An effective approach to unsupervised machine
translation,

 
 arXiv preprint arXiv:1902.01313
(2019).

 

 
 [103] 
 
G. Lample, A. Conneau,

 
 Cross-lingual language model pretraining,

 
 arXiv preprint arXiv:1901.07291
(2019).

 

 
 [104] 
 
J. Chung, K. Cho,
Y. Bengio,

 
 A character-level decoder without explicit
segmentation for neural machine translation,

 
 arXiv preprint arXiv:1603.06147
(2016).

 

 
 [105] 
 
J. Lee, K. Cho,
T. Hofmann,

 
 Fully character-level neural machine translation
without explicit segmentation,

 
 Transactions of the Association for Computational
Linguistics 5 (2017)
365–378.

 

 
 [106] 
 
M.-T. Luong, C. D. Manning,

 
 Achieving open vocabulary neural machine translation
with hybrid word-character models,

 
 arXiv preprint arXiv:1604.00788
(2016).

 

 
 [107] 
 
P. Passban, Q. Liu,
A. Way,

 
 Improving character-based decoding using target-side
morphological information for neural machine translation,

 
 arXiv preprint arXiv:1804.06506
(2018).

 

 
 [108] 
 
H. Chen, S. Huang,
D. Chiang, X. Dai,
J. Chen,

 
 Combining character and word information in neural
machine translation using a multi-level attention,

 
 in: Proceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long Papers), 2018,
pp. 1284–1293.

 

 
 [109] 
 
Y. Gao, N. I. Nikolov,
Y. Hu, R. H. Hahnloser,

 
 Character-level translation with self-attention,

 
 arXiv preprint arXiv:2004.14788
(2020).

 

 
 [110] 
 
C. Cherry, G. Foster,
A. Bapna, O. Firat,
W. Macherey,

 
 Revisiting character-based neural machine translation
with capacity and compression,

 
 arXiv preprint arXiv:1808.09943
(2018).

 

 
 [111] 
 
R. Sennrich, B. Haddow,
A. Birch,

 
 Neural machine translation of rare words with subword
units,

 
 in: Proceedings of ACL, 2016.

 

 
 [112] 
 
T. Kudo,

 
 Subword regularization: Improving neural network
translation models with multiple subword candidates,

 
 arXiv preprint arXiv:1804.10959
(2018).

 

 
 [113] 
 
I. Provilkov, D. Emelianenko,
E. Voita,

 
 Bpe-dropout: Simple and effective subword
regularization,

 
 arXiv preprint arXiv:1910.13267
(2019).

 

 
 [114] 
 
C. Wang, K. Cho, J. Gu,

 
 Neural machine translation with byte-level subwords,

 
 in: Proceedings of the AAAI Conference on
Artificial Intelligence, volume 34, 2020,
pp. 9154–9160.

 

 
 [115] 
 
Z. Tu, Z. Lu, Y. Liu,
X. Liu, H. Li,

 
 Modeling coverage for neural machine translation,

 
 in: Proceedings of ACL, 2016.

 

 
 [116] 
 
J. Zhang, Y. Liu,
H. Luan, J. Xu, M. Sun,

 
 Prior knowledge integration for neural machine
translation using posterior regularization,

 
 in: Proceedings of ACL, 2017,
pp. 1514–1523.

 

 
 [117] 
 
M. Morishita, J. Suzuki,
M. Nagata,

 
 Improving neural machine translation by incorporating
hierarchical subword features,

 
 in: Proceedings of COLING, 2018,
pp. 618–629.

 

 
 [118] 
 
X. Liu, D. F. Wong,
Y. Liu, L. S. Chao,
T. Xiao, J. Zhu,

 
 Shared-private bilingual word embeddings for neural
machine translation,

 
 in: Proceedings of ACL, 2019,
pp. 3613–3622.

 

 
 [119] 
 
K. Chen, R. Wang,
M. Utiyama, E. Sumita,

 
 Content word aware neural machine translation,

 
 in: Proceedings of ACL, 2020,
pp. 358–364.

 

 
 [120] 
 
M. Weller-Di Marco, A. Fraser,

 
 Modeling word formation in english–german neural
machine translation,

 
 in: Proceedings of ACL, 2020,
pp. 4227–4232.

 

 
 [121] 
 
A. Eriguchi, K. Hashimoto,
Y. Tsuruoka,

 
 Tree-to-sequence attentional neural machine
translation,

 
 in: Proceedings of ACL, 2016,
pp. 823–833.

 

 
 [122] 
 
R. Sennrich, B. Haddow,

 
 Linguistic input features improve neural machine
translation,

 
 in: Proceedings of WMT, 2016,
pp. 83–91.

 

 
 [123] 
 
J. Hao, X. Wang, S. Shi,
J. Zhang, Z. Tu,

 
 Multi-granularity self-attention for neural machine
translation,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 886–896.

 

 
 [124] 
 
E. Bugliarello, N. Okazaki,

 
 Enhancing machine translation with dependency-aware
self-attention,

 
 in: Proceedings of ACL, 2020,
pp. 1618–1627.

 

 
 [125] 
 
A. Eriguchi, Y. Tsuruoka,
K. Cho,

 
 Learning to parse and translate improves neural
machine translation,

 
 in: Proceedings of ACL, 2017,
pp. 72–78.

 

 
 [126] 
 
L. H. Baniata, S. Park,
S.-B. Park,

 
 A multitask-based neural machine translation model
with part-of-speech tags integration for arabic dialects,

 
 Applied Sciences 8
(2018) 2502.

 

 
 [127] 
 
J. Gū, H. S. Shavarani,
A. Sarkar,

 
 Top-down tree structured decoding with syntactic
connections for neural machine translation and parsing,

 
 in: Proceedings of EMNLP, 2018,
pp. 401–413.

 

 
 [128] 
 
X. Wang, H. Pham, P. Yin,
G. Neubig,

 
 A tree-based decoder for neural machine translation,

 
 in: Proceedings of EMNLP, 2018,
pp. 4772–4777.

 

 
 [129] 
 
S. Wu, D. Zhang, N. Yang,
M. Li, M. Zhou,

 
 Sequence-to-dependency neural machine translation,

 
 in: Proceedings of ACL, 2017,
pp. 698–707.

 

 
 [130] 
 
R. Aharoni, Y. Goldberg,

 
 Towards string-to-tree neural machine translation,

 
 in: Proceedings of ACL, 2017,
pp. 132–140.

 

 
 [131] 
 
J. Bastings, I. Titov,
W. Aziz, D. Marcheggiani,
K. Sima’an,

 
 Graph convolutional encoders for syntax-aware neural
machine translation,

 
 in: Proceedings of EMNLP, 2017,
pp. 1957–1967.

 

 
 [132] 
 
X. Li, L. Liu, Z. Tu,
S. Shi, M. Meng,

 
 Target foresight based attention for neural machine
translation,

 
 in: Proceedings of NAACL-HLT,
2018, pp. 1380–1390.

 

 
 [133] 
 
X. Yang, Y. Liu, D. Xie,
X. Wang, N. Balasubramanian,

 
 Latent part-of-speech sequences for neural machine
translation,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 780–790.

 

 
 [134] 
 
J. Yang, S. Ma, D. Zhang,
Z. Li, M. Zhou,

 
 Improving neural machine translation with soft
template prediction,

 
 in: Proceedings of WMT, 2020,
pp. 5979–5989.

 

 
 [135] 
 
Y. Belinkov, Y. Bisk,

 
 Synthetic and natural noise both break neural machine
translation,

 
 in: Proceedings of ICLR, 2018.

 

 
 [136] 
 
Y. Cheng, L. Jiang,
W. Macherey,

 
 Robust neural machine translation with doubly
adversarial inputs,

 
 in: Proceedings of ACL, 2019,
pp. 4324–4333.

 

 
 [137] 
 
Y. Ding, Y. Liu, H. Luan,
M. Sun,

 
 Visualizing and understanding neural machine
translation,

 
 in: Proceedings of ACL, 2017,
pp. 1150–1159.

 

 
 [138] 
 
S. Bach, A. Binder,
G. Montavon, F. Klauschen,
K.-R. Müller, W. Samek,

 
 On pixel-wise explanations for non-linear classifier
decisions by layer-wise relevance propagation,

 
 PloS one 10
(2015) e0130140.

 

 
 [139] 
 
A. Bau, Y. Belinkov,
H. Sajjad, N. Durrani,
F. Dalvi, J. Glass,

 
 Identifying and controlling important neurons in
neural machine translation,

 
 in: Proceedings of ICLR, 2019.

 

 
 [140] 
 
H. Strobelt, S. Gehrmann,
M. Behrisch, A. Perer,
H. Pfister, A. M. Rush,

 
 Seq2seq-vis: A visual debugging tool for
sequence-to-sequence models,

 
 IEEE transactions on visualization and computer
graphics 25 (2019)
353–363.

 

 
 [141] 
 
S. He, Z. Tu, X. Wang,
L. Wang, M. Lyu,
S. Shi,

 
 Towards understanding neural machine translation with
word importance,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 953–962.

 

 
 [142] 
 
A. Raganato, J. Tiedemann,

 
 An analysis of encoder representations in
transformer-based machine translation,

 
 in: Proceedings of EMNLP Workshop,
2018, pp. 287–297.

 

 
 [143] 
 
E. Voita, R. Sennrich,
I. Titov,

 
 The bottom-up evolution of representations in the
transformer: A study with machine translation and language modeling
objectives,

 
 in: Proceedings of EMNLP-IJCNLP,
2019, pp. 4396–4406.

 

 
 [144] 
 
F. Stahlberg, D. Saunders,
B. Byrne,

 
 An operation sequence model for explainable neural
machine translation,

 
 in: Proceedings of EMNLP Workshop,
2018, pp. 175–186.

 

 
 [145] 
 
C. Yun, S. Bhojanapalli,
A. S. Rawat, S. Reddi,
S. Kumar,

 
 Are transformers universal approximators of
sequence-to-sequence functions?,

 
 in: Proceedings of ICLR, 2020.

 

 
 [146] 
 
Z. Zhao, D. Dua,
S. Singh,

 
 Generating natural adversarial examples,

 
 in: Proceedings of ICLR, 2018.

 

 
 [147] 
 
M. T. Ribeiro, S. Singh,
C. Guestrin,

 
 Semantically equivalent adversarial rules for
debugging nlp models,

 
 in: Proceedings of ACL, 2018,
pp. 856–865.

 

 
 [148] 
 
Y. Cheng, Z. Tu, F. Meng,
J. Zhai, Y. Liu,

 
 Towards robust neural machine translation,

 
 in: Proceedings of ACL, 2018,
pp. 1756–1766.

 

 
 [149] 
 
J. Ebrahimi, D. Lowd,
D. Dou,

 
 On adversarial examples for character-level neural
machine translation,

 
 in: Proceedings of COLING, 2018,
pp. 653–663.

 

 
 [150] 
 
W. Zou, S. Huang, J. Xie,
X. Dai, J. Chen,

 
 A reinforced generation of adversarial examples for
neural machine translation,

 
 in: Proceedings of ACL, 2020,
pp. 3486–3497.

 

 
 [151] 
 
Y. Cheng, L. Jiang,
W. Macherey, J. Eisenstein,

 
 AdvAug: Robust adversarial augmentation for
neural machine translation,

 
 in: Proceedings of ACL, 2020,
pp. 5961–5970.

 

 
 [152] 
 
P. Michel, G. Neubig,

 
 Mtnt: A testbed for machine translation of noisy
text,

 
 in: Proceedings of EMNLP, 2018,
pp. 543–553.

 

 
 [153] 
 
J. Tiedemann,

 
 Opus–parallel corpora for everyone,

 
 Baltic Journal of Modern Computing
(2016) 384.

 

 
 [154] 
 
B. Zhang, P. Williams,
I. Titov, R. Sennrich,

 
 Improving massively multilingual neural machine
translation and zero-shot translation,

 
 arXiv preprint arXiv:2004.11867
(2020).

 

 
 [155] 
 
G. Wenzek, M.-A. Lachaux,
A. Conneau, V. Chaudhary,
F. Guzmán, A. Joulin,
É. Grave,

 
 Ccnet: Extracting high quality monolingual datasets
from web crawl data,

 
 in: Proceedings of The 12th Language Resources
and Evaluation Conference, 2020, pp.
4003–4012.

 

 
 [156] 
 
M. Abadi, P. Barham,
J. Chen, Z. Chen,
A. Davis, J. Dean,
M. Devin, S. Ghemawat,
G. Irving, M. Isard, et al.,

 
 Tensorflow: A system for large-scale machine
learning,

 
 in: 12th { \{ USENIX } \} symposium on operating
systems design and implementation ( { \{ OSDI } \} 16), 2016,
pp. 265–283.

 

 
 [157] 
 
A. Paszke, S. Gross,
F. Massa, A. Lerer,
J. Bradbury, G. Chanan,
T. Killeen, Z. Lin,
N. Gimelshein, L. Antiga, et al.,

 
 Pytorch: An imperative style, high-performance deep
learning library,

 
 in: Advances in NeurIPS, 2019,
pp. 8026–8037.

 

 
 [158] 
 
A. Vaswani, S. Bengio,
E. Brevdo, F. Chollet,
A. Gomez, S. Gouws,
L. Jones, Ł. Kaiser,
N. Kalchbrenner, N. Parmar,
R. Sepassi, N. Shazeer,
J. Uszkoreit,

 
 Tensor2Tensor for neural machine translation,

 
 in: Proceedings of AMTA, 2018,
pp. 193–199.

 

 
 [159] 
 
M. Ott, S. Edunov,
A. Baevski, A. Fan,
S. Gross, N. Ng,
D. Grangier, M. Auli,

 
 fairseq: A fast, extensible toolkit for sequence
modeling,

 
 in: Proceedings of NAACL-HLT (Demonstrations),
2019, pp. 48–53.

 

 
 [160] 
 
M. Luong, E. Brevdo,
R. Zhao,

 
 Neural machine translation (seq2seq) tutorial,

 
 https://github.com/tensorflow/nmt
(2017).

 

 
 [161] 
 
F. Hieber, T. Domhan,
M. Denkowski, D. Vilar,
A. Sokolov, A. Clifton,
M. Post,

 
 Sockeye: A toolkit for neural machine translation,

 
 arXiv preprint arXiv:1712.05690
(2017).

 

 
 [162] 
 
T. Chen, M. Li, Y. Li,
M. Lin, N. Wang,
M. Wang, T. Xiao,
B. Xu, C. Zhang,
Z. Zhang,

 
 Mxnet: A flexible and efficient machine learning
library for heterogeneous distributed systems,

 
 in: Proceedings of NeurIPS, Workshop,
2016.

 

 
 [163] 
 
M. Junczys-Dowmunt, R. Grundkiewicz,
T. Dwojak, H. Hoang,
K. Heafield, T. Neckermann,
F. Seide, U. Germann,
A. F. Aji, N. Bogoychev,
A. F. T. Martins, A. Birch,

 
 Marian: Fast neural machine translation in C++,

 
 in: Proceedings of ACL, System Demonstrations,
2018, pp. 116–121.

 

 
 [164] 
 
Z. Tan, J. Zhang,
X. Huang, G. Chen,
S. Wang, M. Sun,
H. Luan, Y. Liu,

 
 THUMT: An open-source toolkit for neural machine
translation,

 
 in: Proceedings of AMTA, 2020,
pp. 116–122.

 

 
 [165] 
 
A. Peris, F. Casacuberta,

 
 Nmt-keras: a very flexible toolkit with a focus on
interactive nmt and online learning,

 
 The Prague Bulletin of Mathematical Linguistics
111 (2018) 113–124.

 

 
 [166] 
 
K. Papineni, S. Roukos,
T. Ward, W. Zhu,

 
 Bleu: A method for automatic evaluation of machine
translation,

 
 in: Proceedings of ACL, 2002.

 

 
 [167] 
 
M. Post,

 
 A call for clarity in reporting bleu scores,

 
 arXiv preprint arXiv:1804.08771
(2018).

 

 
 [168] 
 
G. Neubig, Z.-Y. Dou,
J. Hu, P. Michel,
D. Pruthi, X. Wang,

 
 compare-mt: A tool for holistic comparison of
language generation systems,

 
 in: Proceedings of NAACL-HLT (Demonstrations),
2019, pp. 35–41.

 

 
 [169] 
 
P. Koehn, H. Hoang,
A. Birch, C. Callison-Burch,
M. Federico, N. Bertoldi,
B. Cowan, W. Shen,
C. Moran, R. Zens, et al.,

 
 Moses: Open source toolkit for statistical machine
translation,

 
 in: Proceedings of ACL on interactive poster and
demonstration sessions, 2007, pp. 177–180.