A review of on-device fully neural end-to-end automatic speech recognition algorithms 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2012.07974v3 [cs.LG] 27 Aug 2021 
 
 
 \secondlinename 
 Ankur Kumar, Sungsoo Kim, Abhinav Garg, and Changwoo Han

 

# A review of on-device fully neural end-to-end automatic speech
recognition algorithms

 
 
 Chanwoo Kim
 
    
 Dhananjaya Gowda
 
    
 Dongsoo Lee
 
    
 Jiyeon Kim
 † † thanks: Thanks to Samsung Electronics for funding this research.
The authors are thankful to President Sebasitan Seung, Executive Vice President Seunghwan Cho
and speech processing Lab. members at Samsung Research. 

 Abstract 
 
 In this paper, we review various end-to-end automatic speech recognition
algorithms and their optimization techniques for on-device applications.
Conventional speech recognition systems comprise a large number of discrete
components such as an acoustic model, a language model, a pronunciation model,
a text-normalizer, an inverse-text normalizer, a decoder based on a Weighted Finite State
Transducer (WFST), and so on. To obtain sufficiently high speech recognition
accuracy with such conventional speech recognition systems, a very large
language model (up to 100 GB) is usually needed. Hence, the corresponding
WFST size becomes enormous, which prohibits their on-device implementation. Recently,
fully neural network end-to-end speech recognition algorithms have been
proposed. Examples include speech recognition systems based on Connectionist Temporal Classification (CTC), Recurrent Neural Network Transducer (RNN-T), Attention-based Encoder-Decoder models (AED), Monotonic
Chunk-wise Attention (MoChA),
transformer-based speech recognition systems, and so on. These fully neural
network-based systems require much smaller memory footprints compared to
conventional algorithms, therefore their on-device implementation has become
feasible. In this paper, we review such end-to-end speech recognition models.
We extensively discuss their structures, performance, and advantages compared
to conventional algorithms.

 
 
 
 Index Terms:  
end-to-end speech recognition, attention-based model, recurrent neural network transducer, on-device speech recognition

 † † address: Samsung Research, Seoul, South Korea 
 {chanw.com, d.gowda, dongsoo3.lee, jstacey7.kim, 
 ankur.k, ss216.kim, abhinav.garg, cw1105.han}@samsung.com 
 

## 1 Introduction

 
 The advent of deep learning techniques has dramatically improved accuracy of
speech recognition models [ 1 ] .
Deep learning techniques first saw success by replacing
the Gaussian Mixture Model (GMM) of the Acoustic Model (AM) part of the
conventional speech recognition systems
 [ 2 ] with the Feed-Forward Deep Neural Networks
(FF-DNNs), further with Recurrent Neural Network (RNN) such as the
Long Short-Term Memory (LSTM) networks
 [ 3 ] or Convonlutional Neural
Networks (CNNs). In addition to this, there have been improvements in noise
robustness by using models motivated by auditory processing
 [ 4 , 5 , 6 ] , data augmentation techniques
 [ 7 , 8 , 9 ] , and beam-forming [ 10 ] .
Thanks to these advances, voice assistant devices such as Google Home
 [ 11 ] and Amazon Alexa
have been widely used at home environments.

 
 
 Nevertheless, it was not easy to run such high-performance speech recognition systems
on devices largely because of the size of the Weighted Finite State Transducer (WFST)
handling the lexicon and the language model.
Fortunately, all-neural end-to-end (E2E) speech recognition systems
were introduced which do not need a large WFST or an n-gram Language Model (LM)
 [ 12 ] .
These complete end-to-end systems have started surpassing the performance of
the conventional WFST-based decoders with a very large training
dataset [ 13 ] and a better choice of target unit
such as Byte Pair Encoded (BPE) subword units.

 
 
 In this paper, we provide a comprehensive review of the various components and algorithms of an end-to-end speech recognition system.
In Sec.  2 , we give a brief overview of the various neural building blocks of an E2E Automatic Speech Recognition (ASR) model.
The most popular E2E ASR architectures are reviewed in Sec.  3 .
Additional techniques used to improve the performance of E2E ASR models are discussed in Sec.  4 .
Techniques used for compression and quantization of the all-neural E2E ASR models are covered in Sec.  5 .
Sec.  6 gives a summary of the paper.

 
 
 

## 2 Neural network components for end-to-end speech recognition

 
 
 
 
 σ \sigma σ \sigma tanh 𝗵 [ m ] \bm{\mathsf{h}}_{[m]} 𝗰 [ m ] \bm{\mathsf{c}}_{[m]} input gate output gate forget gate 𝗵 [ m − 1 ] \bm{\mathsf{h}}_{[m-1]} 𝗰 [ m − 1 ] \bm{\mathsf{c}}_{[m-1]} 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} tanh σ \sigma 
 
 (a) 
Long Short-Term Memory (LSTM)
 [ 3 ] :
 𝗰 [ m ] \bm{\mathsf{c}}_{[m]} and 𝗵 [ m ] \bm{\mathsf{h}}_{[m]} are cell and
 hidden 
states at time m m 
respectively.

 
 
 
 𝘅 \bm{\mathsf{x}} 𝗵 \bm{\mathsf{h}} * relu D filters ∙ \bullet ∙ \bullet ∙ \bullet Fully Connected (opt.) 
 
 (b) 
Convolutional Neural Network (CNN): 𝘅 \bm{\mathsf{x}} and 𝗵 \bm{\mathsf{h}} are the input and the hidden output, respectively.

 
 
 
 MatMul Scale Mask (opt.) SoftMax MatMul 𝗤 \bm{\mathsf{Q}} 𝗞 \bm{\mathsf{K}} 𝗩 \bm{\mathsf{V}} 
 
 (c) 
Scaled Dot-Product Attention [ 14 ] :
 𝗤 \bm{\mathsf{Q}} , 𝗞 \bm{\mathsf{K}} , and 𝗩 \bm{\mathsf{V}} denote queries , keys , and values 
respectively.

 
 
 Figure 1 : 
Structures of neural net components frequently used for end-to-end speech
recognition: 
 (a) LSTM, (b) CNN, and (c) Scaled Dot-Product Attention.
 
 
 
 Table 1 : Size of the intermediate buffer required for streaming speech recognition [ 15 ] . L L is the sequence length, D D is the unit size or the dimension of each layer, and T T is the filter width of convolution. Refer to Sec. 2 about how to obtain the typical values shown in this table. 
 
 
 
 Model | 
 

 
 
 Memory | 

 
 Footprint | 

 | 
 Typical Value | 
 

 
 
 Intra-Sequence | 

 
 Parallelism | 

 | 

 
 
 
 LSTM | 
 O ⁡ ( N ​ D ) O(ND) | 
 98KB | 
 X | 

 
 Convolution | 
 O ⁡ ( T ​ N ​ D ) O(TND) | 
 245KB | 
 O | 

 
 Self-attention | 
 O ⁡ ( L ​ N ​ D ) O(LND) | 
 12MB | 
 O | 

 

 
 
 Fig. 1 illustrates neural network components commonly
employed in end-to-end speech recognition systems. Gated RNNs such as LSTMs
and Gated Recurrent Units (GRUs) [ 16 ] have been used
from the early days of encoder-decoder [ 17 ] and
attention-based models. The operation of an LSTM
is described by the following equation:

 
 
 
 | 

 
 | 
 𝗶 [ m ] \displaystyle\bm{\mathsf{i}}_{[m]} | 
 = σ ⁡ ( 𝗪 i ​ 𝘅 [ m ] + 𝗨 i ​ 𝗵 [ m − 1 ] + 𝗯 i ) \displaystyle=\sigma\big({\bm{\mathsf{W}}_{i}\bm{\mathsf{x}}_{[m]}+\bm{\mathsf{U}}_{i}\bm{\mathsf{h}}_{[m-1]}+\bm{\mathsf{b}}_{i}}\big) | 
 | 
 (1a) | 

 
 | 
 𝗳 [ m ] \displaystyle\bm{\mathsf{f}}_{[m]} | 
 = σ ⁡ ( 𝗪 f ​ 𝘅 [ m ] + 𝗨 f ​ 𝗵 [ m − 1 ] + 𝗯 f ) \displaystyle=\sigma\left({\bm{\mathsf{W}}_{f}\bm{\mathsf{x}}_{[m]}+\bm{\mathsf{U}}_{f}\bm{\mathsf{h}}_{[m-1]}+\bm{\mathsf{b}}_{f}}\right) | 
 | 
 (1b) | 

 
 | 
 𝗼 [ m ] \displaystyle\bm{\mathsf{o}}_{[m]} | 
 = σ ⁡ ( 𝗪 o ​ 𝘅 [ m ] + 𝗨 o ​ 𝗵 [ m − 1 ] + 𝗯 o ) \displaystyle=\sigma\left({\bm{\mathsf{W}}_{o}\bm{\mathsf{x}}_{[m]}+\bm{\mathsf{U}}_{o}\bm{\mathsf{h}}_{[m-1]}+\bm{\mathsf{b}}_{o}}\right) | 
 | 
 (1c) | 

 
 | 
 𝗰 ~ [ m ] \displaystyle\tilde{\bm{\mathsf{c}}}_{[m]} | 
 = tanh ⁡ ( 𝗪 c ​ 𝘅 [ m ] + 𝗨 c ​ 𝗵 [ m − 1 ] + 𝗯 c ) \displaystyle=\tanh\left({\bm{\mathsf{W}}_{c}\bm{\mathsf{x}}_{[m]}+\bm{\mathsf{U}}_{c}\bm{\mathsf{h}}_{[m-1]}+\bm{\mathsf{b}}_{c}}\right) | 
 | 
 (1d) | 

 
 | 
 𝗰 [ m ] \displaystyle\bm{\mathsf{c}}_{[m]} | 
 = 𝗶 [ m ] ⊙ 𝗰 ~ [ m ] + 𝗳 [ m ] ⊙ 𝗰 [ m − 1 ] \displaystyle=\bm{\mathsf{i}}_{[m]}\odot\tilde{\bm{\mathsf{c}}}_{[m]}+\bm{\mathsf{f}}_{[m]}\odot\bm{\mathsf{c}}_{[m-1]} | 
 | 
 (1e) | 

 
 | 
 𝗵 [ m ] \displaystyle\bm{\mathsf{h}}_{[m]} | 
 = 𝗼 [ m ] ⊙ tanh ⁡ ( 𝗰 [ m ] ) , \displaystyle=\bm{\mathsf{o}}_{[m]}\odot\tanh(\bm{\mathsf{c}}_{[m]}), | 
 | 
 (1f) | 
 

 where 𝗪 ( ⋅ ) \bm{\mathsf{W}}_{(\cdot)} and 𝗨 ( ⋅ ) \bm{\mathsf{U}}_{(\cdot)} are weight matrices,
 𝗯 ( ⋅ ) \bm{\mathsf{b}}_{(\cdot)} is the bias vector. 𝗶 ⁡ [ m ] \bm{\mathsf{i}}[m] , 𝗳 ⁡ [ m ] \bm{\mathsf{f}}[m] , and
 𝗼 [ m ] \bm{\mathsf{o}}_{[m]} are the input, forget, and output gates at time m m , respectively.
 σ ⁡ ( ⋅ ) \sigma(\cdot) is the sigmoid function and ⊙ \odot is the Hadamard product
operator. 𝗰 [ m ] \bm{\mathsf{c}}_{[m]} and 𝗵 [ m ] \bm{\mathsf{h}}_{[m]} are the cell and hidden states. Fig.
 1(a) shows the structure of an LSTM described by
( 1 ).
One of notable
advantages of gated RNNs is that they require relatively smaller memory
footprint compared to other models as shown
in Table 1 . Their another
advantage is streaming capability if unidirectional models ( e.g. 
uni-directional LSTMs or GRUs) are employed. Because of these advantages,
at the time of writing this paper, most of the commercially available end-to-end on-device speech recognition systems
are based on LSTMs [ 18 , 19 ] .
However, as shown in Table 1 , LSTMs have disadvantages
in terms of intra-sequence parallelism , since the computation needs to be
done sequentially.

 
 
 Various CNN-based approaches have been successfully employed in building
end-to-end speech recognition systems [ 20 , 21 ] . These approaches are characterized by a group
of filters and a nonlinear rectifier as shown in Fig. 1(b) .
As an example of various CNN-based approaches, depthwise 1-dimensional CNN
is represented by the following equations [ 15 ] :

 
 | 

 
 | 
 x [ m ] , d ′ \displaystyle x^{\prime}_{[m],\,d} | 
 = ∑ t = − ( T − 1 ) / 2 ( T − 1 ) / 2 W t , d ⋅ x [ m + t ] , d , 0 ≤ d ≤ D − 1 \displaystyle=\sum_{t=-(T-1)/2}^{(T-1)/2}{W_{t,d}}\cdot{x_{[m+t],\,d}},\quad 0\leq d\leq D-1 | 
 | 
 (2a) | 

 
 | 
 𝘅 [ m ] ′ \displaystyle\bm{\mathsf{x}}^{\prime}_{[m]} | 
 = [ x [ m ] ,  0 ′ , x [ m ] ,  1 ′ , ⋯ , x [ m ] , D − 1 ′ ] ⊺ \displaystyle=\big[x^{\prime}_{[m],\,0},x^{\prime}_{[m],\,1},\,\cdots\,,x^{\prime}_{[m],\,{D-1}}\big]^{\intercal} | 
 | 
 (2b) | 

 
 | 
 𝗵 [ m ] \displaystyle{\bm{\mathsf{h}}_{[m]}} | 
 = relu ​ ( 𝗩𝘅 [ m ] ′ + 𝗯 ) , \displaystyle=\text{relu}({\bm{\mathsf{V}}\bm{\mathsf{x}}^{\prime}_{[m]}+\bm{\mathsf{b}}}), | 
 | 
 (2c) | 
 

 where T T and D D are the length of the 1-dimensional filter and the number of
such filters respectively. x [ m ] , d ,  0 ≤ d ≤ D − 1 x_{[m],\,d},\,0\leq d\leq D-1 is the d d -th element of
the input 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} at the time index m m .
 𝗪 ∈ ℝ T × D \bm{\mathsf{W}}\in\mathbb{R}^{T\times D} ,
 𝗩 ∈ ℝ D × D \bm{\mathsf{V}}\in\mathbb{R}^{D\times D} , and 𝗯 ∈ ℝ D \bm{\mathsf{b}}\in\mathbb{R}^{D} 
are trainable variables. Unlike gated RNNs, to calculate the output of the current time step, CNN does not require the completion of computation for the previous time steps,
which enables intra-sequence parallelism as summarized in Table
 1 . Thus, CNN-based
end-to-end speech recognition systems have advantages in computational
efficiency on embedded processors supporting Single Instruction Multiple Data
(SIMD) by exploiting intra-sequence parallelism [ 15 ] .

 
 
 Recently, self-attention has been also successfully applied to speech recognition.
In [ 14 ] , self-attention mechanism was implemented
using a scaled-dot attention described by the following equation:

 

 
 | 
 Attention ​ ( 𝗤 , 𝗞 , 𝗩 ) = softmax ​ ( 𝗤𝗞 ⊺ d k ) ​ 𝗩 , \displaystyle\text{Attention}(\bm{\mathsf{Q}},\bm{\mathsf{K}},\bm{\mathsf{V}})=\text{softmax}\left(\frac{\bm{\mathsf{Q}}\bm{\mathsf{K}}^{\intercal}}{\sqrt{d_{k}}}\right)\bm{\mathsf{V}}, | 
 | 
 (3) | 
 

 where 𝗤 \bm{\mathsf{Q}} , 𝗞 \bm{\mathsf{K}} , and 𝗩 \bm{\mathsf{V}} are matrices representing a query , a key ,
and a value , and d k d_{k} is the dimension of the key . In
a self-attention layer, all of the queries , keys , and values 
are the outputs of the previous layer.

 
 
 In Table 1 , we compare typical values of memory
footprint required for stacks of these neural network layers
 [ 15 ] . These stacks correspond
to the neural net layers in Fig. 2(a) or the encoder portion of
Recurrent Neural Network-Transduer (RNN-T) [ 22 , 23 ] or attention-based models in Fig. 2(b) and Fig. 2(c) , respectively.
In obtaining these values, we assume that the number
of layer N N is 6 for the LSTM case [ 18 ] , and
15 for the CNN and the self-attention cases.
For speech recognition, L L is usually a few hundred, while T T is
about ten or less. Based on this, we assume that the dimension D D is 2048,
the sequence length L L is 100, and the filter length T T is 5, respectively,
to calculate typical values [ 15 ] .

 
 
 

## 3 End-to-End speech recognition architectures

 
 
 
 
 LSTM Softmax . . . LSTM 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} y ^ [ m ] \hat{y}_{[m]} LSTM 
 
 (a) 
CTC-loss-based Model.

 
 
 
 LSTM Softmax . . . LSTM 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} LSTM LSTM Joint Network 𝗵 [ m ] (enc) \bm{\mathsf{h}}^{\text{(enc)}}_{[m]} Encoder Prediction Network LSTM 𝗵 l (pred) \bm{\mathsf{h}}^{\text{(pred)}}_{l} 𝗷 [ m ] , l \bm{\mathsf{j}}_{[m],\,l} y ^ [ m ] , l \hat{y}_{[m],\,l} 𝘆 prev ​ ( [ m ] , l )   (embed) \bm{\mathsf{y}}^{\text{ (embed)}}_{\text{prev}([m],\,l)} 
 
 (b) 
RNN-T Model.

 
 
 
 LSTM Attention Softmax . . . 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} 𝗰 l \bm{\mathsf{c}}_{l} 𝘆 l − 1 (embed) \bm{\mathsf{y}}^{\text{(embed)}}_{l-1} 𝗵 [ m ] (enc) \bm{\mathsf{h}}^{\text{(enc)}}_{[m]} 𝗵 l − 1 (dec) \bm{\mathsf{h}}^{\text{(dec)}}_{l-1} LSTM LSTM LSTM y ^ l \hat{y}_{l} Encoder Attention Decoder 
 
 (c) 
Attention-based Model.

 
 
 Figure 2 : 
Comparison of block diagrams of different sequence-to-sequence speech
recognition approaches.
 
 
 
 Speech recognition is a task of finding the sequence-to-sequence mapping
from an input sequence of acoustic features to a output sequence of labels. Let
us denote the input and output sequences by 𝘅 [ 0 : M ] \bm{\mathsf{x}}_{[0:M]} and y 0 : L y_{0:L} 
as shown below:

 
 | 

 
 | 
 𝘅 [ 0 : M ] \displaystyle\bm{\mathsf{x}}_{[0:M]} | 
 = [ 𝘅 [ 0 ] , 𝘅 [ 1 ] , 𝘅 [ 2 ] , ⋯ , 𝘅 [ M − 1 ] ] , \displaystyle=\left[\bm{\mathsf{x}}_{[0]},\,\bm{\mathsf{x}}_{[1]},\,\bm{\mathsf{x}}_{[2]},\,\cdots,\,\bm{\mathsf{x}}_{[M-1]}\right], | 
 | 
 (4a) | 

 
 | 
 y 0 : L \displaystyle y_{0:L} | 
 = [ y 0 , y 1 , y 2 , ⋯ , y L − 1 ] , \displaystyle=\big[y_{0},\,y_{1},\,y_{2},\,\cdots,\,y_{L-1}\big], | 
 | 
 (4b) | 
 

 where M M and L L are the lengths of the input acoustic feature sequence and
the output label sequence, respectively. The sequence notation adopted in this
paper including ( 4 )
follows the Python array slice notation. In this paper, by convention, we use
a bracket to represent a periodically sampled sequence such as the acoustic feature,
and use a subscript to represent a non-periodic sequence such as the output label.
Fig. 2 shows structures of end-to-end all neural
speech recognition systems. Even though we use a stack of LSTMs in Fig.
 2 , any kinds of neural network components described in
Sec. 2 may be employed instead of LSTMs.

 
 

### 3.1 Connectionist Temporal Classification (CTC)

 
 The simplest way of implementing an
end-to-end speech recognizer is using a stack of neural network layers with
a Connectionist Temporal Classification loss [ 24 ] as
shown in Fig. 2(a) . This model defines a probability distribution
over the set of output labels augmented with a special blank symbol, ⟨ b ⟩ {\langle b\rangle} .
We define the set of all possible alignments
 ℬ CTC ( 𝘅 [ 0 : M ] , y 0 : L ) \mathcal{B}^{\text{CTC}}\left(\bm{\mathsf{x}}_{[0:M]},\,y_{0:L}\right) , as the set of all label
sequences 𝘇 [ 0 : M ] \bm{\mathsf{z}}_{[0:M]} where z ⁡ [ m ] ∈ 𝒴 ∪ ⟨ b ⟩ ,  0 ≤ m M z[m]\in\mathcal{Y}\cup{\langle b\rangle},\,0\leq m M , such that
 𝘇 [ 0 : M ] \bm{\mathsf{z}}_{[0:M]} is identical to y 0 : l y_{0:l} after removing all blank symbols
 ⟨ b ⟩ {\langle b\rangle} . 𝒴 \mathcal{Y} is the set of the entire alphabet of the output
labels. Under this assumption, the posterior probability of the output label sequence is given by the following equation:

 

 
 | 
 P ( y 0 : L | 𝘅 [ 0 : M ] ) = ∑ z [ 0 : M ] ∈ ℬ CTC ( 𝘅 [ 0 : M ] , y 0 : L ) Π m = 0 M − 1 P ( z [ m ] | 𝘅 [ 0 : M ] ) . \displaystyle P\left(y_{0:L}\big|\bm{\mathsf{x}}_{[0:M]}\right)=\sum_{\begin{subarray}{c}z_{[0:M]}\in\\
\mathcal{B}^{\text{CTC}}\left(\bm{\mathsf{x}}_{[0:M]},\,y_{0:L}\right)\end{subarray}}{\Pi_{m=0}^{M-1}P\left(z_{[m]}\big|\bm{\mathsf{x}}_{[0:M]}\right)}. | 
 | 
 (5) | 
 

 The CTC loss is defined by the following equation:

 

 
 | 
 𝕃 CTC = − E [ log ( P ( y 0 : L | 𝘅 [ 0 : M ] ) ) ] . \displaystyle\mathbbm{L}_{\text{CTC}}=-E\Big[\log\left(P\left(y_{0:L}\big|\bm{\mathsf{x}}_{[0:M]}\right)\right)\Big]. | 
 | 
 (6) | 
 

 The parameters of a model with the CTC loss are updated using the forward-backward recursion assuming conditional indepedence
 [ 24 ] , which is similar to the
forward-backward algorithm used for training Hidden Markov Models (HMMs)
 [ 2 ] .

 
 
 

### 3.2 Recurrent Neural Network-Transducer (RNN-T)

 
 In spite of its simplicity, the model described in Sec. 3.1 has several shortcomings including the conditional independence assumption
during the model training [ 25 ] and the lack of explicit feedback paths from the output label.
An improved version of this model is the RNN-T model [ 23 ] 
shown in Fig. 2(b) . In this model,
there is an explicit feedback path
from the output label to the prediction network which plays a similar role to
an LM. The probability model of an RNN-T based model is similar to the CTC
model:

 

 
 | 
 P ( y 0 : L | 𝘅 [ 0 : M ] ) = ∑ z [ 0 : M + L ] ∈ ℬ RNN-T ( 𝘅 [ 0 : M ] , y 0 : L ) Π u = 0 M + L − 1 P ( z [ u ] | 𝘅 [ 0 : M ] ) , \displaystyle P\left(y_{0:L}\big|\bm{\mathsf{x}}_{[0:M]}\right)=\sum_{\begin{subarray}{c}z_{[0:M+L]}\in\\
\mathcal{B}^{\text{RNN-T}}\big(\bm{\mathsf{x}}_{[0:M]},\,y_{0:L}\big)\end{subarray}}{\Pi_{u=0}^{M+L-1}P\left(z_{[u]}\big|\bm{\mathsf{x}}_{[0:M]}\right)}, | 
 | 
 (7) | 
 

 where ℬ RNN-T ( 𝘅 [ 0 : M ] , y 0 : L ) \mathcal{B}^{\text{RNN-T}}\left(\bm{\mathsf{x}}_{[0:M]},\,y_{0:L}\right) is the set of all possible label sequences 𝘇 [ 0 : M + L ] \bm{\mathsf{z}}_{[0:M+L]} where z ⁡ [ u ] ∈ 𝒴 ∪ ⟨ b ⟩ ,  0 ≤ u M + L z[u]\in\mathcal{Y}\cup{\langle b\rangle},\,0\leq u M+L , such that after removing blank symbols ⟨ b ⟩ \langle b\rangle from 𝘇 [ 0 : M + L ] \bm{\mathsf{z}}_{[0:M+L]} , it becomes the same as y 0 : L y_{0:L} . As in
Sec. 3.1 , 𝒴 \mathcal{Y} is the set of the entire alphabet of the
output labels.

 
 
 

### 3.3 Attention-based Models

 
 Another popular end-to-end speech recognition approach is employing the attention-mechanism
as shown in Fig. 2(c) [ 12 ] . In attention-based approaches, we use
the attention between the encoder and decoder hidden outputs. The equation for
encoder-decoder attention is basically the same as ( 3 ):

 
 | 

 
 | 
 e [ m ] , l \displaystyle e_{[m],\,l} | 
 = Energy ​ ( 𝗵 [ m ] ( enc ) , 𝗵 l − 1 ( dec ) ) \displaystyle=\text{\it Energy}(\bm{\mathsf{h}}^{(\text{\it enc})}_{[m]},\bm{\mathsf{h}}^{(\text{\it dec})}_{l-1}) | 
 | 
 (8a) | 

 
 | 
 a [ m ] , l \displaystyle a_{[m],\,l} | 
 = softmax ​ ( e [ m ] , l ) \displaystyle=\text{softmax}(e_{[m],\,l}) | 
 | 
 (8b) | 

 
 | 
 𝗰 l \displaystyle\bm{\mathsf{c}}_{l} | 
 = ∑ m = 0 M − 1 a [ m ] , l ​ 𝗵 [ m ] ( enc ) , \displaystyle=\sum_{m=0}^{M-1}a_{[m],\,{l}}\bm{\mathsf{h}}^{(\text{\it enc})}_{[m]}, | 
 | 
 (8c) | 
 

 where 𝗰 l \bm{\mathsf{c}}_{l} is the context vector which is used as the input to the decoder.
 𝗵 ( enc ) ​ [ m ] \bm{\mathsf{h}}^{\text{({\it enc})}}[m] and 𝗵 l − 1 ( dec ) \bm{\mathsf{h}}^{\text{({\it dec})}}_{l-1} 
are hidden outputs from the encoder at time m m and from the decoder at the label
index l − 1 l-1 , respectively.
 e [ m ] , l e_{[m],\,l} in ( 8a ) is the energy for the input
time index m m and the output label index l l .

 
 
 

### 3.4 Monotonic Chunkwise Attention(MoChA)-based Models

 
 Although the attention-based approach in 3.3 has been quite
successful, on-line streaming recognition with this approach has been
a challenge. This is because the entire sequence of input features
must be encoded before the decoder starts generating the first output
label. Several variations of the attention including Monotonic Chunkwise
Attention (MoChA) [ 26 ] 
have been proposed to resolve this problem.
In the MoChA model, there
are two attention mechanisms: a hard monotonic attention followed by a soft chunkwise
attention. The hard monotonic attention is employed to determine which element
should be attended from a sequence of hidden encoder outputs
 𝗵 [ m ] ( enc ) \bm{\mathsf{h}}^{\text{({\it enc})}}_{[m]} .
The hard monotonic attention is obtained from the hidden encoder output
 𝗵 [ m ] ( enc ) \bm{\mathsf{h}}^{\text{({\it enc})}}_{[m]} at the time index m m and the hidden
decoder output 𝗵 l − 1 ( dec ) \bm{\mathsf{h}}^{(\text{\it dec})}_{l-1} at the output label index
 l − 1 l-1 as follows:

 
 | 

 
 | 
 e [ m ] , l ( mono ) \displaystyle e^{\text{({\it mono})}}_{[m],\,l} | 
 = MonotonicEnergy ​ ( 𝗵 [ m ] ( enc ) , 𝗵 l − 1 ( dec ) ) \displaystyle=\text{\it MonotonicEnergy}(\bm{\mathsf{h}}^{(\text{\it enc})}_{[m]},\bm{\mathsf{h}}^{(\text{\it dec})}_{l-1}) | 
 | 
 (9a) | 

 
 | 
 a [ m ] , l ( mono ) \displaystyle a^{\text{({\it mono})}}_{[m],l} | 
 = σ ⁡ ( e [ m ] , l ( mono) ) \displaystyle=\sigma(e^{\text{(\it mono)}}_{[m],\,{l}}) | 
 | 
 (9b) | 

 
 | 
 z [ m ] , l \displaystyle z_{[m],l} | 
 ∼ Bernoullli ​ ( a [ m ] , l ( mono ) ) , \displaystyle\sim\text{\it Bernoullli}\left(a^{(\text{\it mono})}_{[m],\,{l}}\right), | 
 | 
 (9c) | 
 

 where σ ⁡ ( ⋅ ) \sigma(\cdot) is a logistic sigmoid function and MonotonicEnergy is the
energy function defined as follows [ 26 ] :

 

 
 | 
 MonotonicEnergy | 
 ( 𝗵 [ m ] ( enc ) , 𝗵 l − 1 ( dec ) ) \displaystyle(\bm{\mathsf{h}}^{\text{({\it enc})}}_{[m]},\;\bm{\mathsf{h}}^{\text{({\it dec})}}_{l-1}) | 
 | 
 
 
 | 
 = g ​ v ⊺ ‖ v ‖ \displaystyle\;=g\frac{v^{\intercal}}{\|v\|} | 
 tanh ⁡ ( W ( dec ) ​ 𝗵 l − 1 ( dec ) + W ( enc ) ​ 𝗵 [ m ] ( enc ) + b ) + r , \displaystyle\tanh(W^{\text{({\it dec})}}\bm{\mathsf{h}}^{\text{({\it dec})}}_{l-1}+W^{\text{({\it enc})}}\bm{\mathsf{h}}^{\text{({\it enc})}}_{[m]}+b)+r, | 
 | 
 (10) | 
 

 where v , W ( dec ) , W ( enc ) , b , g v,W^{\text{({\it dec})}},W^{\text{({\it enc})}},b,g , and r r are learnable variables. After finding out the
position to attend using ( 9c ), a soft attention with a fixed
chunk size is employed to generate the context vector that will be given as an
input to the decoder.
We refer readers interested in the more detailed structure of MoChA to
 [ 18 , 26 ] .
A block schematic of an on-device speech recognizer based on MoChA is shown in Fig.  3 .

 
 
 

### 3.5 Comparison between MoChA and RNN-T based models

 
 In this section, we compare a MoChA-based model described in Sec.
 3.4 with a RNN-T-based model discussed in Sec. 3.2 .
For these RNN-T and MoChA models, we used the same encoder structures
consisting of six LSTM layers with the unit size of 1,024.
The lower three LSTM layers in the encoder are interleaved with 2:1
 max-pool layers as in [ 18 ] .
A single LSTM layer with the unit size of 1024 is employed for both the decoder
of the MoChA-based model and the prediction network of RNN-T-based model.
Training was performed using an in-house training
toolkit built with the same Keras [ 27 ] and Tensorflow 2.3 APIs [ 28 ] .
A 40-dimension power-mel feature with a power coefficient of 1/15
 [ 4 ] is used as the input feature. We prefer the power-mel 
feature to the more frequently used log-mel feature since
the power-mel feature shows better speech recognition accuracy
 [ 29 , 30 , 31 ] .
In Table 2 , we compare the performance of
MoChA and RNN-T in terms of speech recognition accuracy and latency.
The model is compressed by 8-bit quantization and Low Rank Approximation
(LRA) [ 32 ] . More details about the compression procedure
is described in [ 18 ] . The latency was measured on a
 Samsung Galaxy Note 10 device.
As shown in this table, the MoChA-based model shows slightly better
speech recognition accuracy compared to the RNN-T-based model.
However, the RNN-T-based model is noticeably better than the MoChA-based
model in terms of latency and consistency of latency. The MoChA-based model
shows more variation in latency compared to the RNN-T based model.

 
 
 Table 2 : 
Performance comparison between the MoChA-based model and the RNN-T based model on Librispeech test-clean evaluation set.
 
 
 
 
 Model | 
 
 
 
 WER | 

 | 
 
 
 
 Avg. Latency | 

 | 

 
 
 
 MoChA | 
 6.88 % | 
 225 ms | 

 
 RNN-T | 
 7.63 % | 
 86.5 ms | 

 

 
 
 LSTM Chunkwise Attention Linear 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} 𝗰 l − 1 \bm{\mathsf{c}}_{l-1} y l − 1 y_{l-1} 𝗵 [ m ] ( e ​ n ​ c ) \bm{\mathsf{h}}^{(enc)}_{[m]} 𝗵 l ( d ​ e ​ c ) \bm{\mathsf{h}}^{(dec)}_{l} LSTM Max-pool Encoder Decoder Monotonic Attention z [ m ∗ ] , l = 1 z_{[m^{*}],\,l}=1 LSTM MaxOut Softmax Softmax 𝕃 = 𝕃 CE + 𝕃 CTC \mathbb{L}=\mathbb{L}_{\text{CE}}+\mathbb{L}_{\text{CTC}} Attention m ∗ m^{*} 𝗰 l \bm{\mathsf{c}}_{l} z [ m ] , l z_{[m],\,l} × \times 3 × \times 3 y ^ l \hat{y}_{l} 
 
 Figure 3 : 
The structure of an on-device speech recognition system based on
Monotonic Chunkwise Attention (MoChA) [ 18 ] 
 
 
 
 
 

## 4 Approaches for further performance improvement

 
 Recently, various techniques have been proposed to further improve the
performance of all neural end-to-end speech recognition systems described in
Sec. 3 . We will discuss some of these techniques in this
section.

 
 

### 4.1 Combination with a non-streaming model

 
 LSTM Streaming MoCha Attention + Decoder 𝘅 [ m ] \bm{\mathsf{x}}_{[m]} 𝗵 [ m ] (stream_enc) \bm{\mathsf{h}}^{\text{(stream\_enc)}}_{[m]} LSTM Max-Pool Shared Uni-directional Encoder × \times 3 × \times 2 LSTM Backward LSTM 𝗵 [ m ] (batch_enc) \bm{\mathsf{h}}^{\text{(batch\_enc)}}_{[m]} Batch Full Attention + Decoder Streaming Result Low-Latency Batch Result 𝘆 l − 1 ( stream_embed) \bm{\mathsf{y}}^{\text{(stream\_embed)}}_{l-1} 𝘆 l − 1 ( batch_embed) \bm{\mathsf{y}}^{\text{(batch\_embed)}}_{l-1} y ^ l (stream) \hat{y}^{\text{(stream)}}_{l} y ^ l (batch) \hat{y}^{\text{(batch)}}_{l} 
 
 Figure 4 : 
Speech recognition system that performs streaming recognition followed
by low-latency batch recognition 
 for improved performance.
 
 
 
 Even though Bidirectional LSTMs (BLSTMs) perform significantly better than
unidirectional counterparts for on-device
applications [ 18 ] , latency requirement usually prohibits their usage.
To get the advantage of BLSTMs without significantly affecting overall latency,
we proposed a recognition system combining the streaming MoChA model with a batch
full attention model in [ 33 ] . The entire structure
of this system is shown in Fig. 4 . In this model, there is
a shared streaming encoder consisting of uni-directional LSTMs. On one side,
the streaming MoChA attention and decoder can generate streaming speech
recognition result. On the other side, one backward LSTM layer on top of the
shared unidirectional layer comprise a BLSTM layer, which is followed by a full
attention model. When the user finishes speaking, then this full attention
model generates low-latency batch speech recognition result. The latency impact is relatively small since there is only one layer of backward LSTM layer in
the encoder. The experimental result is summarized in Table
 3 . We use the same model architecture and
experiment configuration as described in Sec. 3.5 .
As shown in this table, the proposed model in the bottom row shows
significantly better result than the MoChA-based model shown in the top row of
this table with endurable increase in latency.

 
 
 Table 3 : 
Word Error Rate (WER) comparison between a streaming model, a non-streaming model, and
a streaming model combined with a single layer of backward LSTM followed by a full
attention to minimize the impact on latency.
 
 
 
 
 Model | 
 
 
 
 LibriSpeech | 

 
 test-clean | 

 | 
 
 
 
 LibriSpeech | 

 
 test-other | 

 | 

 
 
 
 
 
 
 Uni-Directional Encoder | 

 
 + MoChA-Attention | 

 | 
 6.88 % | 
 19.11 % | 

 
 
 
 
 Bi-Directional Encoder | 

 
 + Full-Attention | 

 | 
 3.62 % | 
 11.20 % | 

 
 
 
 
 Shared Uni-Directional Encoder | 

 
 + Full-Attention | 

 | 
 4.09 % | 
 12.04 % | 

 

 
 
 

### 4.2 Shallow-fusion with language models

 
 End-to-end speech recognition models discussed in Sec. 3 
are trained using only paired speech-text data. Compared to traditional
AM-LM approaches where an LM is often trained using a much larger text corpus
possibly containing billions of sentences [ 34 ] ,
the end-to-end speech recognition model sees much limited number of word sequences during
the training phase. To get further performance improvement, researchers have
proposed various techniques of incorporating external language models such as
 shallow-fusion [ 35 ] , deep-fusion 
 [ 36 ] , and cold-fusion 
 [ 37 ] . Among them, in spite of its simplicity, shallow-fusion
seems to be more effective than other approaches
 [ 38 ] . In shallow-fusion, the log probability from the
end-to-end speech recognition model is linearly interpolated with the
probability from the language model as follows:

 

 
 | 
 log p sf ( y l | 𝘅 [ 0 : m ] ) \displaystyle\log p_{\text{sf}}\left(y_{l}\big|\bm{\mathsf{x}}_{[0:m]}\right) | 
 = log p ( y l | 𝘅 [ 0 : m ] , y ^ 0 : l ) + λ log p lm ( y l | y ^ 0 : l ) \displaystyle=\log p\left(y_{l}\big|\bm{\mathsf{x}}_{[0:m]},\hat{y}_{0:l}\right)+\lambda\log p_{\text{lm}}\left(y_{l}\big|\hat{y}_{0:l}\right) | 
 | 
 

 where p lm ( y l | y ^ 0 : l ) p_{\text{lm}}\left(y_{l}\big|\hat{y}_{0:l}\right) is the
probability of predicting the label y l y_{l} from the LM, and the
 p ( y l | 𝘅 [ 0 : m ] ) p\left(y_{l}\big|\bm{\mathsf{x}}_{[0:m]}\right) is the posterior probability
obtained from the end-to-end speech recognition model.

 
 
 Shallow-fusion with an LM is helpful in enhancing speech performance in general
domains, thus, most of the state of the art speech recognition results on
publicly available test sets ( ( e.g) LibriSpeech [ 39 ] ) are obtained with this technique
 [ 8 , 29 , 40 ] . In addition to this,
this technique is also very useful for enhancing performance in special domain
like personal names, geographic names, or music names by employing
domain-specific LMs. In case of n-gram LMs, they can be easily built on-device
to support personalized speech recognition [ 18 ] .
For on-device command recognition, shallow-fusion with a WFST is also useful for
specific domains since WFST contains a list of words not just subword units
 [ 41 ] .

 
 
 

### 4.3 Improving NER performance using a spell corrector

 
 Even though end-to-end all neural speech recognition systems have shown quite
remarkable speech recognition accuracy with small memory footprint, it has been
frequently observed that performance is poorer in recognizing named entities
 [ 42 ] . Compared to conventional speech
recognizers built with a WFST containing dictionary information,
an end-to-end speech recognition system does not explicitly contain a list of
named entities. Therefore, the speech recognition accuracy of such systems
is generally low for specialized domains handling song names,
composers’ names, personal names, and geographical names. Without
dictionary information, spelling errors may also occur with all
neural speech recognizers. This named entity recognition issue can be somewhat relieved by applying
shallow-fusion with an LM as described in Sec. 4.2 . However, more drastic
performance improvement is usually obtained by classifying the domain from the
speech recognition output and applying spell correction using a list of
named entities found in that domain. In [ 41 ] , a multi-stage spell correction approach was proposed to handle a large list of named entities with on-device speech recognizers.

 
 
 
 

## 5 Compression

 
 To run speech recognition models on embedded processors with limited memory, we often need to further reduce the parameter size in order to satisfy computational cost and memory footprint requirements. There are many known techniques to reduce the model size such as quantization, factorization, distillation, and pruning. The simplest way of accomplishing compression may be applying 8-bit quantization, which is supported by
 TensorflowLite TM {}^{\text{TM}} . Several commercially available speech recognition systems [ 18 , 19 ] have been built using this technique. In feed-forward neural networks, 2-bit or 3-bit quantization has been
successfully employed [ 43 ] . However, for general purpose processors, there are relatively small gains in going below 8-bit quantization, since Arithmetic Logic Units (ALUs) usually do not support sub 8-bit arithmetic. Another popular technique in reducing the model size of LSTMs is applying Low Rank Approximation (LRA) [ 44 ] . As an example, the DeepTwist algorithm, which is based on LRA, is employed to reduce the parameter size in our previous work [ 18 , 41 ] . More specifically, we were able to reduce the size of MoChA-based models from 531 MB to less than 40 MB without sacrificing the performance using 8-bit quantization and LRA.
Pruning may be a good choice if hardware supports sparse matrix arithmetic. In [ 45 ] , authors propose a three-stage pipeline:
pruning, trained quantization and Huffman coding, that work together
to reduce the storage requirement of neural networks by 35x to 49x
without affecting their accuracy.
In [ 46 ] , the authors studied effectiveness of different compression techniques for end-to-end speech recognition models. They conclude that pruning is the most effective with proper hardware support. In the absence of this, for small models, distillation appears to be the best choice, while factorization appears to be the best approach for larger models.

 
 
 

## 6 Conclusions

 
 In this paper, we reviewed various end-to-end all neural automatic speech recognition
systems and their optimization techniques for on-device applications.
On-device speech recognition has huge advantages compared to the server-side
ones in terms of user privacy, operation without internet, server-cost, and latency.
To operate speech recognition systems on embedded processors, we need to
consider several factors such as recognition accuracy, computational cost, latency,
and the model size.
We compared pros and cons of different neural network components such as Long
Short-Term Memory (LSTM), Convolutional Neural Network (CNN),
and attention mechanism.
We explained and compared different end-to-end neural speech recognition architectures such as
a stack of LSTM layers with the Connectionist Temporal Classification (CTC)
loss [ 24 ] ,
Recurrent Neural Network-Transformer(RNN-T) [ 22 , 23 ] ,
attention-based models, and models based on Monotonic Chunk-wise Attention
(MoChA) [ 26 ] . Further
improvement is achieved by combining a streaming model with a low-latency non-streaming model,
by applying shallow-fusion with a Language Model (LM), and by applying spell correction using
a list of named entities [ 47 ] . We also discussed several model compression
techniques including quantization, singular value decomposition, pruning, and
knowledge distillation.
These recent advances in all neural end-to-end speech recognition made it
possible to commercialize all neural on-device end-to-end speech recognition
systems [ 18 , 19 , 41 ] .

 
 
 

## References

 
 
 [1] 
 
G. Hinton, L. Deng, D. Yu, G. E. Dahl, A. Mohamed, N. Jaitly, A. Senior, V.
Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury, “Deep neural networks
for acoustic modeling in speech recognition: The shared views of four
research groups,” IEEE Signal Processing Magazine , vol. 29, no. 6,
Nov. 2012.

 

 
 [2] 
 
L. R. Rabiner, “A tutorial on hidden markov models and selected applications
in speech recognition,” Proceedings of the IEEE , vol. 77, no. 2, pp.
257–286, Feb 1989.

 

 
 [3] 
 
S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural
Computation , no. 9, pp. 1735–1780, Nov. 1997.

 

 
 [4] 
 
C. Kim and R. M. Stern, “Power-Normalized Cepstral Coefficients (PNCC)
for Robust Speech Recognition,” IEEE/ACM Trans. Audio, Speech, Lang.
Process. , pp. 1315–1329, July 2016.

 

 
 [5] 
 
C. Kim, K. Chin, M. Bacchiani, and R. M. Stern, “Robust speech
recognition using temporal masking and thresholding algorithma,” in
 INTERSPEECH-2014 , Sept. 2014, pp. 2734–2738.

 

 
 [6] 
 
C. Kim and R. M. Stern, “Feature extraction for robust speech
recognition using a power-law nonlinearity and power-bias subtraction,” in
 INTERSPEECH-2009 , Sept. 2009, pp. 28–31.

 

 
 [7] 
 
C. Kim, K. Kumar and R. M. Stern, “Robust speech recognition using
small power boosting algorithm,” in IEEE Automatic Speech Recognition
and Understanding Workshop (ASRU) , Dec. 2009, pp. 243–248.

 

 
 [8] 
 
D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le,
“SpecAugment: A Simple Data Augmentation Method for Automatic Speech
Recognition,” in Proc. Interspeech 2019 , 2019, pp. 2613–2617.

 

 
 [9] 
 
C. Kim, E. Variani, A. Narayanan, and M. Bacchiani, “Efficient
implementation of the room simulator for training deep neural network
acoustic models,” in INTERSPEECH-2018 , Sept 2018, pp. 3028–3032.

 

 
 [10] 
 
J. Heymann, L. Drude, and R. Haeb-Umbach, “Neural network based spectral
mask estimation for acoustic beamforming,” in 2016 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) , March 2016,
pp. 196–200.

 

 
 [11] 
 
C. Kim, A. Misra, K. Chin, T. Hughes, A. Narayanan, T. N. Sainath, and
M. Bacchiani, “Generation of large-scale simulated utterances in virtual
rooms to train deep-neural networks for far-field speech recognition in
google home,” in Proc. Interspeech 2017 , 2017, pp. 379–383.

 

 
 [12] 
 
J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio,
“Attention-based models for speech recognition,” in Advances in
Neural Information Processing Systems 28 , C. Cortes, N. D. Lawrence, D. D.
Lee, M. Sugiyama, and R. Garnett, Eds. Curran Associates, Inc., 2015, pp. 577–585.

 

 
 [13] 
 
C.-C. Chiu, T. N. Sainath, Y. Wu, R. Prabhavalkar, P. Nguyen,
Z. Chen, A. Kannan, R. J. Weiss, K. Rao, E. Gonina, N. Jaitly,
B. Li, J. Chorowski, and M. Bacchiani, “State-of-the-art speech
recognition with sequence-to-sequence models,” in 2018 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP) , April 2018, pp. 4774–4778.

 

 
 [14] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u.
Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances
in Neural Information Processing Systems 30 , I. Guyon, U. V. Luxburg,
S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds. Curran Associates, Inc., 2017, pp.
5998–6008.

 

 
 [15] 
 
J. Park, C. Kim, and W. Sung, “Convolution-based attention model with
positional encoding for streaming speech recognition on embedded devices,”
in 2021 IEEE Spoken Language Technology Workshop (SLT) , Jan. 2021.

 

 
 [16] 
 
K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares,
H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN
encoder–decoder for statistical machine translation,” in
 Proceedings of the 2014 Conference on Empirical Methods in Natural
Language Processing (EMNLP) . Doha,
Qatar: Association for Computational Linguistics, Oct. 2014, pp. 1724–1734.

 

 
 [17] 
 
I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with
neural networks,” in Advances in Neural Information Processing
Systems , Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Q.
Weinberger, Eds., vol. 27. Curran
Associates, Inc., 2014, pp. 3104–3112.

 

 
 [18] 
 
K. Kim, K. Lee, D. Gowda, J. Park, S. Kim, S. Jin, Y.-Y. Lee, J. Yeo, D. Kim,
S. Jung, J. Lee, M. Han, and C. Kim, “Attention based on-device
streaming speech recognition with large speech corpus,” in 2019 IEEE
Automatic Speech Recognition and Understanding Workshop (ASRU) , Dec. 2019,
pp. 956–963.

 

 
 [19] 
 
Y. He, T. N. Sainath, R. Prabhavalkar, I. McGraw, R. Alvarez,
D. Zhao, D. Rybach, A. Kannan, Y. Wu, R. Pang, Q. Liang,
D. Bhatia, Y. Shangguan, B. Li, G. Pundak, K. C. Sim, T. Bagby,
S. yiin Chang, K. Rao, and A. Gruenstein, “Streaming end-to-end speech
recognition for mobile devices,” in ICASSP 2019 - 2019 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP) , May 2019, pp. 6381–6385.

 

 
 [20] 
 
J. Park, Y. Boo, I. Choi, S. Shin, and W. Sung, “Fully neural network based
speech recognition on mobile and embedded devices,” in Advances in
Neural Information Processing Systems , S. Bengio, H. Wallach, H. Larochelle,
K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018, pp. 10 620–10 630.

 

 
 [21] 
 
A. Hannun, A. Lee, Q. Xu, and R. Collobert, “Sequence-to-Sequence Speech
Recognition with Time-Depth Separable Convolutions,” in Proc.
Interspeech 2019 , 2019, pp. 3785–3789.

 

 
 [22] 
 
A. Graves, “Sequence transduction with recurrent neural networks,” in
 International Conference of Machine Learning (ICML) 2012 Workshop on
Representation Learning .

 

 
 [23] 
 
A. Graves, A. rahman Mohamed, and G. Hinton, “Speech recognition with
deep recurrent neural networks,” in 2013 IEEE International Conference
on Acoustics, Speech and Signal Processing , May 2013, pp. 6645–6649.

 

 
 [24] 
 
A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist
temporal classification: Labelling unsegmented sequence data with recurrent
neural networks,” in Proceedings of the 23rd International Conference
on Machine Learning , ser. ICML ’06. New York, NY, USA: ACM, 2006, pp. 369–376.

 

 
 [25] 
 
S. Kim, T. Hori, and S. Watanabe, “Joint ctc-attention based end-to-end
speech recognition using multi-task learning,” in 2017 IEEE
International Conference on Acoustics, Speech and Signal Processing
(ICASSP) , 2017, pp. 4835–4839.

 

 
 [26] 
 
C.-C. Chiu and C. Raffel, “Monotonic chunkwise attention,” in
 International Conference on Learning Representations , Apr. 2018.

 

 
 [27] 
 
F. Chollet et al. , “Keras,” , 2015.

 

 
 [28] 
 
M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin,
S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga,
S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden,
M. Wicke, Y. Yu, and X. Zheng, “Tensorflow: A system for large-scale machine
learning,” in 12th USENIX Symposium on Operating Systems Design and
Implementation (OSDI 16) . Savannah,
GA: USENIX Association, 2016, pp. 265–283.

 

 
 [29] 
 
C. Kim, M. Shin, A. Garg, and D. Gowda, “Improved vocal tract length
perturbation for a state-of-the-art end-to-end speech recognition system,”
in INTERSPEECH-2019 , Graz, Austria, Sept. 2019, pp. 739–743.

 

 
 [30] 
 
C. Kim, M. Kumar, K. Kim, and D. Gowda, “Power-law nonlinearity with
maximally uniform distribution criterion for improved neural network training
in automatic speech recognition,” in 2019 IEEE Automatic Speech
Recognition and Understanding Workshop (ASRU) , Dec. 2019, pp. 988–995.

 

 
 [31] 
 
C. Kim, S. Kim, K. Kim, M. Kumar, J. Kim, K. Lee, C. Han, A. Garg, E. Kim,
M. Shin, S. Singh, L. Heck, and D. Gowda, “End-to-end training of a large
vocabulary end-to-end speech recognition system,” in 2019 IEEE
Automatic Speech Recognition and Understanding Workshop (ASRU) , Dec. 2019,
pp. 562–569.

 

 
 [32] 
 
D. Lee, P. Kapoor, and B. Kim, “Deeptwist: Learning model compression via
occasional weight distortion,” CoRR , vol. abs/1810.12823, 2018.

 

 
 [33] 
 
D. Gowda, A. Kumar, K. Kim, H. Yang, A. Garg, S. Singh, J. Kim, M. Kumar,
S. Jin, S. Singh, and C. Kim, “Utterance Invariant Training for Hybrid
Two-Pass End-to-End Speech Recognition,” in Proc. Interspeech 2020 ,
2020, pp. 2827–2831. 

 

 
 [34] 
 
A. Kannan, Y. Wu, P. Nguyen, T. N. Sainath, Z. Chen, and
R. Prabhavalkar, “An analysis of incorporating an external language model
into a sequence-to-sequence model,” in 2018 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP) , Apr. 2018,
pp. 5824–5828.

 

 
 [35] 
 
J. Chorowski and N. Jaitly, “Towards better decoding and language model
integration in sequence to sequence models,” in Proc. Interspeech
2017 , 2017, pp. 523–527.

 

 
 [36] 
 
Ç. Gülçehre, O. Firat, K. Xu, K. Cho, L. Barrault, H. Lin,
F. Bougares, H. Schwenk, and Y. Bengio, “On using monolingual corpora in
neural machine translation,” CoRR , vol. abs/1503.03535, 2015.

 

 
 [37] 
 
A. Sriram, H. Jun, S. Satheesh, and A. Coates, “Cold fusion: Training seq2seq
models together with language models,” 2017.

 

 
 [38] 
 
S. Toshniwal, A. Kannan, C. Chiu, Y. Wu, T. N. Sainath, and
K. Livescu, “A comparison of techniques for language model integration in
encoder-decoder speech recognition,” in 2018 IEEE Spoken Language
Technology Workshop (SLT) , 2018, pp. 369–375.

 

 
 [39] 
 
V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An asr corpus
based on public domain audio books,” in IEEE Int. Conf. Acoust.,
Speech, and Signal Processing , April 2015, pp. 5206–5210.

 

 
 [40] 
 
A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang,
Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented Transformer
for Speech Recognition,” in Proc. Interspeech 2020 , 2020, pp.
5036–5040. 

 

 
 [41] 
 
A. Garg, G. P. Vadisetti, D. Gowda, S. Jin, A. Jayasimha, Y. Han, J. Kim,
J. Park, K. Kim, S. Kim, Y. yoon Lee, K. Min, and C. Kim, “Streaming
On-Device End-to-End ASR System for Privacy-Sensitive Voice-Typing,” in
 Proc. Interspeech 2020 , 2020, pp. 3371–3375.

 

 
 [42] 
 
R. Prabhavalkar, K. Rao, T. N. Sainath, B. Li, L. Johnson, and N. Jaitly, “A
comparison of sequence-to-sequence models for speech recognition,” in
 Proc. Interspeech 2017 , 2017, pp. 939–943.

 

 
 [43] 
 
K. Hwang and W. Sung, “Fixed-point feedforward deep neural network design
using weights +1, 0, and -1,” in 2014 IEEE Workshop on Signal
Processing Systems (SiPS) , 2014, pp. 1–6.

 

 
 [44] 
 
I. McGraw, R. Prabhavalkar, R. Alvarez, M. G. Arenas, K. Rao,
D. Rybach, O. Alsharif, H. Sak, A. Gruenstein, F. Beaufays, and
C. Parada, “Personalized speech recognition on mobile devices,” in
 2016 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP) , 2016, pp. 5955–5959.

 

 
 [45] 
 
S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural
network with pruning, trained quantization and huffman coding,” in 4th
International Conference on Learning Representations, ICLR 2016, San Juan,
Puerto Rico, May 2-4, 2016, Conference Track Proceedings , Y. Bengio and
Y. LeCun, Eds., 2016. 

 

 
 [46] 
 
R. Pang, T. Sainath, R. Prabhavalkar, S. Gupta, Y. Wu, S. Zhang, and C.-C.
Chiu, “Compression of end-to-end models,” in Proc. Interspeech 2018 ,
2018, pp. 27–31. 

 

 
 [47] 
 
A. Garg, A. Gupta, D. Gowda, S. Singh, and C. Kim, “Hierarchical
Multi-Stage Word-to-Grapheme Named Entity Corrector for Automatic Speech
Recognition,” in Proc. Interspeech 2020 , 2020, pp. 1793–1797.