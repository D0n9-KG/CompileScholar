# Harnessing Event Sensory Data for Error Pattern Prediction in Vehicles: A Language Model Approach

Hugo Math, Rainer Lienhart, Robin Schön

Augsburg University, Augsburg 86159, Germany

# Abstract

In this paper, we draw an analogy between processing natural languages and processing multivariate event streams from vehicles in order to predict when and what error pattern is most likely to occur in the future for a given car. Our approach leverages the temporal dynamics and contextual relationships of our event data from a fleet of cars. Event data is composed of discrete values of error codes as well as continuous values such as time and mileage. Modelled by two causal Transformers, we can anticipate vehicle failures and malfunctions before they happen. Thus, we introduce CarFormer, a Transformer model trained via a new self-supervised learning strategy, and EPredictor, an autoregressive Transformer decoder model capable of predicting when and what error pattern will most likely occur after some error code apparition. Despite the challenges of high cardinality of event types, their unbalanced frequency of appearance and limited labelled data, our experimental results demonstrate the excellent predictive ability of our novel model. Specifically, with sequences of 160 error codes on average, our model is able with only half of the error codes to achieve 80% F1 score for predicting what error pattern will occur and achieves an average absolute error of $58.4 \pm 13.2$ h when forecasting the time of occurrence, thus enabling confident predictive maintenance and enhancing vehicle safety.

Code — https://github.com/Mathugo/AAAI2025-CarFormer-EPredictor

# 1 Introduction

Today's vehicles generate an astounding amount of data, typically reported as events on an irregular basis, but continuously over time. Some events occur simultaneously, while others are scattered over time, with unequally distributed time intervals between their occurrence. They usually report numerical and/or categorical features. In our work, we focus on processing and analyzing such multivariate and irregular event streams produced by modern cars, for which related research is still sparse. Our sequences of discrete events in time are known as DTCs (diagnostic trouble codes). DTCs are preferred over raw sensory data because they provide less granular and discrete information. This makes them easier to analyze.

Our goal is to learn the correlation between DTCs using a DTC-based language model to predict when and what with

![](images/fff1e89dca8034b615e366e4f4f3a4db155ca4156db1065d0e7654607bffb373.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["DTC1"] --> B["DTC2"]
    B --> C["DTC3"]
    C --> D["EP"]
    style A fill:#cce5ff,stroke:#333
    style B fill:#cce5ff,stroke:#333
    style C fill:#cce5ff,stroke:#333
    style D fill:#ffcccc,stroke:#333
```
</details>

Figure 1: Error pattern (EP) prediction (when and what with which probability) based on the past sequence S of diagnostic trouble codes (DTCs).

which probability an error pattern (EP) is occurring after having seen a number of DTCs (see Figure 1). Thus, we consider DTCs to be the words in our language. In contrast, EPs are very different from DTCs. They are defined by domain experts after observing DTC sequences. Therefore, they are way more precise about the critical error that the car is having. While some DTCs can also be noisy and repetitive events about recurring errors (e.g. electrical issues, software updates), EPs characterize a whole error sequence consisting of precise vehicle failures (e.g. engine or battery failures). Recent research has approached predictive maintenance in the automotive field using sequences of DTCs via RNNs (Hafeez, Alonso, and Ter-Sarkisov 2021) and, more recently, Transformers (Hafeez, Alonso, and Riaz 2024) to predict the next DTC. However, distinguishing minor errors or noise-like DTCs from important events such as error patterns (EPs) is essential since the latter poses higher risks and necessitates greater safety and maintenance measures, such as vehicle immobilization or addressing critical malfunctions. Furthermore, as the data volume increases, accurately predicting the next DTC becomes challenging, particularly when the event type cardinality approaches $\sim 10^{4}$ . This phenomenon is akin to language processing, where the accuracy of the next token prediction using greedy decoding or other methods exponentially decreases with sequence length and vocabulary size due to error accumulation (Bachmann and

Nagarajan 2024).

Historically, Hawkes Processes and their neural variants have advanced the state of the art in event modelling for next event and time prediction tasks (Hawkes 1971; Du et al. 2016; Shchur et al. 2021). Transformer-based models like BERT (Devlin et al. 2019) and GPT-3 (Brown et al. 2020) have gained overwhelming popularity due to their attention-based architecture, flexibility, parallelization, and state-of-the-art performance in sequence modelling. Consequently, models adapted to discrete-time sequences using Transformers have emerged naturally (Zhang et al. 2020a; Zuo et al. 2020; Shou et al. 2024), achieving state-of-the-art performance in next event prediction benchmarks. By leveraging the sequential nature of our data, we can define sentences as the concatenation of each discrete event:

$$
" <   s > D T C 1 D T C 2.. D T C n <   / s >"
$$

as shown in Figure 1, and embed it into $R^{D}$ . We make several modifications to the vanilla Transformer from (Vaswani et al. 2017), incorporating continuous-time and mileage positional embeddings as additional context. Using two distinct training phases, we introduce CarFormer a pretrained model acting as an encoder and EPredictor a decoder Transformer-based model that generates a probability distribution over a set of error patterns for each event step i to determine what EPs will most likely happen and estimates a time for when it will occur.

# Contributions

To the best of our knowledge, introducing a causal Transformer model for error pattern and DTC prediction (event type and time) has not been explored, despite some related work on event-data and next-DTC prediction (Shou et al. 2024; Hafeez, Alonso, and Riaz 2024; Hafeez, Alonso, and Ter-Sarkisov 2021). The main contributions of this paper are:

- CarFormer: An encoder Transformer-based model designed to ingest scattered continuous event streams from vehicles, trained via a multi-task learning strategy. This model will transform the DTC-sequences into hidden representations that can be processed by the EPredictor.   
- EPredictor: An autoregressive decoder Transformer-based model that specializes in predictive maintenance, particularly error patterns by estimating when and what error patterns will most likely occur.

# 2 Background and Related Work

# Event Sequence Modelling with TPP

Event data is commonly modelled via Temporal Point Processes (TPPs), which describe stochastic processes of discrete events. Each event is composed of a time of occurrence $t \in R^{+}$ and an event type $u \in U$ forming a pair $(t, u)$ . U is a finite set of discrete event types. A sequence is constructed with multiple pairs of events, such as $S = \{(t_{1}, u_{1}), ..., (t_{L}, u_{L})\}$ where $0 < t_{1} < ... < t_{n}$ .

TPPs are usually represented as a counting process $N(t) \forall t \geq 0$ for the events of type u, which describes the number of occurrences of an event over time. The goal is to predict the next event $(u', t')$ given the history $H_t := \{(t_i, u_i) \in \mathbb{R}^+ \times U | t_i < t\}$ of all events that occurred up to time t. We define $\lambda^*$ to model the instantaneous rate of an event in continuous time. Thus, the probability of occurrence for an event $(u', t')$ is conditioned on the history of events $H_t$ :

$$
\begin{array}{l} \lambda^ {*} (t) d t := P \left(\left(u ^ {\prime}, t ^ {\prime}\right): t ^ {\prime} \in [ t, t + d t) \mid H _ {t}\right) \tag {1} \\ = \mathbb {E} (d N (t) | H _ {t}) \\ \end{array}
$$

which we could translate as the expected number of events during an infinitesimal time window $[t, t + dt)$ knowing the history $H_{t}$ . We assume that two events do not occur simultaneously, i.e., $dN(t) \in \{0, 1\}$ .

The Hawkes Process (Hawkes 1971) has been arguably the most studied modelling technique for TPPs. It assumes some parametric form of the conditional intensity function $\lambda^{*}(t)$ and states that an event excites future events additively and decays using a function $f$ over time. For example:

$$
\lambda^ {*} (t) = \mu + \sum_ {(u _ {i}, t _ {i}) \in H _ {t}} f (t - t _ {i}) \tag {2}
$$

where $\mu\geq0$ is the base intensity. Neural TPPs, on the other hand, aim at reducing the inductive bias of the Hawkes Process, which states that past events excite future ones additively. They do this by approximating $\lambda^{*}$ using a neural network (RNN, LSTM, Transformer) (Zhang et al. 2020a; Du et al. 2016). Recent papers suggest using a generative and contrastive approach for event sequence modeling, showing promising results across predictive benchmarks. (Lin et al. 2022) uses next-event prediction as their main training objective, while (Shou et al. 2024) have three specific objectives plus a contrastive loss.

# Transformers for Event Streams

Encoding the history $H_{t}$ into historical hidden vectors using a Transformer enhanced the performance on event prediction benchmarks as shown in (Zuo et al. 2020) with the Transformer Hawkes Process (THP) or the self-attentive Hawkes Process (Zhang et al. 2020a). They typically reused the vanilla Transformer (Vaswani et al. 2017) and created two embeddings:

Time Embedding Time embedding replaces the traditional positional encoding which grants the Transformer model positional information of each token within the sequence. This time embedding is defined deterministically with periodic functions exactly like in (Vaswani et al. 2017):

$$
\mathbf {P} _ {i, j} := \left\{ \begin{array}{l l} \sin (t _ {i} \times \omega_ {0} ^ {j / d}) & \text { if   } j \mod 2 = 0 \\ \cos (t _ {i} \times \omega_ {0} ^ {(j - 1) / d}) & \text { if   } j \mod 2 = 1 \end{array} \right. \tag {3}
$$

where i is the index of the i-th event, $\omega_{0}$ is the frequency (usually $10^{-4}$ ).

Event-type Embedding To get a dense representation of our sequence, we embed each event into a d dimensional space using an embedding matrix $L^{V \times d}$ where V is the distinct number of events (vocabulary). As we would do for

word embeddings, we create a sequence of one-hot encoded vectors from the event types $\{u_{i}\}_{i=0}^{L}$ as $Y \in R^{L \times V}$ . Thus, the event-type embedding $E = YL \in R^{L \times d}$ and the input embedding U is defined as $U = E + P \in R^{L \times d}$

Attention The majority of research utilizing Transformer models for sequence data employs the architecture introduced by (Vaswani et al. 2017). We define three linear projection matrices $\mathbf{Q} = \mathbf{UW}^Q$ , $\mathbf{K} = \mathbf{UW}^K$ , and $\mathbf{V} = \mathbf{UW}^V$ . They are called query, key, and value, respectively. $\mathbf{W}^Q$ , $\mathbf{W}^K$ , $\mathbf{W}^V$ are trainable weights. Essentially $\mathbf{Q}$ represents what the model is looking based on the input $\mathbf{U}$ , $\mathbf{K}$ is the label for the input's information and $\mathbf{V}$ is the desired representation of the input's semantics. The attention score can be computed as:

$$
\mathbf {A} = \operatorname{softmax} (\mathbf {Q K} ^ {T} / \sqrt {d}) \tag {4}
$$

$$
\mathbf {C} = \mathbf {A V} \tag {5}
$$

where d denotes the number of attention heads and $A \in R^{L \times L}$ the attention scores of each event pair i, j. A final hidden representation H is obtained via a layer normalization (LayerNorm), a pointwise feed-forward neural network (FFN) and residual connections via:

$$
\mathbf {U} ^ {\prime} = \text { LayerNorm } (\mathbf {C} + \mathbf {U})
$$

$$
H = \text { LayerNorm } (\mathbf {U} ^ {\prime} + \text { FFN } (\mathbf {U})) \tag {6}
$$

# Self-supervised learning

By leveraging an efficient pre-training task (e.g. token masking or next token prediction) and then fine-tuning a smaller model with fewer parameters for specific tasks, Transformer-based models achieve state-of-the-art performance in natural language processing tasks (Devlin et al. 2019; Brown et al. 2020). For example, the GPT model is pre-trained on the next token prediction task, where the labels are generated by shifting the tokens to the right. Then, a classification head is added on top of the Transformer model to assign a probability to each token. Contrary to BERT (Devlin et al. 2019) a causal mask is applied so that tokens can only attend to the previous one, thus preventing cheating. In section 4 we will introduce a pre-trained model serving as an encoder trained on specific event prediction tasks to ensure adaptability to the event stream domain of vehicles. We found that modelling event streams with autoregressive Transformer-based models for fault predictions were not addressed well in the literature (Shchur et al. 2021) although there is some related work for TPPs (Lin et al. 2022; Shou et al. 2024).

# 3 Data

Overview Diagnostic data is generated by various Electronic Control Units (ECU) in a vehicle at irregular intervals. Diagnostic data differs substantially from raw sensor data, since diagnostic data or fault events are categorical and relate to various problems within the vehicle. We construct a Diagnosis Trouble Code (DTC) indicating the precise error from 3 pieces of information arriving at the same timestamp ts and mileage d: (1) the ID number of the ECU, (2) an error

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>u</td><td>Discrete event type, in our case a DTC.</td></tr><tr><td>d</td><td>Absolute mileage of the vehicle in km.</td></tr><tr><td>m</td><td>Mileage of the vehicle in km since the first DTC ( $u_0$ ) occurred in a sequence S such as  $m_i = d_i - d_0$ </td></tr><tr><td>ts</td><td>Unix timestamp attached to each DTC.</td></tr><tr><td>t</td><td>Number of hours passed since the first DTC occurred in a sequence. More specifically,  $t_i$  is defined as  $t_i = ts_i - ts_0$ .</td></tr><tr><td>S</td><td>Sequence of triplets (event type, time, mileage) defined as  $S = \{(u_i, t_i, m_i)\}_{i=0}^L$  of length L with index starting from 0.</td></tr><tr><td> $i \in \{0, ..., L\}$ </td><td>index of element in (event) sequence.</td></tr></table>

Table 1: List of symbols and their respective meanings

code (Base-DTC) and (3) a Fault-Byte. A single DTC token is composed of these 3 elements:

$$
D T C = E C U | B a s e - D T C | F a u l t - B y t e
$$

Thus, we can encode uniquely each DTC by a single token to predict the next token directly.

This research uses an anonymized vehicular DTC sequence dataset of $1.7 \times 10^{6}$ sequences with on average 150 DTCs per sequence. Each sequence belongs to a unique vehicle. In a sequence S, each DTC (commonly referred to as event type $u_{i}$ ) is attached with a time $t_{i}$ and a mileage $m_{i}$ constructing a single event ( $u_{i}, t_{i}, m_{i}$ ). You can find an overview of the DTC elements in Table 2.

<table><tr><td>Data</td><td># of values</td><td>Description</td></tr><tr><td>DTC</td><td>8710</td><td>Diagnostic Trouble Code</td></tr><tr><td>ECU</td><td>61</td><td>Electronic Control Unit</td></tr><tr><td>Base-DTC</td><td>7726</td><td>Error Code</td></tr><tr><td>Fault-Byte</td><td>2</td><td>Binary Value</td></tr></table>

Table 2: Number of Distinct values of the DTC elements

To get a full sequence $S = \{(u_i, t_i, m_i)\}_{i=0}^L$ , we obtain the last known timestamp $ts_L$ and mileage $d_L$ and select all DTCs that are no further than: (1) a given period in the past ( $ts_L - ts_i \leq 30$ days) and (2) a given distance in the past ( $d_L - d_i \leq 300km$ ). Table 1 explains the different data notation.

Time and Mileage We draw the distribution of $t_{i}$ and $m_{i}$ in Figure 2. We observe peaks at 0 on both distributions due to truncation and missing values. In order to feed our model with the time t feature, we need a scaling method to level out the left-tail distribution somewhat. Using $\log_{b}(t+1)$ is a natural choice. At the same time, we want to approximately map t into the range of [-1,1]. Therefore, we apply the following non-linear function $f_{t}: R^{+} \to R$ to t:

$$
t ^ {\prime} = f _ {t} (t, b) = \log_ {b} (t + 1) - 1 \forall t \in \mathbb {R} ^ {+} \tag {7}
$$

with b chosen appropriately and feed $t'$ into a neural network. In TPPs the time is usually represented as inter-event time or its logarithm (Shchur et al. 2021; Du et al. 2016).

![](images/b9df35cedf61e484d911ec9d995171af80b1e3e431b2f7e3a580bac99f1392ad.jpg)  
Figure 2: Distribution of $t_i$ and $m_i$ in our data set

# 4 Pre-training

# Embeddings

CarFormer uses 4 different embeddings to capture spatial and temporal dependencies of irregular event apparition. These embeddings differ from the positional embedding P used in TPPs (2020), (2020b) and next-DTC prediction studies (2024), (2021). We embed both time t and mileage m and use a rotation matrix to induce absolute and relative event positions for our Transformer such as:

- Event-type embedding $\mathbf{E} \in \mathbb{R}^{L \times d}$ is obtained like described in section 2.   
- Absolute time embedding $\mathbf{T} \in \mathbb{R}^{L \times d}$ is constructed on-the-fly at each forward pass by a linear transformation $t_{i,j} = t_i'w_j + b_j$ where $w_j, b_j$ are learnable parameters and $t_i'$ is the scaled time at event step $i$ .   
- Mileage embedding $\mathbf{M} \in \mathbb{R}^{L \times d}$ is obtained via a learnable lookup table $\mathbf{W}^{m_{\max} \times d}$ where $m_{\max} = 300\mathrm{km}$ . Each row $\mathbf{w}_m \in \mathbb{R}^d$ corresponds to the learnable embedding vector for the discrete mileage $m$ . The continuous mileage $m_i \in \mathbb{R}^+$ is cast to an integer value $m = \lfloor m_i \rfloor$ .   
- Rotary Position Embedding (RoPE) $\mathbf{R}_{\Theta}^{d}$ Due to the permutation invariance of the Transformer model and the scattered time $t$ , we still need to integrate positional event information. To do so $\mathbf{Q}, \mathbf{K}$ are rotated using the orthogonal matrix $\mathbf{R}_{\Theta}^{d}$ from (Su et al. 2024) in function of the absolute event position $i$ in the sequence $S$ . This method has two advantages: (1) it's not learnable (less likely to over-fitting), and (2) it integrates natively the relative position instead of altering $\mathbf{A}$ with a learnable bias like in (Shaw, Uszkoreit, and Vaswani 2018).

We make the distinction between our event-type embedding E and the other information per event (time and mileage) which we call context embedding $CE = T + M$ .

Continuous Time Mileage Aware Attention We modify the vanilla Transformer from (Vaswani et al. 2017) by (1) adding the context embedding to the projected event-type embedding at every layer (Touvron et al. 2023), and then (2) a Rotary Position Embedding (RoPE) (Su et al. 2024) is applied to both query Q and key K:

$$
\mathbf {Q} = \mathbf {R} _ {\Theta} ^ {d} (\mathbf {W} ^ {Q} \mathbf {E} + \mathbf {C E}),
$$

$$
\mathbf {K} = \mathbf {R} _ {\Theta} ^ {d} (\mathbf {W} ^ {K} \mathbf {E} + \mathbf {C E})
$$

![](images/2d33eb8b5403bb7c01898c63ea1f724a739e7446f5e9f3cc938596436631b953.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Next Event Head"] --> B["RMS Norm"]
    C["Next Time Head"] --> B
    D["Random Event Head"] --> B
    B --> E["Feed Forward"]
    E --> F["RMS Norm"]
    F --> G["Masked Multi Head Attention"]
    G --> H["Rotary Position Embedding"]
    G --> I["Key"]
    G --> J["Value"]
    H --> K["RMS Norm"]
    I --> K
    J --> K
    K --> L["Context Embedding"]
    L --> M["Event-type Embedding"]
    M --> N["Time"]
    N --> O["+"]
    O --> P["+"]
    P --> Q["N x"]
    Q --> R["H(S)"]
    R --> S["+"]
    S --> T["+"]
    T --> U["+"]
    U --> V["+"]
    V --> W["+"]
    W --> X["+"]
    X --> Y["+"]
    Y --> Z["+"]
    Z --> AA["+"]
    AA --> AB["+"]
    AB --> AC["+"]
    AC --> AD["+"]
    AD --> AE["+"]
    AE --> AF["+"]
    AF --> AG["+"]
    AG --> AH["+"]
    AH --> AI["+"]
    AI --> AJ["+"]
    AJ --> AK["+"]
    AK --> AL["+"]
    AL --> AM["+"]
    AM --> AN["+"]
    AN --> AO["+"]
    AO --> AP["+"]
    AP --> AQ["+"]
    AQ --> AR["+"]
    AR --> AS["+"]
    AS --> AT["+"]
    AT --> AU["+"]
    AU --> AV["+"]
    AV --> AW["+"]
    AW --> AX["+"]
    AX --> AY["+"]
    AY --> AZ["+"]
    AZ --> BA["+"]
    BA --> BB["+"]
    BB --> BC["+"]
    BC --> BD["+"]
    BD --> BE["+"]
    BE --> BF["+"]
    BF --> BG["+"]
    BG --> BH["+"]
    BH --> BI["+"]
    BI --> BJ["+"]
    BJ --> BK["+"]
    BK --> BL["+"]
    BL --> BM["+"]
    BM --> BN["+"]
    BN --> BO["+"]
    BO --> BP["+"]
    BP --> BQ["+"]
    BQ --> BR["+"]
    BR --> BS["+"]
    BS --> BT["+"]
    BT --> BU["+"]
    BU --> BV["+"]
    BV --> BW["+"]
    BW --> BX["+"]
    BX --> BY["+"]
    BY --> BZ["+"]
```
</details>

Figure 3: CarFormer architecture

where $\Theta=\{\theta_{i}=\theta_{0}^{-2(i-1)/d},i\in[1,2,\ldots,d/2]\},\theta_{0}=10^{4}$ . More specifically, the inner product between query $q_{m}$ and key $k_{n}$ takes the event-type embedding $e_{m},e_{n}$ where m-n is their relative position with context $CE_{m}$ and $CE_{n}$ :

$$
\begin{array}{l} \mathbf {q} _ {m} ^ {T} \mathbf {k} _ {n} = (\mathbf {R} _ {\Theta , m} ^ {d} (\mathbf {W} _ {q} \mathbf {e} _ {m} + \mathbf {C E} _ {m})) ^ {T} \mathbf {R} _ {\Theta , n} ^ {d} (\mathbf {W} _ {k} \mathbf {e} _ {n} + \mathbf {C E} _ {n}) \\ = \mathbf {e} _ {m} ^ {T} \mathbf {W} _ {q} \mathbf {R} _ {\Theta , n - m} ^ {d} \mathbf {W} _ {k} \mathbf {e} _ {n} + \mathbf {e} _ {m} ^ {T} \mathbf {W} _ {q} \mathbf {R} _ {\Theta , n - m} ^ {d} \mathbf {C E} _ {n} + \\ \mathbf {C E} _ {m} ^ {T} \mathbf {R} _ {\Theta , n - m} ^ {d} \mathbf {W} _ {k} \mathbf {e} _ {n} + \mathbf {C E} _ {m} ^ {T} \mathbf {R} _ {\Theta , n - m} ^ {d} \mathbf {C E} _ {n} \\ = (1): \text { query - to - key } + (2): \text { query - to - ce } \\ (3): \text { ce - to - key } + (4): \text { ce - to - ce } \tag {8} \\ \end{array}
$$

where $\mathbf{R}_{\Theta,n-m}^{d} = (\mathbf{R}_{\Theta,m}^{d})^{T}\mathbf{R}_{\Theta,n}^{d}$ is a sparse orthogonal matrix. The additional terms (2), (3), (4) provide richer query and key representations when computing the attention scores. Thus modifying Equation 4:

$$
\mathbf {A} = \operatorname{softmax} \left(\frac {\left(\mathbf {R} _ {\Theta} ^ {d} \left(\mathbf {W} ^ {Q} \mathbf {E} + \mathbf {C E}\right)\right) \left(\mathbf {R} _ {\Theta} ^ {d} \left(\mathbf {W} ^ {K} \mathbf {E} + \mathbf {C E}\right)\right) ^ {T}}{\sqrt {3 d}}\right)
$$

Adding CE after the projection to query and key can be seen as a refinement of Q, K by T, M, providing additional context to the attention scores. We also add a scaling factor of 3 to compensate for the additional terms in Equation 8.

# Multi-task Learning

Next Event Prediction. We use a standard language modelling objective which aims to minimize the cross-entropy loss between our output distribution $\hat{u}_{i}$ generated by our model's Next Event Head and the next event $u_{i+1}$ . (Shou et al. 2024) used a BERT (Devlin et al. 2019) model trained on a masked event modelling task which is commonly used for bidirectional models. However, they simultaneously applied a causal mask resulting in a loss of the bidirectional property. We argue that by doing so, we lose a lot of sample efficiency, thus we will stick to a standard next token prediction. $\hat{u}_{i} \in R^{V}$ is the predicted probability distribution by

the Next Event Head, which integrates an RMS normalization (Zhang and Sennrich 2019) and one linear layer. The cross-entropy loss between $\hat{u}_{i}$ and $u_{i+1}$ (= a one-hot vector in $\{0,1\}^{V}$ ) is obtained by:

$$
\mathcal {L} _ {c} := - \sum_ {i = 0} ^ {L} \sum_ {j = 0} ^ {V} u _ {i + 1, j} \log (\hat {u} _ {i, j}) (1 - \delta_ {i, r}) \tag {9}
$$

where $\delta_{i,r}$ is the Kronecker delta, which equals 1 when $i = r, r \in R$ (set of randomly generated events) and 0 otherwise.

Next Event Time Prediction. In addition, we compute the Huber loss (Jadon, Patil, and Jadon 2024) between the estimated inter-event time $\Delta \hat{t}'_i$ for the event $u_i$ and the ground truth $\Delta t_i' = f_t(t_{i+1}, 10) - f_t(t_i, 10)$ to deal with outliers and prevent exploding gradients with $\beta = 1$ , $\epsilon_i = \Delta \hat{t}_i' - \Delta t_i'$ .

$$
\mathcal {L} _ {t} := \sum_ {i = 0} ^ {L} (1 - \delta_ {i, r}) \left\{ \begin{array}{l l} 0. 5 \epsilon_ {i} ^ {2} & \text { if } | \epsilon_ {i} | <   \beta , \\ (| \epsilon_ {i} | - 0. 5) & \text { otherwise }, \end{array} \right. \tag {10}
$$

$t'$ is obtained using a log, hence we are essentially computing a kind of Mean Squared Logarithmic Error (MSLE) but with a $\beta$ , useful to stabilize training and help convergence.

Random Event Prediction. Finally, a binary classifier that predicts whether an event was true or randomly generated is added. We motivate this choice by several papers (Shou et al. 2024; Gao et al. 2020) stating that a model should learn when an event does not happen to reinforce the negative evidence of no observable events within each inter-event. At each step i, a random event is injected with probability p. If a random event is successfully injected, the process continues until a failure occurs (i.e., the event is not injected), this allows for multiple random events to be injected in a row, allowing for more complexity. A detailed explanation of the injection process can be found in Algorithm 1 and Appendix A. The $L_{r}$ loss is defined as the binary cross-entropy loss between the probability distribution $\hat{y}_{i}^{r}$ generated by our Random Event Head and the ground truth $y_{i}^{r}$ at event step i:

$$
\mathcal {L} _ {r} := - \sum_ {i = 0} ^ {| R |} y _ {i} ^ {r} \log (\hat {y} _ {i} ^ {r}) + (1 - y _ {i} ^ {r}) \log (1 - \hat {y} _ {i} ^ {r}) \tag {11}
$$

Total Loss. The total loss is defined as follows:

$$
\mathcal {L} = \frac {1}{L - | R |} (\mathcal {L} _ {c} + \alpha \mathcal {L} _ {t}) + \beta \frac {1}{| R |} \mathcal {L} _ {r} \tag {12}
$$

where $\alpha,\beta$ are trade-off between the different loss, L the sequence length, R the set of random events injected in S.

# 5 EPredictor

Predicting only the next DTC in a sequence of DTC faults has its inherent limitation and remains a difficult task. (Hafeez, Alonso, and Riaz 2024) use DTCs those ECU, Base-DTC and Fault-Byte data have a cardinality of 83, 419 and 64, respectively, and only report a 81% top-5 accuracy for next DTC prediction. This is because DTCs are not always correlated nor have causal links. Instead, we are also using repair and warranty data to predict more important events such as EPs (error patterns). Repair and warranty data differ from DTCs since they are manually defined by domain experts after observing all DTCs and characterize a whole sequence S and not an individual event $u_{i}$ . Hence, we can define and say that a certain error pattern y has happened at index i = L in a sequence S. Note, that multiple EPs can occur at the same time. We can now define a supervised multi-label classification learning problem of predicting EPs. With EPredictor, we leverage the seq2seq nature of Transformers, where CarFormer outputs a sequence $H(S)$ of tokens encoded in a high-dimensional space d, positioning tokens with similar characteristics nearby. Then, this hidden representation $H(S)$ is fed into EPredictor which acts as an autoregressive multi-label classifier for EPs. We approach EP prediction as a machine translation task. By utilizing the contextualized hidden states $H(S)$ from CarFormer as key and value (i.e., through cross-attention), this effectively transitions our model from a "seq2seq" to a "dtc2errorpattern" framework.

![](images/6c1e17df47e3ea03f1df71d90f8655f303931de610a39cb74c3063e56287b662.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Minimum context c"] --> B["Next EP Head"]
    A --> C["Next EP Time Head"]
    B --> D["Feed Forward"]
    C --> D
    D --> E["Add & RMS Norm"]
    E --> F["Masked Multi Head Attention"]
    F --> G["Output Context Embedding"]
    F --> H["Output H(S) CarFormer"]
    F --> I["Output Event-type Embedding"]
    G --> J["Feed Forward"]
    H --> J
    I --> J
    J --> K["Add & RMS Norm"]
    K --> L["Masked Multi Head Attention"]
    L --> M["Rotary Position Embedding"]
    L --> N["RMS Norm"]
    M --> O["Query"]
    N --> P["Key"]
    O --> Q["Value"]
    P --> R["Value"]
    Q --> S["+"]
    R --> T["+"]
    S --> U["Time"]
    T --> V["Time"]
    U --> W["P(EP|S')"]
    V --> X["P(EP|S'')"]
    W --> Y["..."]
    X --> Z["..."]
```
</details>

Figure 4: EPredictor architecture

# Multi-Label Event Prediction

Multi-label classification has recently gained interest in event prediction. (Zhang et al. 2020b) uses an LSTM for fault detection. More recently, (Shou et al. 2023) considers concurrent event predictions as a multi-label classification and models such data with a Transformer architecture.

To define the multi-label event prediction task with N labels ( $\equiv$ EPs), we reuse each $S = \{(u_i, t_i, m_i)\}_{i=0}^L$ and attach a binary vector $y \in [0, 1]^N$ to indicate the EPs occurring at time $t_L$ . It's important to note that y is invariant per sequence S, meaning for all events within S the

ground truth y will be the same. By using a causal mask on both CarFormer and EPredictor, we enable predictive maintenance predictions since tokens can only attend to the previous ones, as shown in Figure 4. In our case, and in most real-world problems, EPs are highly imbalanced across our dataset, which is a considerably more challenging problem for our event prediction task, especially when the least occurring classes are the most important to detect (Zhang et al. 2020b). In natural language processing, traditional upsampling methods involve perturbation of S by shuffling and replacement of tokens. However, in event data, we cannot afford to lose spatial and temporal information. Therefore, we inject random events $(u_{i}, m_{i}, t_{i})$ with the same probability of p = 0.05 as in Section 4 and reuse the associated Algorithm 1. We up-sample the different EPs classes up to a minimum $\theta_{1} = 6000$ and downsample the most popular ones down to a maximum $\theta_{2} = 12000$ . We drop also classes below 100 apparitions across the dataset. We define a minimum context c = 30 which acts as a “minimum history” of DTCs to retain. The Next EP Head output a vector of probabilities $\hat{y}_{i} = \text{sigmoid}(\text{MLP}_{c}(H_{i}))$ for each history $\{H_{1}, ..., H_{i}\}$ , $i \in c, ..., L$ where $H_{i} \in R^{d}$ is the generated hidden representation from EPredictor at step i. For the regression task we forecast the time till the EP(s) occurrence $\Delta\hat{t}'_{i} = \text{MLP}_{t}(H_{i})$ where the ground truth is $\Delta t'_{i} = f_{t}(t_{L}, 30) - f_{t}(t_{i}, 30)$ . Formally, we define our binary cross-entropy loss over the N possible EPs for one step i as follows:

$$
\mathcal {L} _ {i} ^ {e p} := - \frac {1}{N} \sum_ {j = 0} ^ {N} y _ {j} \log (\hat {y} _ {i, j}) + (1 - y _ {j}) \log (1 - \hat {y} _ {i, j}) \tag {13}
$$

The total loss across $S$ with $L$ events and context $c$ is:

$$
\mathcal {L} ^ {e p} := \frac {1}{L - c} \sum_ {i = c} ^ {L} \mathcal {L} _ {i} ^ {e p} \tag {14}
$$

We use the Huber loss (Jadon, Patil, and Jadon 2024) and define $\epsilon_{i} = \Delta \hat{t}_{i}^{\prime} - \Delta t_{i}^{\prime}$ , with $\beta = 1$ thus:

$$
\mathcal {L} ^ {t} := \frac {1}{L - c} \sum_ {i = c} ^ {L} \left\{ \begin{array}{l l} 0. 5 \epsilon_ {i} ^ {2} & \text { if } \epsilon_ {i} <   \beta , \\ | \epsilon_ {i} | - 0. 5 & \text { otherwise }, \end{array} \right. \tag {15}
$$

Our final loss to minimize is then: $\mathcal{L} = \mathcal{L}^{ep} + \gamma \mathcal{L}^t$

# 6 Experiments

We implemented and trained CarFormer and EPredictor models using PyTorch, the code is publicly available. Training details are included in Appendix A,B.

# Ablation I: CarFormer Embeddings

Multiple CarFormer models with different choices of embeddings were evaluated on the next token prediction accuracy (ACC) and on the regression task with mean absolute percentage error (MAPE) and root mean square error (RMSE) to determine the best working CarFormer model. The different choices were rot (RoPE), time (absolute time embedding added to the input U), mileage (also added to U), m2c, and c2m (Appendix A). In our case, we are more interested in the next event prediction task, thus we accept to lose some MAPE % over the ACC (%).

<table><tr><td>Model</td><td>ACC(%)</td><td>MAPE(%)</td><td>RMSE</td></tr><tr><td>rot-ce</td><td>22.64</td><td>3.2</td><td>0.04770</td></tr><tr><td>time</td><td>21.48</td><td>2.9</td><td>0.04762</td></tr><tr><td>time-mileage</td><td>21.38</td><td>3.0</td><td>0.04785</td></tr><tr><td>time-c2m-m2c</td><td>21.58</td><td>3.5</td><td>0.04794</td></tr><tr><td>time-m2c</td><td>21.52</td><td>3.6</td><td>0.04823</td></tr><tr><td>GPT</td><td>19.89</td><td>-</td><td>-</td></tr></table>

Table 3: Overall prediction performance of CarFormer with different embeddings. Best results are in bold.

Using only time gave the best MAPE (2.9) but not the best ACC (21.48), suggesting that other features might improve the model predictions. For the mileage integration, our intuition was that doing an early summation of the two embeddings (T, M) seemed to denature the input U (ACC of time-mileage < ACC of time) but the mileage of the vehicle could help differentiate between different DTCs. We tried to modify the attention dot products which seemed to help the ACC a bit (time-c2m-m2c, time-m2c) but increased the MAPE drastically. So we fused it with CE directly in Q, K. Then, by applying a RoPE to the transformed input, the ACC increased while preserving the RMSE, leading to the best performing model, namely: rot-ce.

# EPredictor Experiments

We used the micro-F1 score (Zhang and Zhou 2014) to assess the performance of the multi-label classification. To better understand and enhance our model's predictive maintenance capabilities, we introduce the concept of Confident Predictive Maintenance Window (CPMW) which represents the interval within which our model can make reliable predictive maintenance predictions (similar to the "prediction window" described in (Pirasteh et al. 2019)). We quantify this with the CPMW Area Under Curve (CPMWAUC). The F1 score, MAE and MAPE have been calculated on average for all observations in Table 4, and additionally for each history $H_t^i$ in Figure 4 to understand how each model performs with different numbers of observations. To monitor the predictive maintenance capability of each model, the CPMWAUC $_{\text{f1}}$ and CPMWAUC $_{\text{mae}}$ were computed. The Appendix B contains the models and metrics definition.

# Ablation II: EPredictor Architecture & CPMW

We explored several architectural changes and their impact on the CPMW: we applied a RoPE (rot), a cross attention with query or key or value to the second multi-head attention block (cross), added the context embedding CE (ce) to layer 1 and/or 2, applied a scaling factor of $\sqrt{3d}$ to Q, K as shown in Equation 4 (scale), injected a relative matrix $\mathbf{S}_{rel}$ into the attention scores (speed), and applied a mixed feed-forward network (mixffn) (Xie et al. 2021) to the mileage embedding. The time model refers to T which is also added to E. Finally, we trained a model with and without the Random

<table><tr><td>Model</td><td>Micro F1 (%)</td><td>MAPE (%)</td><td>MAE</td><td>CPMWAUC $_{f1}$  ↑</td><td>CPMWAUC $_{mae}$  ↓</td></tr><tr><td>rotcross-query-key-ce-1-2</td><td>82.69</td><td>32.45</td><td>0.0268</td><td>52.80</td><td>0.888</td></tr><tr><td>rotcross-query-key-ce-2</td><td>82.69</td><td>31.18</td><td>0.0254</td><td>56.61</td><td>0.882</td></tr><tr><td>rotcross-query-ce-2</td><td>82.69</td><td>31.18</td><td>0.0254</td><td>53.00</td><td>0.884</td></tr><tr><td>rotcross-key-value-ce-2</td><td>84.38</td><td>33.47</td><td>0.0263</td><td>65.06</td><td>0.904</td></tr><tr><td>rotcross-key-value-scaled-ce-2</td><td>84.38</td><td>31.44</td><td>0.0252</td><td>67.63</td><td>0.874</td></tr><tr><td>rotnocross-ce-1-2</td><td>80.74</td><td>37.61</td><td>0.0275</td><td>42.95</td><td>0.927</td></tr><tr><td>cross-speed</td><td>83.53</td><td>34.73</td><td>0.0260</td><td>49.28</td><td>0.877</td></tr><tr><td>cross-mixffn</td><td>83.41</td><td>33.37</td><td>0.0270</td><td>54.39</td><td>0.891</td></tr><tr><td>time-cross-query</td><td>83.34</td><td>35.89</td><td>0.0275</td><td>45.71</td><td>0.896</td></tr></table>

Table 4: EPredictor evaluation results with different model architecture on the test set (no up- nor down-sampling)

Event Head. Our experiments revealed several key insights: By applying a cross attention, we can see improvement in all metrics (rotnocross-ce-1-2), which is consistent with the machine translation analogy "dtc2errorpattern". The best cross attention results were shown with $H(S)$ used as key-value. Adding the mileage via an MLP layer (cross-mixffn) seemed to help the MAPE (-2.5%), the CPMWAUC $_{f1}$ (+9) and also the CPMWAUC $_{mae}$ (-0.005) compared to time-cross-query model, suggesting that mileage is beneficial for both task. This makes sense since EPs are also dependent on the traveled distances between DTCs and the different stationary behavior of the vehicle. Furthermore, models incorporating a RoPE (rot) performed significantly better in both regression and classification tasks like in the pre-training, highlighting the performance of RoPE in machine translation tasks (Su et al. 2024). Adding CE to the last layer (ce-2) gave the best results, as opposed to adding it to both layers (ce-1-2). Surprisingly, doing feature engineering on T, M with a speed matrix S $_{rel}$ didn't help the metrics, which could indicate some missing modalities (e.g. mileage) during training, thus CE is more adapted for real-world scenarios. We monitored the need for our Random Event Head in Figure 7 (Appendix B) and noticed +1.2% in the F1 Score and +10.1 in the CPMW $_{f1}$ . When taking the best performing model (rotcross-key-value-scaled-ce-2), the model entered the CPMW after 81 observations i.e. half of the sequence. Within this window, the model obtained an error of ≈ 65 ± 14h when estimating the time of EP occurrence (Figure 6), highlighting the model's predictive capability within the CPMW. Otherwise, the average absolute error across all observations was approximately 58.4 ± 13.2h. By experimenting with these modifications, we aimed to identify the optimal architecture for predictive maintenance. The findings reveal that cross attention, context embedding in the second layer, and scaled attention significantly improve performance within the CPMW.

# 7 Conclusion

This study bridges the gap between traditional event sequence modeling and fault event prediction in vehicles by using two language models. We have demonstrated through specific relevant metrics that our approach can accurately perform predictive maintenance by effectively predicting when and what error patterns are likely to occur, even with

![](images/a8ecfa7b0bb3d7ffa42f912319517cc17849fc276355ea9c7a614d4bfdb8f01a.jpg)

<details>
<summary>line</summary>

| Number of observations | F1 micro (%) - c = rotcross-key-value-ce-2 | F1 micro (%) - c = rotcross-key-value-scaled-ce-2 | F1 micro (%) - c = rotcross-query-key-ce-2 | F1 micro (%) - c = time-cross-query | F1 micro (%) - c = cross-mixfin | F1 micro (%) - c = cross-speed | F1 micro (%) - c = rotnocross-ce-1-2 | F1 micro (%) - x = 81, Confident predictions | F1 micro (%) - x = 158, Average Sequence length |
| ---------------------- | ---------------------------------------- | ----------------------------------------------- | ----------------------------------------- | ---------------------------------- | ------------------------------ | ----------------------------- | ----------------------------------- | ------------------------------------------ | -------------------------------------------- |
| 20                     | 50                                       | 50                                              | 50                                        | 50                                 | 50                             | 50                          | 50                                  | 50                                         | 50                                           |
| 40                     | 60                                       | 60                                              | 60                                        | 60                                 | 60                             | 60                          | 60                                  | 60                                         | 60                                           |
| 60                     | 70                                       | 70                                              | 70                                        | 70                                 | 70                             | 70                          | 70                                  | 70                                         | 70                                           |
| 80                     | 80                                       | 80                                              | 80                                        | 80                                 | 80                             | 80                          | 80                                  | 80                                         | 80                                           |
| 100                    | 85                                       | 85                                              | 85                                        | 85                                 | 85                             | 85                          | 85                                  | 85                                         | 85                                           |
| 120                    | 87                                       | 87                                              | 87                                        | 87                                 | 87                             | 87                          | 87                                  | 87                                         | 87                                           |
| 140                    | 88                                       | 88                                              | 88                                        | 88                                 | 88                             | 88                          | 88                                  | 88                                         | 88                                           |
| 160                    | 89                                       | 89                                              | 89                                        | 89                                 | 89                             | 89                          | 89                                  | 89                                         | 89                                           |
| 180                    | 90                                       | 90                                              | 90                                        | 90                                 | 90                             | 90                          | 90                                  | 90                                         | 90                                           |
| 200                    | 90                                       | 90                                              | 90                                        | 90                                 | 90                             | 90                          | 90                                  | 90                                         | 90                                           |
</details>

Figure 5: F1 Score comparison with multiple Epredictor architectures in function of the number of observations.

![](images/254863d7c3d8b192888b2d228ee614d2d72d75e975e841275d4f2a407cd12d81.jpg)

<details>
<summary>line</summary>

| Number Of Observations | Mean Absolute Error (h) |
| ---------------------- | ------------------------ |
| 40                     | ~130                     |
| 80                     | ~95                      |
| 120                    | ~70                      |
| 160                    | ~50                      |
| 200                    | ~35                      |
</details>

Figure 6: Evolution of the MAE in function of the number of observations for the best performing model.

continuous, unbalanced, and high cardinality data. In real-world settings, EPredictor is easy to use in a car. After each DTC occurrence and until reaching a minimum number of observations is reached, we would then infer the most likely EP and its time of occurrence. If the model is confident enough, the user will be alerted to an impending critical fault and directed to a nearby dealer, hence enhancing vehicle safety on the road and reducing maintenance costs.

# A CarFormer Pre-training Details

We trained the CarFormer model for 70,000 steps with a learning rate of $5 \times 10^{-4}$ , scheduled using a cosine warm restart with 10,000 warm-up steps, and a weight decay of

0.1. The loss coefficients were set to $\alpha = 1$ and $\beta = 1$ . The model architecture included 12 attention heads, 6 layers, and a feature size of 600, utilizing the GELU activation function in all feed-forward layers. We employed the AdamW optimizer with a batch size of 192 and a sequence length of 258, resulting in approximately 34 million parameters. It is important to note that on all experiments we fixed the same number of parameters when introducing news embeddings for all evaluated model to provide a fair comparison, same for Epredictor. The training data was split into 85% training and 15% testing without up-sampling, and random events were injected with a probability of p = 0.05 per sample. On average, each training session lasted about 20 hours on an Nvidia A10G GPU.

# A.1 Models Definition

A GPT (Radford and Narasimhan 2018) model is added to improve comparison over baseline models. We kept the original implementation of the feed-forward layers (Vaswani et al. 2017) but initialize the intermediate layers with: $W_{l} \sim \mathcal{N}(0, \frac{2}{L\sqrt{d_l}})$ . We used root-mean-square normalization (referred to as RMS Norm) from (Zhang and Sennrich 2019) instead of traditional layer normalization and initialize all other linear layers using the SMALLINIT schema (Nguyen and Salazar 2019) $W_{l} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_l + 4d_l}}\right)$ . The names printed in the Tab 3 are model variations based on the embeddings given to CarFormer:

time : Along with the event type E an absolute time embedding is added such as input $U = E + T$

mileage : Same as time except we add a mileage embedding
M such as U = E + M

rot : A RoPE (Su et al. 2024) is applied to Q, K such as $Q = R_{\Theta}^{d} W^{Q} E$ , $K = R_{\Theta}^{d} W^{Q} E$

ce : We add a context embedding $CE = T + M$ to Q, K such as $Q = UW^{Q} + CE$ to every attention layers.

m2c, c2m : Inspired by the Disentangled Attention mechanism from DeBerta (He et al. 2021). Attention score  \( A\_{i,j} \)  between tokens i and j is computed from hidden states vector  \( \{H\_{i}\} \)  at event step i and a mileage vector  \( \{M\_{i}\} \)  at event step i such as:  \( A\_{i,j} = \{H\_{i}, M\_{i}\} \times \{H\_{j}, M\_{j}\}^{T} = H\_{i}H\_{j}^{T} + H\_{i}M\_{j}^{T} + M\_{i}H\_{j}^{T} + M\_{i}M\_{j}^{T} = "content-to-content" + "content-to-mileage" + "mileage-to-content" + "mileage-to-mileage" = "c2c" + "c2m" + "m2c" + "m2m".

# A.2 Random Event Injection

Let $A_{i}$ be the random variable representing the number of random events injected at step i. We can say that:

$$
P (A _ {i} = r) = (1 - p) p ^ {r} \quad \text { for } \quad r = 0, 1, 2, \ldots
$$

Where $A_{i}$ is the number of random events injected at step i, r is the number of trials (injected events) until the first failure, p is the probability of injecting a random event. Hence, it's trivial to derive the expected number of random events injected per sequence S of length L: Therefore, the expected

Algorithm 1: Random event injection algorithm   
Input: Given a dataset D with sequences S, each with length L, $S = \{(u_i, t_i, m_i)\}_{i=0}^L$ Parameter: p injection probability

Output: Dataset D'

1: Let $D' = []$ .

2: for $d \leftarrow 1$ to D do

3:    S' = []

4:    S = D[d]

5:    for i $\leftarrow 1$ to L - 2 do

6:    S'.append(S[i])

7:    while ( $p \leq random.float(0, 1)$ ) & (i < (L - 2))

    do

8: $t'_i \sim Unif(t_i, t_{i+1})$ 9: $m'_i \sim Unif(m_i, m_{i+1})$ 10: $u'_i \sim Unif\{u_1, u_2, \ldots, u_n\}$ 11:    S'.append(( $u'_i, m'_i, t'_i$ ))

12:    end while

13:    if $len(S') == (L - 2)$ then

14:    break

15:    end if

16:    end for

17:    D'.append(S')

18: end for

19: return D'

![](images/46382c1f226be5701087de410d533be825886cf77dca9666e4ff3e812b9d8fc0.jpg)

<details>
<summary>line</summary>

| Number of observations | F1 micro (%) - c = w/o random head | F1 micro (%) - c = w random head |
| ---------------------- | ----------------------------------- | --------------------------------- |
| 0                      | 65                                  | 65                                |
| 50                     | 70                                  | 70                                |
| 75                     | 78                                  | 78                                |
| 100                    | 82                                  | 83                                |
| 125                    | 83                                  | 84                                |
| 150                    | 84                                  | 84                                |
| 175                    | 84                                  | 84                                |
| 200                    | 84                                  | 84                                |
</details>

Figure 7: EPredictor CPMW with and w/o the random head.

total number of injected fake events over the entire sequence of length $L$ is:

$$
\mathbb {E} \left[ \sum_ {i = 1} ^ {L} A _ {i} \right] = L p
$$

Which in our case would translate $\sim 150 \times 0.05 = 7.5$ random events per sequence.

# B EPredictor Training Details

We trained the EPredictor model for 10 epochs, using a patience of 2 on the evaluation loss to prevent overfitting. We used an AdamW optimizer with a learning rate of $10^{-4}$ , scheduled by a cosine warm restart. The model employed the GELU activation function in all feed-forward layers, with a batch size of 384 and $\gamma = 1$ for the regression loss. The model architecture comprised attention layers with 12 attention heads each, and 2 feed-forward networks (FFN) with

1200 hidden neurons ( $d_{model} \times 2$ ) each. The total number of trainable parameters (excluding the backbone) was 6.5 million per EPredictor. Each training session lasted about 6 hours on an Nvidia A10G GPU. We set $c = 30$ . The dataset was split into 70%, 15%, and 15% for training, validation, and testing, respectively. The test set consists of 45,000 labeled sequences, with 269 unique error patterns (EP). A confidence threshold of 0.7 is used for classification. For the regression task, when referring to MAE (Mean Absolute Error) or MAPE (Mean Absolute Percentage Error), we calculate the time $t'$ using the function $f_t(t,b)$ . The MAE is defined as:

$$
\mathrm{MAE} = \frac {1}{N} \sum_ {i = 1} ^ {L ^ {\prime}} | \Delta \hat {t ^ {\prime}} _ {i} - \Delta t _ {i} ^ {\prime} |
$$

In Figure 6, the MAE (h) is calculated using the inverse function $f_{t}^{-1} : \mathbb{R} \to \mathbb{R}^{+}$ , defined as $t = f_{t}^{-1}(t', b) = b^{t' + 1} - 1\forall t' \in \mathbb{R}$ . Thus, the MAE (h) is:

$$
\mathrm{MAE} _ {h} = \frac {1}{N} \sum_ {i = 1} ^ {L ^ {\prime}} | f _ {t} ^ {- 1} (\Delta \hat {t ^ {\prime}} _ {i}, 3 0) - f _ {t} ^ {- 1} (\Delta t _ {i} ^ {\prime}, 3 0) |
$$

For EPredictor, $L' = L - c$ and i starts at c, whereas for CarFormer, $L' = L$ and i = 0.

# B.1 Architecture choice

We defined multiple models for comparison with different architectures:

mixffn : Adds mileage to the first attention layer through a mix feed-forward network (MLP and 3x3 Conv) following (Xie et al. 2021). The formula is: $U' = E + MLP(GELU(Conv_{3x3}(MLP(M)))$ where E is the output event embedding from CarFormer, and $U' \in R^{L \times d}$ is the input for the first attention layer of EPredictor.

rot : Applies RoPE (Rotary Position Embeddings) to query (Q) and key (K) matrices as described in Section 4.

speed : Adds a relative speed matrix to the attention scores. Defines matrices $T_{r}$ and $M_{r} \in R^{L}$ containing time and mileage features, forming $S = \frac{M}{T}$ and $S_{rel} = S - S^{T} \in R^{L \times L}$ . The attention scores are modified as $\mathbf{A} = \text{softmax}((\mathbf{Q}\mathbf{K}^{T} + \mathbf{S}_{rel}) / \sqrt{2d})$ . This model explores if incorporating relative speed information improves predictive performance.

Note: Some models such as rotcross-query-key-ce-1-2 are not shown in the figures to improve visual clarity

# B.2 Context Size

Ablation III: Context Size We explored the necessity of c to provide EPredictor minimal observations before generating any predictions. Intuitively, with just 1 or 2 DTCs in S, it is not possible to predict future EPs. We experimented with several contexts $c = [0, 10, 20, 30, 40]$ on the test set. Our experiments revealed +2% improvement in the F1 score using a context c = 30. However, it is not trivial, as depicted in Figure 8, c = 10, 20, 40 are worse than no c at all. We found that c = 30 is beneficial for the CPMWAUC $_{f1}$ but should be used with caution and tested empirically. This indicates that increasing the context size might not always enhance the model's predictive performance and depends surely on the data distribution.

![](images/cf06bf95b0dcfd38c2391de17c2516679bb4e46e5efb637a5898db1df7ad900d.jpg)

<details>
<summary>line</summary>

| Number of observations | c = 0 | c = 10 | c = 20 | c = 30 | c = 40 |
| ---------------------- | ----- | ------ | ------ | ------ | ------ |
| 0                      | 20    | 20     | 20     | 20     | 20     |
| 25                     | 40    | 40     | 40     | 40     | 40     |
| 50                     | 60    | 60     | 60     | 60     | 60     |
| 75                     | 75    | 75     | 75     | 75     | 75     |
| 100                    | 80    | 80     | 80     | 80     | 80     |
| 125                    | 82    | 82     | 82     | 82     | 82     |
| 150                    | 83    | 83     | 83     | 83     | 83     |
| 175                    | 83    | 83     | 83     | 83     | 83     |
| 200                    | 83    | 83     | 83     | 83     | 83     |
</details>

Figure 8: Evolution of the F1 Score with multiple c in function of the number of observations.

# B.3 Confident Predictive Maintenance Window

The Confident Predictive Maintenance Window (CPMW) can be defined as the interval $[x_{\theta}, \mu_{seq} + \delta]$ where our predictive maintenance model is making prediction confidently on a set of sequences $S = \{(u_i, t_i, m_i)\}_{i=0}^L$ .

- $\theta$ is the threshold evaluation score (e.g., $80\%$ F1 score).   
- $\mu_{\mathrm{seq}}$ is the average sequence length.   
- $\delta$ is the adjustment for noise (e.g., 8 for $5\%$ random events).

Confidently means at a typically chosen "high" or "low" score $\theta = \zeta(x_{\theta})$ with $\zeta$ a metric function. The CPMW Area Under the Curve or CPMWAUC of a given metric function $\zeta$ is given by its integral within the CPMW. where $\zeta(x)$ continuous and integrable metric function.

# References

Bachmann, G.; and Nagarajan, V. 2024. The Pitfalls of Next-Token Prediction. In ICLR 2024 Workshop: How Far Are We From AGI.

Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and Lin, H., eds., Advances in Neural Information Processing Systems, volume 33, 1877–1901. Curran Associates, Inc.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In North American Chapter of the Association for Computational Linguistics.

Du, N.; Dai, H.; Trivedi, R.; Upadhyay, U.; Gomez-Rodriguez, M.; and Song, L. 2016. Recurrent Marked Temporal Point Processes: Embedding Event History to Vector. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, 1555–1564. New York, NY, USA: Association for Computing Machinery. ISBN 9781450342322.   
Gao, T.; Subramanian, D.; Shanmugam, K.; Bhattacharjya, D.; and Mattei, N. 2020. A Multi-Channel Neural Graphical Event Model with Negative Evidence. Proceedings of the AAAI Conference on Artificial Intelligence, 34: 3946–3953.   
Hafeez, A. B.; Alonso, E.; and Riaz, A. 2024. DTC-TranGru: Improving the performance of the next-DTC Prediction Model with Transformer and GRU. Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing.   
Hafeez, A. B.; Alonso, E.; and Ter-Sarkisov, A. 2021. Towards Sequential Multivariate Fault Prediction for Vehicular Predictive Maintenance. In 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA), 1016–1021.   
Hawkes, A. G. 1971. Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58(1): 83–90.   
He, P.; Liu, X.; Gao, J.; and Chen, W. 2021. DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In International Conference on Learning Representations.   
Jadon, A.; Patil, A.; and Jadon, S. 2024. A Comprehensive Survey of Regression-Based Loss Functions for Time Series Forecasting. In Sharma, N.; Goje, A. C.; Chakrabarti, A.; and Bruckstein, A. M., eds., Data Management, Analytics and Innovation, 117–147. Singapore: Springer Nature Singapore. ISBN 978-981-97-3245-6.   
Lin, H.; Wu, L.; Zhao, G.; Pai, L.; and Li, S. Z. 2022. Exploring Generative Neural Temporal Point Process. Transactions on Machine Learning Research.   
Nguyen, T. Q.; and Salazar, J. 2019. Transformers without Tears: Improving the Normalization of Self-Attention. In Niehues, J.; Cattoni, R.; Stüker, S.; Negri, M.; Turchi, M.; Ha, T.-L.; Salesky, E.; Sanabria, R.; Barrault, L.; Specia, L.; and Federico, M., eds., Proceedings of the 16th International Conference on Spoken Language Translation. Hong Kong: Association for Computational Linguistics.   
Pirasteh, P.; Nowaczyk, S.; Pashami, S.; Löwenadler, M.; Thunberg, K.; Ydreskog, H.; and Berck, P. 2019. Interactive feature extraction for diagnostic trouble codes in predictive maintenance: A case study from automotive domain. In Proceedings of the Workshop on Interactive Data Mining, WIDM'19. New York, NY, USA: Association for Computing Machinery. ISBN 9781450362962.   
Radford, A.; and Narasimhan, K. 2018. Improving Language Understanding by Generative Pre-Training.   
Shaw, P.; Uszkoreit, J.; and Vaswani, A. 2018. Self-Attention with Relative Position Representations. In Walker, M.; Ji, H.; and Stent, A., eds., Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 464–468. New Orleans, Louisiana: Association for Computational Linguistics.

Shchur, O.; Türkmen, A. C.; Januschowski, T.; and Günnemann, S. 2021. Neural Temporal Point Processes: A Review. In Zhou, Z.-H., ed., Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, 4585–4593. International Joint Conferences on Artificial Intelligence Organization. Survey Track.   
Shou, X.; Gao, T.; Subramanian, D.; Bhattacharjya, D.; and Bennett, K. P. 2023. Concurrent Multi-Label Prediction in Event Streams. Proceedings of the AAAI Conference on Artificial Intelligence, 37(8): 9820–9828.   
Shou, X.; Subramanian, D.; Bhattacharjya, D.; Gao, T.; and Bennet, K. P. 2024. Self-Supervised Contrastive Pre-Training for Multivariate Point Processes. ArXiv, abs/2402.00987.   
Su, J.; Ahmed, M.; Lu, Y.; Pan, S.; Bo, W.; and Liu, Y. 2024. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomput., 568(C).   
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L. u.; and Polosukhin, I. 2017. Attention is All you Need. In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.; Vishwanathan, S.; and Garnett, R., eds., Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.   
Xie, E.; Wang, W.; Yu, Z.; Anandkumar, A.; Alvarez, J. M.; and Luo, P. 2021. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y.; Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems, volume 34, 12077–12090. Curran Associates, Inc.   
Zhang, B.; and Sennrich, R. 2019. Root Mean Square Layer Normalization. In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds., Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.   
Zhang, M.-L.; and Zhou, Z.-H. 2014. A Review On Multi-Label Learning Algorithms. Knowledge and Data Engineering, IEEE Transactions on, 26: 1819–1837.   
Zhang, Q.; Lipani, A.; Kirnap, O.; and Yilmaz, E. 2020a. Self-Attentive Hawkes Process. In III, H. D.; and Singh, A., eds., Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 11183–11193. PMLR.   
Zhang, W.; Jha, D. K.; Laftchiev, E.; and Nikovski, D. 2020b. Multi-label Prediction in Time Series Data using Deep Neural Networks. CoRR, abs/2001.10098.   
Zuo, S.; Jiang, H.; Li, Z.; Zhao, T.; and Zha, H. 2020. Transformer Hawkes Process. In III, H. D.; and Singh, A., eds., Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 11692–11702. PMLR.