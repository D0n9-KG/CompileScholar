# Group and Shuffle: Efficient Structured Orthogonal Parametrization

Mikhail Gorbunov

HSE University

gorbunovmikh73@gmail.com

Nikolay Yudin

HSE University

Vera Soboleva

AIRI\*

Aibek Alanov

AIRI $^{*}$ ,

HSE University

Alexey Naumov

HSE University,

Steklov Mathematical Institute RAS

Maxim Rakhuba

HSE University

# Abstract

The increasing size of neural networks has led to a growing demand for methods of efficient fine-tuning. Recently, an orthogonal fine-tuning paradigm was introduced that uses orthogonal matrices for adapting the weights of a pretrained model. In this paper, we introduce a new class of structured matrices, which unifies and generalizes structured classes from previous works. We examine properties of this class and build a structured orthogonal parametrization upon it. We then use this parametrization to modify the orthogonal fine-tuning framework, improving parameter and computational efficiency. We empirically validate our method on different domains, including adapting of text-to-image diffusion models and downstream task fine-tuning in language modeling. Additionally, we adapt our construction for orthogonal convolutions and conduct experiments with 1-Lipschitz neural networks.

# 1 Introduction

Orthogonal transforms have proven useful in different deep learning tasks. For example, they were shown to stabilize CNNs [Li et al., 2019, Singla and Feizi, 2021] or used in RNNs to combat the problem of exploding/vanishing gradients [Arjovsky et al., 2016]. Recent works OFT (Orthogonal Fine-Tuning) and BOFT (Butterfly Orthogonal Fine-Tuning) [Qiu et al., 2023, Liu et al., 2024b] use learnable orthogonal matrices for parameter-efficient fine-tuning of neural networks, which prevents training instabilities and overfitting that alternative methods like LoRA [Hu et al., 2022] suffer from.

Nevertheless, parametrization of orthogonal matrices is a challenging task, and the existing methods typically lack in either computational efficiency or expressiveness. Classical methods like Cayley parametrization and matrix exponential map cannot operate under low parameter budget, while Givens rotations and Householder reflections requires computing products of several matrices, which makes their use less efficient in deep learning tasks. Alternative approach in OFT method uses block-diagonal matrix structure in an attempt to be more computationally efficient and use less trainable parameters. Unfortunately, this simple structure can be too restrictive. Thus, arises the problem of constructing dense orthogonal matrix while still being parameter-efficient. While attempting to tackle this task, BOFT method uses a variation of butterfly matrices, parametrizing orthogonal matrices as a product of several matrices with different sparsity patterns, enforcing orthogonality on each of them. This parametrization is able to construct dense matrices while still being parameter-efficient. However it requires to compute a product of multiple matrices (typically up to 6) which can be computationally

expensive. In this paper, we aim to overcome these issues and build dense orthogonal matrices in a more efficient way.

We present a novel structured matrix class parametrized by an alternating product of block-diagonal matrices and several permutations. Multiplying by these matrices can be seen as a consecutive application of independent linear transforms within certain small groups and then shuffling the elements between them, hence the name Group-and-Shuffle matrices (or $\mathcal{GS}$ -matrices for short). This class generalizes Monarch matrices [Dao et al., 2022] and with the right permutation choices, is able to form dense orthogonal matrices more effectively compared to approach proposed in BOFT, decreasing number of matrices in the product as well as the number of trainable parameters. We build efficient structured orthogonal parametrization with this class and use it to construct a new parameter-efficient fine-tuning method named GSOFT.

Our contributions:

- We introduce a new class of structured matrices, called $\mathcal{GS}$ , that is more effective at forming dense matrices than block butterfly matrices from the BOFT method.   
- Using $\mathcal{GS}$ -matrices, we propose an efficient structured orthogonal parametrization, provide theoretical insights and study its performance in the orthogonal fine-tuning framework.   
- We adapt our ideas for convolutional architectures, providing a framework to compress and speed-up orthogonal convolution layers.

# 2 Orthogonal Fine-tuning

Orthogonal Fine-tuning method (OFT) introduced in [Qiu et al., 2023] is a Parameter-Efficient Fine-Tuning (PEFT) method which fine-tunes pre-trained weight matrices through a learnable orthogonal block-diagonal matrix. Some of the properties that make orthogonal transforms desirable are preservation of pair-wise angles of neurons, spectral properties and hyperspherical energy. More precisely, OFT optimizes an orthogonal matrix $Q \in \mathbb{R}^{d \times d}$ for a pre-trained frozen weight matrix $W^0 \in \mathbb{R}^{d \times n}$ and modifies the multiplication $y = (W^0)^\top x$ to $y = (QW^0)^\top x$ . Note that the identity matrix $I$ is orthogonal, which makes it a natural initialization for $Q$ . OFT uses block-diagonal structure for $Q$ , parameterizing it as

$$
Q = \operatorname{diag} (Q _ {1}, Q _ {2}, \dots , Q _ {r}),
$$

where $Q_{i}\in \mathbb{R}^{b\times b}$ are small orthogonal matrices and $br = d$ . Orthogonality is enforced by Cayley parametrization, i.e.

$$
Q _ {i} = (I + K _ {i}) (I - K _ {i}) ^ {- 1},
$$

where $K_{i}$ are skew-symmetric: $K_{i} = -K_{i}^{\top}$ . This ensures orthogonality of $Q_{i}$ and, hence, of Q.

Nevertheless, block-diagonal matrices can be too restrictive, as they divide neurons into r independent groups based on their indices. This motivates the construction of dense parameter-efficient orthogonal matrices. To address this problem, the Orthogonal Butterfly method (BOFT) was introduced [Liu et al., 2024b]. BOFT uses block-butterfly structure to construct Q. Essentially, Q is parameterized as a product of m orthogonal sparse matrices:

$$
Q = B _ {m} B _ {m - 1} \dots B _ {1}.
$$

Each matrix $B_{i}$ is a block-diagonal matrix up to a permutation of rows and columns, consisting of r block matrices of sizes $b \times b$ . Similarly to OFT, the orthogonality is enforced by the Cayley parametrization applied to each block. However, BOFT method has some areas for improvement as well. To construct a dense matrix, BOFT requires at least

$$
m = 1 + \lceil \log_ {2} (r) \rceil
$$

matrices. For example, the authors of BOFT use m = 5 or 6 matrices in the BOFT method for fine-tuning of Stable Diffusion [Rombach et al., 2022]. Large amount of stacked matrices leads to significant time and memory overhead during training. There is also a room for improvement in terms of parameter-efficiency. To overcome these issues, we introduce a new class of structured matrices that we denote GS (group-and-shuffle) that generalizes Monarch matrices [Dao et al., 2022, Fu et al.,

[2023] and show how to use this class to construct parameter-efficient orthogonal parametrization. Similarly to BOFT, our approach uses block-diagonal matrices and permutations, but requires only

$$
m = 1 + \lceil \log_ {b} (r) \rceil
$$

matrices of the same size to construct a dense matrix. See details in Section 5.2. The reduced requirements on m allow us to use m = 2 in experiments to maximize computational efficiency, while still maintaining accurate results.

# 3 GS-matrices

Our motivation within this work is to utilize orthogonal matrices of the form:

$$
A = P _ {L} (L P R) P _ {R} \tag {1}
$$

where matrices L and R are block-diagonal matrices with r blocks of sizes $b \times b$ and $P_{L}, P, P_{R}$ are certain permutation matrices, e.g. $P_{L} = P^{\top}, P_{R} = I$ in the orthogonal fine-tuning setting and $P_{R} = P, P_{L} = I$ for convolutional architectures. Note that although the case $P_{L} = P^{\top}, P_{R} = I$ resembles Monarch matrices [Dao et al., 2022], they are unable to form such a structure, e.g., with equal-sized blocks in L and R. The issue is that the Monarch class has a constraint that interconnects the number of blocks in matrix L and the number of blocks in matrix R (see Appendix C for details). Moreover, Monarch matrices have not been considered with orthogonality constraints.

To build matrices of the form (1), we first introduce a general class of $\mathcal{GS}$ -matrices and study its properties. We then discuss orthogonal matrices from this class in Section 4.

# 3.1 Definition of $\mathcal{GS}$ -matrices

Definition 3.1. An $m \times n$ matrix $A$ is in $\mathcal{GS}(P_L, P, P_R)$ class with $k_L, k_R$ blocks and block sizes $b_L^1 \times b_L^2$ , $b_R^1 \times b_R^2$ if

$$
A = P _ {L} (L P R) P _ {R},
$$

where $L = \mathrm{diag}(L_1, L_2, \ldots, L_{k_L}), L_i \in \mathbb{R}^{b_L^1 \times b_L^2}$ , $R = \mathrm{diag}(R_1, R_2, \ldots, R_{k_R})$ , $R_i \in \mathbb{R}^{b_R^1 \times b_R^2}$ , $P_L, P, P_R$ are permutation matrices and $b_L^2 \cdot k_L = b_R^1 \cdot k_R = s$ , $b_L^1 \cdot k_L = m$ , $b_R^2 \cdot k_R = n$ .

In practice, we fix $P_{L}, P, P_{R}$ depending on the application and only make matrices L, R subject for change. GS-matrices are hardware-efficient, as they are parametrized by two simple types of operations that can be implemented efficiently: multiplications by block-diagonal matrices and permutations.

Let us also illustrate a forward pass $Ax \equiv LPRx$ for a matrix $A \in \mathcal{GS}(I, P, I)$ as a building block for the more general class with two additional permutations. The first operation y = Rx consists of several fully-connected layers, applied individually to subgroups of x, see Figure 1. The next multiplication LPy ensures that these groups interact with each other. Indeed, the permutation matrix P shuffles the entries of y into new subgroups. These subgroups are then again processed by a number of fully-connected layers using L. This motivates the naming for our class of matrices: Group-and-Shuffle or GS for short.

Another useful insight on these matrices is that the class $\mathcal{GS}(I,P,I)$ consists of block matrices with low-rank blocks. The permutation matrix P is responsible for the formation of these blocks and defines their ranks (note that rank may vary

from block to block). The result below formally describes our findings and is key to the projection operation that we describe afterwards.

![](images/b1b959efb5f3d9081ba488dfa70093b6b30bbaf1842e5f7ed46634e6e6a48584.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Layer_L
        L1["L"] --> P1["P"]
        L2["L"] --> P2["P"]
        L3["L"] --> P3["P"]
        L4["L"] --> P4["P"]
        L5["L"] --> P5["P"]
        L6["L"] --> P6["P"]
        L7["L"] --> P7["P"]
        L8["L"] --> P8["P"]
        L9["L"] --> P9["P"]
        L10["L"] --> P10["P"]
        L11["L"] --> P11["P"]
        L12["L"] --> P12["P"]
        L13["L"] --> P13["P"]
        L14["L"] --> P14["P"]
        L15["L"] --> P15["P"]
        L16["L"] --> P16["P"]
        L17["L"] --> P17["P"]
        L18["L"] --> P18["P"]
        L19["L"] --> P19["P"]
        L20["L"] --> P20["P"]
        L21["L"] --> P21["P"]
        L22["L"] --> P22["P"]
        L23["L"] --> P23["P"]
        L24["L"] --> P24["P"]
        L25["L"] --> P25["P"]
        L26["L"] --> P26["P"]
        L27["L"] --> P27["P"]
        L28["L"] --> P28["P"]
        L29["L"] --> P29["P"]
        L30["L"] --> P30["P"]
        L31["L"] --> P31["P"]
        L32["L"] --> P32["P"]
        L33["L"] --> P33["P"]
        L34["L"] --> P34["P"]
        L35["L"] --> P35["P"]
        L36["L"] --> P36["P"]
        L37["L"] --> P37["P"]
        L38["L"] --> P38["P"]
        L39["L"] --> P39["P"]
        L40["L"] --> P40["P"]
        L41["L"] --> P41["P"]
        L42["L"] --> P42["P"]
        L43["L"] --> P43["P"]
        L44["L"] --> P44["P"]
        L45["L"] --> P45["P"]
        L46["L"] --> P46["P"]
        L47["L"] --> P47["P"]
        L48["L"] --> P48["P"]
        L49["L"] --> P49["P"]
        L50["L"] --> P50["P"]
        L51["L"] --> P51["P"]
        L52["L"] --> P52["P"]
        L53["L"] --> P53["P"]
        L54["L"] --> P54["P"]
        L55["L"] --> P55["P"]
        L56["L"] --> P56["P"]
        L57["L"] --> P57["P"]
        L58["L"] --> P58["P"]
        L59["L"] --> P59["P"]
        L60["L"] --> P60["P"]
        L61["L"] --> P61["P"]
        L62["L"] --> P62["P"]
        L63["L"] --> P63["P"]
        L64["L"] --> P64["P"]
        L65["L"] --> P65["P"]
        L66["L"] --> P66["P"]
        L67["L"] --> P67["P"]
        L68["L"] --> P68["P"]
        L69["L"] --> P69["P"]
        L70["L"] --> P70["P"]
        L71["L"] --> P71["P"]
        L72["L"] --> P72["P"]
        L73["L"] --> P73["P"]
        L74["L"] --> P74["P"]
        L75["L"] --> P75["P"]
        L76["L"] --> P76["P"]
        L77["L"] --> P77["P"]
        L78["L"] --> P78["P"]
        L79["L"] --> P79["P"]
        L80["L"] --> P80["P"]
        L81["L"] --> P81["P"]
        L82["L"] --> P82["P"]
        L83["L"] --> P83["P"]
        L84["L"] --> P84["P"]
        L85["L"] --> P85["P"]
        L86["L"] --> P86["P"]
        L87["L"] --> P87["P"]
        L88["L"] --> P88["P"]
        L89["L"] --> P89["P"]
        L90["L"] --> P90["P"]
    end
    subgraph Layer_R
        R1["R1"] --> R2["R2"]
    end
    subgraph Output
        R3["R3"] --> R4["R4"]
    end
    subgraph Output
        R5["R5"] --> R6["R6"]
    end
    subgraph Output
        R7["R7"] --> R8["R8"]
    end
    subgraph Output
        R9["R9"] --> R10["R10"]
```
</details>

Figure 1: $\mathcal{GS}(I, P, I)$ matrices with $b_{L}^{1} = b_{L}^{2} = 3$ , $b_{R}^{1} = b_{R}^{2} = 2$ , $k_{L} = 2$ , $k_{R} = 3$ . Edges between nodes denote nonzero weights.

Proposition 1. Let A be a matrix from $\mathcal{GS}(I, P, I)$ with a permutation matrix P defined by the function $\sigma : \{0, \ldots, n-1\} \to \{0, \ldots, n-1\}$ . Let $\{v_{i}^{\top}\} - be$ the rows of the blocks $R_{1}, \ldots, R_{k_{R}}$ ,

$\{u_i\} -$ the columns of the blocks $L_{1},\ldots ,L_{k_{L}}$ in the consecutive order. Then the matrix $A$ can be written as a block matrix with $k_{L}\times k_{R}$ blocks using the following formula for each block $A_{k_1,k_2}$ :

$$
A_{k_{1},k_{2}} = \sum_{\substack{\lfloor \frac{\sigma(i)}{k_{L}}\rfloor = k_{1}\\ \lfloor \frac{i}{k_{R}}\rfloor = k_{2}}}u_{\sigma (i)}v_{i}^{\top}.
$$

Note that we use zero-indexing for this proposition for simplicity of formulas.

![](images/583183c18010844ef456b68b2aa0f29bd7ba14f0e61b8f7579c817f35bacce49.jpg)

<details>
<summary>text_image</summary>

... + u₂v₄ᵀ
= u₂
v₄ᵀ
</details>

Figure 2: Illustration of Proposition 1 that provides block low-rank interpretation of $\mathcal{GS}(I, P, I)$ matrices. The matrix $R$ contains 2 blocks and matrix $L$ contains 4 blocks.

Let us illustrate this proposition in Figure 2. We consider $\mathcal{GS}(I, P, I)$ with $k_L = 4$ and $k_R = 2$ blocks in $L$ and $R$ and with the block sizes $3 \times 3$ and $6 \times 6$ respectively. Let us consider the leading block $A_{00}$ of the size $3 \times 6$ . According to Proposition 1, $A_{00} = u_0 v_2^\top + u_2 v_4^\top$ . Indeed, let us take a closer look, e.g., at the term $u_2 v_4^\top$ . In the permutation matrix $P$ , we have a nonzero element in the position $(2, 4)$ as $i = 4$ and $\sigma(4) = 2$ . Therefore, we select the third column $u_2$ in $L_1$ and the fifth row $v_4^\top$ in $R_1$ . This leads to adding a rank-one term $u_2 v_4^\top$ to $A_{00}$ as we see in the formula above.

Another direct corollary from Proposition 1 is a projection operation $\pi\colon\mathbb{R}^{m\times n}\to\mathcal{GS}(P_{L},P,P_{R})$ that satisfies:

$$
\pi (A) \in \operatorname * {a r g   m i n} _ {B \in \mathcal {G S} (P _ {L}, P, P _ {R})} \| A - B \| _ {F},
$$

where $\|\cdot\|_{F}$ is the Frobenius norm. Thanks to the block low-rank representation of matrices from $\mathcal{GS}(P_{L}, P, P_{R})$ , the projection $\pi$ is simply constructed using SVD truncations of the blocks $(P_{L}^{\top}AP_{R}^{\top})_{k_{1}, k_{2}}$ and is summarized in Algorithm 1.

Algorithm 1 Projection $\pi (\cdot)$ of $A$ onto $\mathcal{GS}(P_L,P,P_R)$   
Input: $A, P_{L}, P, P_{R}$ Return: L, R
for $k_{1} = 1 \ldots k_{L}$ do
    for $k_{2} = 1 \ldots k_{R}$ do
    Compute SVD of $(P_{L}^{T}AP_{R}^{T})_{k_{1},k_{2}} = U\Sigma V^{\top}$ ;
    Set $r = r_{k_{1},k_{2}} - rank$ of block determined by P;
    Take $U_{r} = U[:r,:], \Sigma_{r} = \Sigma[:r,:r], V_{r} = V[:r,:]$ ;
    Pack columns of $U_{r}\Sigma_{r}^{1/2}$ into $L_{k_{1}}$ and rows of $\Sigma_{r}^{1/2}V_{r}$ into $R_{k_{2}}$ according to P;
    end for
end for

# 4 Orthogonal $\mathcal{G}\mathcal{S}(P_{L}, P, P_{R})$ matrices

In this section, we study the orthogonality constraint for the $\mathcal{GS}(P_{L}, P, P_{R})$ to obtain structured orthogonal representation. This is one of the main contributions of our paper and we utilize this class in all the numerical experiments. Since we are interested only in square orthogonal matrices, we additionally assume that m = n and $b_{L}^{1} = b_{L}^{2} = b_{L}$ ; $b_{R}^{1} = b_{R}^{2} = b_{R}$ . Similarly to parametrizations in OFT and BOFT, a natural way to enforce orthogonality of $\mathcal{GS}(P_{L}, P, P_{R})$ -matrices is to enforce

orthogonality of each block of $L$ and $R$ . This indeed leads an orthogonal matrix since permutation matrices are also orthogonal as well as a product of orthogonal matrices. However, it is not immediately obvious that there exist no orthogonal matrices from $\mathcal{GS}(P_L, P, P_R)$ that cannot be represented this way. Surprisingly, we find that such a way to enforce orthogonality is indeed sufficient for covering of all orthogonal matrices from $\mathcal{GS}(P_L, P, P_R)$ .

Theorem 1. Let A be any orthogonal matrix from $\mathcal{GS}(P_{L}, P, P_{R})$ . Then, A admits $P_{L}(LPR)P_{R}$ representation with the matrices L, R consisting of orthogonal blocks.

Proof. Matrices $P_{L}, P_{R}$ are orthogonal as they are permutation matrices. It means that it is sufficient to prove theorem in the case when A is from $\mathcal{GS}(I, P, I)$ , which means that we can use low block-rank structure interpretation from Proposition 1. Consider a skeleton decomposition of the blocks $A_{ij} = U_{ij} V_{ij}^{\top}$ , $U_{ij} \in R^{b_{L} \times r_{ij}}$ , $V_{ij} \in R^{b_{R} \times r_{ij}}$ such that $U_{ij}^{\top} U_{ij} = I_{r_{ij}}$ (this can be ensured, e.g., using the QR decomposition). Then

$$
A = \left( \begin{array}{c c c} U _ {1, 1} V _ {1, 1} ^ {\top} & \ldots & U _ {1, k _ {L}} V _ {1, k _ {L}} ^ {\top} \\ \vdots & \ddots & \vdots \\ U _ {k _ {L}, 1} V _ {k _ {L}, 1} ^ {\top} & \ldots & U _ {k _ {L}, k _ {R}} V _ {k _ {L}, k _ {R}} ^ {\top} \end{array} \right).
$$

Take the $j$ -th block-column of $A$ . Since $A$ is an orthogonal matrix, we get:

$$
\left( \begin{array}{c c c} V _ {1, j} U _ {1, j} ^ {\top} & \ldots & V _ {k _ {L}, j} U _ {k _ {L}, j} ^ {\top} \end{array} \right) \left( \begin{array}{c} U _ {1, j} V _ {1, j} ^ {\top} \\ \vdots \\ U _ {k _ {L}, j} V _ {k _ {L}, j} ^ {\top} \end{array} \right) = I _ {b _ {R}}
$$

Multiplying matrices in the l.h.s. we get $V_{1,j}U_{1,j}^{\top}U_{1,j}V_{1,j}^{\top}+\cdots+V_{k_{L},j}U_{k_{L},j}^{\top}U_{k_{L},j}V_{k_{L},j}^{\top}=I_{b_{R}}$ . Since $U_{ij}^{\top}U_{ij}=I_{r_{ij}}$ we conclude $V_{1,j}V_{1,j}^{\top}+\cdots+V_{k_{L},j}V_{k_{L},j}^{\top}=I_{b_{R}}$ . This implies that $(V_{1,j}\ldots V_{k_{L},j})$ is an orthogonal matrix. Note that if we now parameterize A=LPR with the matrices $V_{ij}$ packed into R and $U_{ij}$ packed into L, then $(V_{1,j}\ldots V_{k_{L},j})$ is exactly the j-th block matrix in R up to permutation of rows. Therefore, every block in R is an orthogonal matrix. Since we now proved that $V_{ij}^{\top}V_{ij}=I$ , we can use same the derivation for the rows of A and conclude that blocks of L are also orthogonal.

# 5 $\mathcal{GS}(P_{m + 1},\dots ,P_1)$ matrices

In this section we describe an the extension of GS-matrices that uses more than two block-diagonal matrices and show that with the right permutations choices GS-matrices are more effective than block butterfly matrices in forming dense matrices. Here by dense matrices we imply matrices that do not contain zero entries at all.

Definition 5.1. A is said to be in $\mathcal{GS}(P_{m+1},\ldots,P_1)$ if

$$
A = P _ {m + 1} \prod_ {i = m} ^ {1} (B _ {i} P _ {i}),
$$

where each matrix $B_{i}$ is a block-diagonal matrix with $k_{i}$ blocks of size $b_{i}^{1} \times b_{i}^{2}$ , matrices $P_{i}$ are permutation matrices and $b_{i}^{1} \cdot k_{i} = b_{i+1}^{2} \cdot k_{i+1}$ .

Remark 1. Similarly to the case $m = 2$ described in Section 3, we may use orthogonal blocks in $B_i$ , $i = 1, \ldots, m + 1$ to obtain orthogonal matrices. However, it is not clear if an analog to Theorem 1 is correct in this case as well.

Remark 2. For each of the classes of Block Butterfly matrices [Chen et al., 2022], Monarch matrices [Dao et al., 2022] and order-p Monarch matrices [Fu et al., 2023], there exist permutation matrices $P_{m+1}, \ldots, P_1$ such that $\mathcal{GS}(P_{m+1}, \ldots, P_1)$ coincides with a respective class. Indeed, Monarch matrices have the form of alternating block-diagonal matrices and permutations with some specific size constraints and sparse matrices in the product of Block Butterfly matrices can be easily transformed to block-diagonal matrices with permutations of rows and columns.

# 5.1 Choosing permutation matrices

We suggest using the following matrices with $k = k_{i}$ for $P_{i}$ . Note that this is efficient for forming dense matrices as follows from the proof of Theorem 2. This is by contrast to the permutations used in [Fu et al., 2023] that are restricted to particular matrix sizes.

Definition 5.2 ([Dao et al., 2022]). Let $P_{(k,n)}$ be a permutation matrix given by permutation $\sigma$ on $\{0,1,\ldots ,n - 1\}$ :

$$
\sigma (i) = (i \bmod k) \cdot \frac {n}{k} + \left\lfloor \frac {i}{k} \right\rfloor .
$$

Applying this permutation to a vector can be viewed as reshaping an input of size n into an $k \times \frac{n}{k}$ matrix in a row-major order, transposing it, and then vectorizing the result back into a vector (again in row-major column). We provide several examples of such permutations in Figure 3.

![](images/c0c0187b695a2911564c668240c6b49e44953a8a18a88894b20cc679ea0cc014.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Layer"] --> B["Hidden Layer"]
    B --> C["Output Layer"]
    subgraph Layer1
        D["Input Layer"] --> E["Hidden Layer"]
        E --> F["Output Layer"]
    end
    subgraph Layer2
        G["Input Layer"] --> H["Hidden Layer"]
        H --> I["Output Layer"]
    end
    subgraph Layer3
        J["Input Layer"] --> K["Hidden Layer"]
        K --> L["Output Layer"]
    end
    subgraph Layer4
        M["Input Layer"] --> N["Hidden Layer"]
        N --> O["Output Layer"]
    end
    subgraph Layer5
        P["Input Layer"] --> Q["Hidden Layer"]
        Q --> R["Output Layer"]
    end
    subgraph Layer6
        S["Input Layer"] --> T["Hidden Layer"]
        T --> U["Output Layer"]
    end
    subgraph Layer7
        V["Input Layer"] --> W["Hidden Layer"]
        W --> X["Output Layer"]
    end
    subgraph Layer8
        Y["Input Layer"] --> Z["Hidden Layer"]
        Z --> AA["Output Layer"]
    end
    subgraph Layer9
        AB["Input Layer"] --> AC["Hidden Layer"]
        AC --> AD["Output Layer"]
    end
    subgraph Layer10
        AE["Input Layer"] --> AF["Hidden Layer"]
        AF --> AG["Output Layer"]
    end
    subgraph Layer11
        AH["Input Layer"] --> AI["Hidden Layer"]
        AI --> AJ["Output Layer"]
    end
    subgraph Layer12
        AK["Input Layer"] --> AL["Hidden Layer"]
        AL --> AM["Output Layer"]
```
</details>

![](images/9304503aa59ace798097bb261755c67063914fcba8d9618c141adfaf67e061c2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Node 1"] --> B["Output Node 1"]
    C["Input Node 2"] --> D["Output Node 2"]
    E["Input Node 3"] --> F["Output Node 3"]
    G["Input Node 4"] --> H["Output Node 4"]
    I["Input Node 5"] --> J["Output Node 5"]
    K["Input Node 6"] --> L["Output Node 6"]
    M["Input Node 7"] --> N["Output Node 7"]
    O["Input Node 8"] --> P["Output Node 8"]
    Q["Input Node 9"] --> R["Output Node 9"]
    S["Input Node 10"] --> T["Output Node 10"]
    U["Output Node 11"] --> V["Output Node 11"]
    W["Output Node 12"] --> X["Output Node 12"]
```
</details>

![](images/6395448e7532df5b872406a95a3e878f65fa639486659961ed04feda78a26194.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A[" "] --> B[" "]
    A --> C[" "]
    A --> D[" "]
    A --> E[" "]
    A --> F[" "]
    A --> G[" "]
    A --> H[" "]
    A --> I[" "]
    A --> J[" "]
    A --> K[" "]
    A --> L[" "]
    B --> M[" "]
    C --> N[" "]
    D --> O[" "]
    E --> P[" "]
    F --> Q[" "]
    G --> R[" "]
    H --> S[" "]
    I --> T[" "]
    J --> U[" "]
    K --> V[" "]
    L --> W[" "]
    M --> X[" "]
    N --> Y[" "]
    O --> Z[" "]
    P --> AA[" "]
    Q --> AB[" "]
    R --> AC[" "]
    S --> AD[" "]
    T --> AE[" "]
    U --> AF[" "]
    V --> AG[" "]
    W --> AH[" "]
    X --> AI[" "]
    Y --> AJ[" "]
    Z --> AK[" "]
    AA --> AL[" "]
    AB --> AM[" "]
    AC --> AN[" "]
    AD --> AO[" "]
    AE --> AP[" "]
    AF --> AQ[" "]
    AG --> AR[" "]
    AH --> AS[" "]
    AI --> AT[" "]
    AJ --> AU[" "]
    AK --> AV[" "]
```
</details>

![](images/5ef8e44ec74b53d7e5be62ddaf13427fdcbd9c114ff2e9b21e8534b6fb8188b5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A1["●"] --> B1["●"]
    A2["○"] --> B2["○"]
    A3["○"] --> B3["○"]
    A4["○"] --> B4["○"]
    A5["○"] --> B5["○"]
    A6["○"] --> B6["○"]
    A7["○"] --> B7["○"]
    A8["○"] --> B8["○"]
    A9["○"] --> B9["○"]
    A10["●"] --> B10["●"]
    A11["○"] --> B11["○"]
    A12["○"] --> B12["○"]
    A13["○"] --> B13["○"]
    A14["○"] --> B14["○"]
    A15["○"] --> B15["○"]
    A16["○"] --> B16["○"]
    A17["○"] --> B17["○"]
    A18["○"] --> B18["○"]
    A19["○"] --> B19["○"]
    A20["●"] --> B20["●"]
    A21["○"] --> B21["○"]
    A22["○"] --> B22["○"]
    A23["○"] --> B23["○"]
    A24["○"] --> B24["○"]
    A25["○"] --> B25["○"]
    A26["○"] --> B26["○"]
    A27["○"] --> B27["○"]
    A28["○"] --> B28["○"]
    A29["○"] --> B29["○"]
    A30["●"] --> B30["●"]
    A31["○"] --> B31["○"]
    A32["○"] --> B32["○"]
    A33["○"] --> B33["○"]
    A34["○"] --> B34["○"]
    A35["○"] --> B35["○"]
    A36["○"] --> B36["○"]
    A37["○"] --> B37["○"]
    A38["○"] --> B38["○"]
    A39["○"] --> B39["○"]
    A40["●"] --> B40["●"]
    A41["○"] --> B41["○"]
    A42["○"] --> B42["○"]
    A43["○"] --> B43["○"]
    A44["○"] --> B44["○"]
    A45["○"] --> B45["○"]
    A46["○"] --> B46["○"]
    A47["○"] --> B47["○"]
    A48["○"] --> B48["○"]
    A49["●"] --> B49["●"]
    A50["○"] --> B50["○"]
    A51["○"] --> B51["○"]
    A52["○"] --> B52["○"]
    A53["○"] --> B53["○"]
    A54["○"] --> B54["○"]
    A55["○"] --> B55["○"]
```
</details>

Figure 3: Illustration of $P_{(k,12)}$ permutations for $k \in \{3,4,6,2\}$ .

# 5.2 Comparison to block butterfly matrices and BOFT

Block Butterfly matrices were introduced in [Chen et al., 2022] and are used to construct orthogonal matrices in the BOFT method. Block Butterfly matrix class is a special case of higher-order $\mathcal{GS}$ -matrices with $k_{i} = r$ and $b_{i}^{1} = b_{i}^{2} = b = 2s$ and certain permutation choices. However, we argue that the choice of these permutations are sub-optimal for construction of dense matrix and using permutations from Definition 5.2 is more effective. When using block-diagonal matrices with $r$ blocks, block butterfly matrices need $1 + \lceil \log_2(r) \rceil$ matrices to construct a dense matrix. For $\mathcal{GS}$ -matrices we have the following result.

Theorem 2. Let $k_{i}=r, b_{i}^{1}=b_{i}^{2}=b$ . Then using $m=1+\lceil\log_{b}(r)\rceil$ is sufficient for the class $\mathcal{GS}(P_{L}, P_{(k,br)}, \ldots, P_{(k,br)}, P_{R})$ to form a dense matrix for any $P_{L}, P_{R}$ . Moreover, the choice of $P_{2}=\cdots=P_{m}=P_{(k,b)}$ is optimal in the sense that all matrices from $\mathcal{GS}(P_{m+1}, \ldots, P_{1})$ contain zero blocks for any integer $m<1+\lceil\log_{b}(r)\rceil$ and any permutations $P_{1}, \ldots, P_{m+1}$ .

Proof. See Appendix D.

![](images/d20d37d71674b9108a3233434f45781c28974f27ed6980bbc335ad6eb0831f0f.jpg)

For example, let us consider a case of constructing a dense orthogonal matrix of the size $1024 \times 1024$ . Suppose also that we use block matrices with block size 32. Constructing a dense matrix with Block Butterfly matrices requires $1 + \log_{2}(32) = 6$ butterfly matrices, which leads to $6 \times 32^{3}$ parameters in the representation. $\mathcal{GS}(P_{L}, P, P_{R})$ matrices with $P = P_{(32,1024)}$ only need two matrices to construct a dense matrix yielding $2 \times 32^{3}$ parameters. The $\mathcal{GS}(P_{L}, P, P_{R})$ parametrization is also naturally more computationally efficient as fewer number of multiplications is both faster and requires less cached memory for activations.

# 6 Applications

# 6.1 Orthogonal fine-tuning with $\mathcal{GS}(P_L, P, P_R)$ (GSOFT)

We utilize the pipeline of OFT and BOFT methods with the exception of parametrizing Q with orthogonal permuted $\mathcal{GS}(P_{L}, P, P_{R})$ matrices. In particular, for parametrization of $Q \in R^{d \times d}$ , we utilize the $\mathcal{GS}(P^{\top}, P, I)$ class, i.e. $Q = P^{\top} LPR$ , where $L = \text{diag}(L_{1}, \ldots, L_{r})$ , $L_{i} \in R^{b \times b}$ , $R = \text{diag}(R_{1}, \ldots, R_{r})$ , $R_{i} \in R^{b \times b}$ . For consistency, we use the same notation for the number of blocks and block sizes as in BOFT and OFT methods. We use $P_{(r, br)}$ as a permutation matrix P. To enforce orthogonality, we parameterize each block in matrices L, R with the Cayley parametrization. We initialize Q as an identity matrix by initializing each block to be an identity matrix. Additional techniques like magnitude scaling and multiplicative dropout that are used in OFT and BOFT can be utilized the same way in our method, though we only use scaling in our experiments. Note that likewise in OFT, BOFT weights of the matrix Q can be merged with the pretrained weight W producing no inference overhead.

# 6.2 Two-sided orthogonal fine-tuning (Double GSOFT)

Consider SVD decomposition of a matrix $W^{0} = U\Sigma V^{\top}$ . Applying orthogonal fine-tuning, we get $W' = (QU)\Sigma V^{\top}$ , which is an SVD decomposition for the adapted weight $W'$ . This shows that we can only change left singular vectors U with the standard orthogonal fine-tuning paradigm. At the same time, the LoRA method modifies both matrices U and V. Moreover, recent papers [Meng et al., 2024, Li et al., 2023] show that initializing matrices A, B with singular vectors can additionally boost performance of LoRA. This motivates an extension of orthogonal fine-tuning method, that can adapt both matrices U and V. We introduce a simple approach that multiplies pre-trained weight matrices from both sides, rather than one. This method modifies forward pass from $z = (W^{0})^{\top}x$ to

$$
z = (Q _ {U} W ^ {0} Q _ {V}) ^ {\top} x
$$

Where $Q_{U}$ and $Q_{V}$ are parametrized as orthogonal GS-matrices. In cases where BOFT utilizes 5-6 matrices, we can leverage the fact that our method uses only 2 and adapt both sides while still using less matrices and trainable parameters than BOFT.

# 6.3 GS Orthogonal Convolutions

Recall, that due to linearity of a multichannel convolution operation, we can express the convolution of tensor $X \in R^{c_{in} \times h \times w}$ with a kernel $L \in R^{c_{out} \times c_{in} \times k \times k}$ $L \star X$ in terms of matrix multiplication [Singla and Feizi, 2021]:

$$
Y = L \star X \quad \Leftrightarrow \quad v e c (Y) = \left[ \begin{array}{c c c} L _ {0, 0} & \dots & L _ {0, c _ {i n} - 1} \\ \vdots & \ddots & \vdots \\ L _ {c _ {o u t} - 1, 0} & \dots & L _ {c _ {o u t} - 1, c _ {i n} - 1} \end{array} \right] v e c (X), \tag {2}
$$

where $L_{i,j}$ is doubly Toeplitz matrix, corresponding to convolution between i-th and j-th channels and $vec(X)$ is a vectorization of tensor into a vector in a row-major order. Thus, the convolution is essentially a block matrix, where each block represents a standard convolution operation. Using this block interpretation (2), we may apply the concept of GS matrices to convolutional layers as well. Considering each convolution between channels as an element of our block matrix, we can set some of these blocks to zero, obtaining some additional structure. Thus, we can construct block matrix which has block-diagonal structure, corresponding to grouped convolution (further, in all equations we will denote it as GrConv). Then, defining ChShuffle as a permutation of channels, like in [Zhang et al., 2017], we obtain structure, which is similar to GSOFT, defined in Section 6:

$$
Y = \operatorname{GrConv} _ {2} \left(\text {ChShuffle} _ {2} \left(\operatorname{GrConv} _ {1} \left(\text {ChShuffle} _ {1} (X)\right)\right)\right). \tag {3}
$$

The proposed $\mathcal{GS}$ convolutional layer shuffles information between each pair of input channels and requires less parameters and FLOPs during computations. In this example we can also choose permutations of channels and change kernel size. This convolutional layer can be treated as $\mathcal{GS}(P_{m+1},\ldots,P_1)$ matrix in vectorized view, that is why choosing permutations between convolutional layers is also very important for information transition properties. In Appendix F we explain the choice of ChShuffle operation.

We can use the proposed layer to construct orthogonal convolutions (transformations with an orthogonal Jacobian matrix) similarly to skew orthogonal convolution (SOC) architecture, that uses Taylor expansion of a matrix exponential. One major downside of methods such as SOC and BCOP [Li et al., 2019] is that they require more time than basic convolution operation. For instance, in the SOC method, one layer requires multiple applications of convolution (6 convolutions per layer). In our framework, we propose a parametrization of a convolutional layer, in which imposing an orthogonality to convolutions has fewer number of FLOPs and parameters thanks to the usage of grouped convolutions.

Let us discuss in more details how SOC works and the way we modify it. In SOC, a convolutional filter is parametrized in the following way:

$$
L = M - \operatorname{ConvTranspose} (M),
$$

where $M \in \mathbb{R}^{c_{in} \times c_{out} \times r \times s}$ is an arbitrary kernel and the ConvTranspose is the following operation:

$$
\operatorname{ConvTranspose} (M) _ {i, j, k, l} = M _ {j, i, r - k - 1, s - l - 1}
$$

This parametrization of filter L makes the matrix from Equation 2 skew-symmetric. As matrix exponential of skew-symmetric matrix is an orthogonal matrix, in SOC the authors define convolution exponential operation, which is equivalent to matrix exponential in matrix-vector notation:

Definition 6.1. [Singla and Feizi, 2021] Let $X \in \mathbb{R}^{c \times h \times w}$ be an input tensor and $L \in \mathbb{R}^{c \times c \times k \times k}$ be a convolution kernel. Then, define convolution exponential $L \star_e X$ as follows:

$$
L \star_ {e} X = X + \frac {L \star X}{1 !} + \frac {L \star^ {2} X}{2 !} + \dots
$$

where $L \star^{i} X$ is a convolution with kernel L applied i times consequently.

As mentioned above, with proper initialization we get a convolutional layer with orthogonal Jacobian matrix. Using the parametrization of convolution layer from the Equation 3 and substituting there two grouped convolution exponentials (e.g. in our parametrization we have the same convolution exponential, but we have grouped convolution instead of basic one) with the parameterized kernel:

$$
Y = \operatorname{GrExpConv} _ {2} \left(\operatorname{ChShuffle} _ {2} \left(\operatorname{GrExpConv} _ {1} \left(\operatorname{ChShuffle} _ {1} (X)\right)\right)\right)
$$

In our experiments we tried different layer architectures and we found that making kernel size of the second convolution equal to 1 speeds up our convolutional layer, maintaining quality metrics. Thus, if convolutional layer consists of two grouped convolutional exponentials, the second convolutional exponential has $\text{kernel\_size} = 1 \times 1$

# 7 Experiments

All the experiments below were conducted on NVIDIA V100-SXM2-32Gb GPU. We ran all the experiments within $\sim$ 2000 GPU hours.

# 7.1 Natural language understanding

We report result on the GLUE [Wang et al., 2018] benchmark with RoBERTa-base [Liu et al., 2019] model. Benchmark includes several classification tasks that evaluate general language understanding. We follow training settings of [Liu et al., 2024b, Zhang et al., 2023]. We apply adapters for all linear layers in the attention and MLP and only tune learning rate for all methods. Table 1 reports best results on the evaluation set from the whole training. LoRA, OFT and BOFT are implemented with PEFT library [Mangrulkar et al., 2022]. GSOFT method outperforms OFT, BOFT and also have a slight edge over LoRA. Note that even though skew-symmetric K theoretically matrix only requires approximately half the parameters of a full matrix, in practice it is parametrized as $K = A - A^{T}$ for the ease of computations. However, after fine-tuning, one can only save upper-triangular part of K. Doing this, orthogonal fine-tuning methods become approximately 2 times more efficient in terms of memory savings.

Table 1: Results on GLUE benchmark with RoBERTa-base model. We report Pearson correlation for STS-B, Matthew's correlation for CoLA and accuracy for other tasks. # Params denotes number of trainbale parameters 

<table><tr><td>Method</td><td># Params</td><td>MNLI</td><td>SST-2</td><td>CoLA</td><td>QQP</td><td>QNLI</td><td>RTE</td><td>MRPC</td><td>STS-B</td><td>ALL</td></tr><tr><td>FT</td><td>125M</td><td>87.62</td><td>94.38</td><td>61.97</td><td>91.5</td><td>93.06</td><td>80.14</td><td>88.97</td><td>90.91</td><td>86.07</td></tr><tr><td> $LoRA_{r=8}$ </td><td>1.33M</td><td>87.82</td><td>95.07</td><td>64.02</td><td>90.97</td><td>92.81</td><td>81.95</td><td>88.73</td><td>90.84</td><td>86.53</td></tr><tr><td> $OFT_{b=16}$ </td><td>1.41M</td><td>87.21</td><td>95.07</td><td>64.37</td><td>90.6</td><td>92.48</td><td>79.78</td><td>89.95</td><td>90.71</td><td>86.27</td></tr><tr><td> $BOFT_{b=8}^{m=2}$ </td><td>1.42M</td><td>87.14</td><td>94.38</td><td>62.57</td><td>90.48</td><td>92.39</td><td>80.14</td><td>88.97</td><td>90.67</td><td>85.84</td></tr><tr><td> $GSOFT_{b=8}$ </td><td>1.42M</td><td>87.16</td><td>95.06</td><td>65.3</td><td>90.46</td><td>92.46</td><td>81.95</td><td>90.2</td><td>90.76</td><td>86.67</td></tr></table>

Table 2: Results on subject-driven generation. # Params denotes the number of training parameters in each parametrization. Training time is computed for 3000 iterations on a single GPU V100 in hours. 

<table><tr><td rowspan="3">Model</td><td rowspan="3">Full</td><td colspan="3">LoRA</td><td colspan="3">BOFT</td><td colspan="3">GSOFT (Ours)</td><td colspan="3">Double GSOFT (Ours)</td></tr><tr><td colspan="3">rank</td><td colspan="3">r, m</td><td colspan="3">r</td><td colspan="3">r</td></tr><tr><td>4</td><td>32</td><td>128</td><td>32, 4</td><td>32, 6</td><td>16, 5</td><td>32</td><td>16</td><td>8</td><td>64</td><td>32</td><td>16</td></tr><tr><td># Params</td><td>99.9M</td><td>0.8M</td><td>6.6M</td><td>26.6M</td><td>13.6M</td><td>20.4M</td><td>33.8M</td><td>6.8M</td><td>13.6M</td><td>27.1M</td><td>6.5M</td><td>13.0M</td><td>25.9M</td></tr><tr><td>Training time</td><td>1.3</td><td>1.3</td><td>1.3</td><td>1.3</td><td>2.0</td><td>2.2</td><td>2.3</td><td>1.5</td><td>1.6</td><td>1.8</td><td>1.7</td><td>2.0</td><td>1.8</td></tr><tr><td>CLIP-I↑</td><td>0.805</td><td>0.805</td><td>0.819</td><td>0.813</td><td>0.803</td><td>0.796</td><td>0.789</td><td>0.805</td><td>0.803</td><td>0.783</td><td>0.815</td><td>0.802</td><td>0.783</td></tr><tr><td>CLIP-T↑</td><td>0.212</td><td>0.246</td><td>0.236</td><td>0.223</td><td>0.244</td><td>0.234</td><td>0.223</td><td>0.256</td><td>0.245</td><td>0.227</td><td>0.256</td><td>0.242</td><td>0.225</td></tr></table>

# 7.2 Subject-driven generation

Subject-driven generation [Ruiz et al., 2023, Gal et al., 2022] is an important and challenging task in the field of generative modelling. Given several photos of a particular concept, we want to introduce it to the diffusion model so that we can generate this particular object in different scenes described by textual prompts. The main way to do this is to fine-tune the model. However, the large number of fine-tuning parameters together with the lack of training images make the model prone to overfitting, i.e. the model reconstructs the concept almost perfectly, but starts to ignore the textual prompt during generation. To solve this problem and stabilize the fine-tuning process, different lightweight parameterizations [Qiu et al., 2023, Liu et al., 2024b, Hu et al., 2022, Tewel et al., 2023, Han et al., 2023] and regularization techniques [Ruiz et al., 2023, Kumari et al., 2023] are widely used in this task. Therefore, we chose this setting to evaluate the effectiveness of the proposed orthogonal parameterization compared to other approaches.

We use StableDiffusion [Rombach et al., 2022] and the Dreambooth [Ruiz et al., 2023] dataset for all our experiments. The following parameterizations were considered as baselines in this task: full (q, k, v and out.0 layers in all cross- and self- attentions of the UNet are trained), LoRA [Hu et al., 2022] and BOFT [Liu et al., 2024b] applied to the same layer. We use our GSOFT parameterization and a two-sided orthogonal GSOFT (Double GSOFT) applied to the same layers as baselines. For a more comprehensive comparison, we consider different hyperparameters for the models, adjusting the total number of optimized parameters. More training and evaluation details can be found in Appendix E.

CLIP image similarity, CLIP text similarity and visual comparison for this task are presented in Table 2 and Figure 4. As the results show, GSOFT and DoubleGSOFT are less prone to overfitting compared to the baselines. They show better alignment with text prompts while maintaining a high level of concept fidelity. Furthermore, both methods with optimal hyperparameters are more efficient than BOFT and comparable to LoRA and full parameterization in terms of training time. See Appendix E for more visual and quantitative comparison.

# 7.3 GS Orthogonal Convolutions

Following [Singla and Feizi, 2021], we train LipConvnet-n on CIFAR-100 dataset. LipConvnet-n is 1-Lipschitz neural network, i.e. neural network with Lipschitz constant equal to 1, his property provides certified adversarial robustness. LipConvnet uses orthogonal convolutions and gradient preserving activations in order to maintain 1-Lipschitz property.

![](images/dfcbbc8fa723d989809f3c17389377de9b017a21e028007143d480d3c5193d6f.jpg)  
Figure 4: Subject-driven generation visual results on 3000 training iterations.

LipConvnet-n architecture consists of 5 equal blocks, each having $\frac{n}{5}$ skew orthogonal convolutions, where the last convolution at each level downsamples image size. We replace the skew orthogonal convolution layer with the structured version using GSorthogonal convolutions and test it in the setting of [Singla and Feizi, 2021], using the same hyperparameters (learning rate, batch size and scheduler stable during testing). In layers where we have two GrExpConv, the second convolution has kernel size equal to 1.

We also use a modified activation function (MaxMinPermuted instead of MaxMin), which uses different pairing of channels. This makes activations aligned with the ChShuffle operation and grouped convolutions. The choice of permutation for ChShuffle also slightly differs from permutations defin in Definition 5.2 because of the interplay between activations and convolutional layers. We provide definitions and intuition regarding activations and permutations for ChShuffle in Appendix F.

Table 3: Results of training LipConvnet-15 architecture on CIFAR-100. $(a, b)$ in “Groups” column denotes that we have to grouped exponential convolutions (the first one with kernel\_size = 3, the second with kernel\_size = 1). If b = 0, we have only one GS orthogonal convolutional layer. Before each grouped layer with k groups use a ChShuffle operator. 

<table><tr><td>Conv. Layer</td><td># Params</td><td>Groups</td><td>Speedup</td><td>Activation</td><td>Accuracy</td><td>Robust Accuracy</td></tr><tr><td>SOC</td><td>24.1M</td><td>-</td><td>1</td><td>MaxMin</td><td>43.15%</td><td>29.18%</td></tr><tr><td>GS-SOC</td><td>6.81M</td><td>(4, -)</td><td>1.64</td><td>MaxMinPermuted</td><td>43.48%</td><td>29.26%</td></tr><tr><td>GS-SOC</td><td>8.91M</td><td>(4, 1)</td><td>1.21</td><td>MaxMinPermuted</td><td>43.42%</td><td>29.56%</td></tr><tr><td>GS-SOC</td><td>7.86M</td><td>(4, 2)</td><td>1.22</td><td>MaxMinPermuted</td><td>42.86%</td><td>28.98%</td></tr><tr><td>GS-SOC</td><td>7.3M</td><td>(4, 4)</td><td>1.23</td><td>MaxMinPermuted</td><td>42.75%</td><td>28.7%</td></tr></table>

# 8 Concluding remarks

In this paper, we introduce a new class of structured matrices, called GS-matrices, build a structured orthogonal parametrization with them and use them in several domains within deep learning applications. However, we hope that our orthogonal parametrization can be adapted to different settings in future (including tasks outside of deep learning), as it makes orthogonal parametrizations less of a computational burden. GS-matrices without orthogonality constraints is another promising direction to consider.

# 9 Limitations

Although our method for orthogonal fine-tuning is faster than BOFT, it is still slower than LoRA during training. Additionally, since our parametrization provides a trade-off between expressivity and parameter-efficiency, it might be unable to represent some particular orthogonal matrices, which might be required in other settings apart from parameter-efficient fine-tuning.

# References

Cem Anil, James Lucas, and Roger Grosse. Sorting out lipschitz function approximation, 2019.   
Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. In International conference on machine learning, pages 1120–1128. PMLR, 2016.   
Beidi Chen, Tri Dao, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Re. Pixelated butterfly: Simple and efficient sparse training for neural network models. In International Conference on Learning Representations (ICLR), 2022.   
Tri Dao, Beidi Chen, Nimit S Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Re. Monarch: Expressive structured matrices for efficient and accurate training. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 4690–4721. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/dao22a.html.   
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in Neural Information Processing Systems, 36, 2024.   
Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J Clark, and Mehdi Rezagholizadeh. Krona: Parameter efficient tuning with kronecker adapter. arXiv preprint arXiv:2212.10650, 2022.   
Daniel Y. Fu, Simran Arora, Jessica Grogan, Isys Johnson, Sabri Eyuboglu, Armin W. Thomas, Benjamin Spector, Michael Poli, Atri Rudra, and Christopher Ré. Monarch mixer: A simple sub-quadratic gemm-based architecture, 2023.   
Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.   
Ligong Han, Yinxiao Li, Han Zhang, Peyman Milanfar, Dimitris Metaxas, and Feng Yang. Svdiff: Compact parameter space for diffusion fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7323–7334, 2023.   
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pages 2790–2799. PMLR, 2019.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient low-rank hypercomplex adapter layers. Advances in Neural Information Processing Systems, 34:1022–1035, 2021.   
Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.   
Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V. Oseledets, and Victor S. Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6553.   
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021.   
Qiyang Li, Saminul Haque, Cem Anil, James Lucas, Roger B Grosse, and Jörn-Henrik Jacobsen. Preventing gradient attenuation in lipschitz constrained convolutional networks. Advances in neural information processing systems, 32, 2019.   
Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.353. URL https://aclanthology.org/2021.acl-long.353.   
Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, and Tuo Zhao. Loftq: Lora-fine-tuning-aware quantization for large language models, 2023.   
Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353, 2024a.   
Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, and Bernhard Schölkopf. Parameter-efficient orthogonal finetuning via butterfly factorization. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=7NzgkEdGyr.   
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.   
Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/peft, 2022.   
Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models, 2024.   
Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. Advances in neural information processing systems, 28, 2015.   
Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=K30wTdIIYc.   
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022.   
Sahil Singla and Soheil Feizi. Skew orthogonal convolutions. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 9756–9766. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/singla21a.html.   
Sahil Singla, Surbhi Singla, and Soheil Feizi. Improved deterministic 12 robustness on cifar-10 and cifar-100. arXiv preprint arXiv:2108.04062, 2021.   
Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-to-image personalization. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.   
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.   
Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–15953, 2023.   
Yifan Yang, Jiajun Zhou, Ngai Wong, and Zheng Zhang. Loretta: Low-rank economic tensor-train adaptation for ultra-low-parameter fine-tuning of large language models, 2024.   
Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.   
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices, 2017.   
Yufan Zhou, Ruiyi Zhang, Tong Sun, and Jinhui Xu. Enhancing detail preservation for customized text-to-image generation: A regularization-free approach. arXiv preprint arXiv:2305.13579, 2023.

# A Related work

Parameter-Efficient Fine-Tuning (PEFT) With the growth of model sizes, end-to-end training became unavailable for those who want to adapt powerful architectures for specific tasks, as even full fine-tuning became too expensive. This problem sparked research in the direction of parameter-efficient fine-tuning methods, including methods that focus on prompt tuning [Lester et al., 2021, Li and Liang, 2021] and adapter tuning (e.g. [Houlsby et al., 2019, Karimi Mahabadi et al., 2021]), which include LoRA [Hu et al., 2022] and its variations [Meng et al., 2024, Zhang et al., 2023, Liu et al., 2024a, Dettmers et al., 2024, Li et al., 2023], that inject learnable low-rank matrices as an additive injection to the weights of pretrained models. OFT [Qiu et al., 2023], BOFT [Liu et al., 2024b] and our method use similar approach to LoRA, but learn multiplicative injection rather than an additive one.

Structured sparsity Structured sparsity is an approach that replaces dense weight layers with different structured ones, such as matrix factorizations or tensor decompositions in order to compress or speed-up models [Dao et al., 2022, Chen et al., 2022, Novikov et al., 2015, Lebedev et al., 2015]. Some of these techniques were also adapted to PEFT methods in works like [Karimi Mahabadi et al., 2021, Edalati et al., 2022, Yang et al., 2024] or BOFT [Liu et al., 2024b] method, that utilizes a variation of butterfly matrices as a parametrization for parameter-efficient orthogonal matrices, imposing orthogonality to each butterfly factor. See details in Section 2. Monarch matrices [Dao et al., 2022, Fu et al., 2023] are most relevant to our work as our proposed matrix class is their generalization that utilizes similar structure.

Subject-driven generation The emergence of large text-to-image models [Ramesh et al., 2022, 2021, Saharia et al., 2022, Rombach et al., 2022] has propelled the advancement of personalized generation techniques in the research field. Customizing a text-to-image model to generate specific concepts based on multiple input images presents a key challenge. Various methods [Ruiz et al., 2023, Gal et al., 2022, Kumari et al., 2023, Han et al., 2023, Qiu et al., 2023, Zhou et al., 2023, Wei et al., 2023, Tewel et al., 2023] have been proposed to address this challenge, requiring either extensive fine-tuning of the model as a whole [Ruiz et al., 2023] or specific parts [Kumari et al., 2023] to accurately reconstruct concept-related training images. While this facilitates precise learning of the input concept, it also raises concerns regarding overfitting, potentially limiting the model's flexibility in generating diverse outputs in response to different textual prompts. Efforts to mitigate overfitting and reduce computational burden have led to the development of lightweight parameterization techniques [Qiu et al., 2023, Liu et al., 2024b, Hu et al., 2022, Tewel et al., 2023, Han et al., 2023] such as those proposed among others. These methods aim to preserve editing capabilities while sacrificing some degree of concept fidelity. The primary objective is to identify parameterization strategies that enable high-quality concept learning without compromising the model's ability to edit and generate variations of the concept. Our investigation indicates that the orthogonal parameterization approach we propose represents a significant step towards achieving this goal.

Orthogonal convolutions In [Li et al., 2019, Singla et al., 2021] authors discuss main issues of bounding of Lipschitz constant of neural networks and provide Gradient-Norm-Preserving (GNP) architecture in order to avoid vanishing of gradients while bounding Lipschitz constant. The authors propose a specific convolutional layer (Block Convolutional Orthogonal Parametrization) which Jacobian is orthogonal, also providing orthogonal activations with Lipschitz constant equal to 1. These constraints guarantee that the norm of the gradient will not change through backward pass. In other works [Singla and Feizi, 2021, Singla et al., 2021] authors provide a modification of the orthogonal convolutions (Skew Orthogonal Convolution) in terms of hardware-efficiency. Authors provide neural network architecture where each layer is 1-Lipschitz and make a comparison between these two convolutional layers.

# B Proof of Prop. 1

Proof. Let $R' = PR$ . $R'$ can be viewed as a block matrix with $k_{L} \times k_{R}$ blocks of sizes $b_{2}^{L} \times b_{2}^{R}$ . $L$ can be viewed as a block matrix with $k_{L} \times k_{L}$ blocks from which only diagonal are non-zero. The $A$ can be written in the following form:

$$
\left( \begin{array}{c c c} A _ {0, 0} & \ldots & A _ {0, k _ {R} - 1} \\ \vdots & \ddots & \vdots \\ A _ {k _ {L} - 1, 0} & \ldots & A _ {k _ {L} - 1, k _ {R} - 1} \end{array} \right) = \left( \begin{array}{c c c} L _ {0} & \ldots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \ldots & L _ {k _ {L} - 1} \end{array} \right) \left( \begin{array}{c c c} R _ {0, 0} ^ {\prime} & \ldots & R _ {0, k _ {R} - 1} ^ {\prime} \\ \vdots & \ddots & \vdots \\ R _ {k _ {L} - 1, 0} ^ {\prime} & \ldots & R _ {k _ {L} - 1, k _ {R} - 1} ^ {\prime} \end{array} \right).
$$

Using block matrix product formulas, we get:

$$
A _ {k _ {1}, k _ {2}} = L _ {k _ {1}} R _ {k _ {1}, k _ {2}} ^ {\prime}.
$$

We can now rewrite $L_{k_1}R_{k_1,k_2}'$ product in terms of their columns and rows:

$$
L _ {k _ {1}} R _ {k _ {1}, k _ {2}} ^ {\prime} = \left(l _ {1} \dots l _ {b _ {L} ^ {2}}\right) \cdot \left( \begin{array}{c} r _ {1} ^ {\top} \\ \vdots \\ r _ {b _ {L} ^ {2}} ^ {\top} \end{array} \right) = \sum_ {t} l _ {t} r _ {t} ^ {\top}. \tag {4}
$$

Columns of $L_{k_1}$ are just vectors $u_j$ such that $\left\lfloor \frac{j}{k_L} \right\rfloor = k_1$ . Let us examine the rows of $R_{k_1,k_2}'$ . Since $R'$ is a matrix formed by permuting the rows of block-diagonal matrix $R$ , $R_{k_1,k_2}'$ can only contain

rows that were in the $R_{k_2}$ before permutation. Formally, this means that $R_{k_1,k_2}'$ can only contain vector-rows $v_i^T$ such that $\lfloor \frac{i}{k_R} \rfloor = k_2$ . Additionally, rows after permutation should get into the $k_1$ -block row. That implies $\lfloor \frac{\sigma(i)}{k_L} \rfloor = k_1$ . Other rows of $R_{k_1,k_2}'$ are zero-rows. Notice that in (4) non-zero rows $r_t^\top$ represented by $v_i^\top$ will match exactly with columns $u_{\sigma(i)}$ that represent $l_t$ . Keeping only non-zero terms in $\sum_t l_t r_t^\top$ gets us to the desired conclusion.

# C Comparison of Monarch matrices and GS-matrices

$\mathcal{GS}(P_L, P, P_r)$ class is inspired by Monarch matrices [Dao et al., 2022] and their primary goal is to introduce additional flexibility in the block structure of matrices $L$ and $R$ . Generalized Monarch matrices are parameterized as $P_1LP_2R$ , where $L$ and $R$ are block-diagonal matrices and $P_1$ and $P_2$ are certain permutations defined in Definition 5.2. This resembles $\mathcal{GS}(P_1, P_2, I)$ matrix class, however in comparison monarch matrices have additional hard constraints on relation between $k_L$ and $k_R$ . Being more precise, Monarch matrices are a special case of $\mathcal{GS}(P_1, P_2, I)$ matrices with additional constraints $k_L = b_R^1$ , $k_R = b_L^2$ . Such constraints lead to several theoretical and practical limitations of Monarch matrices. From theoretical point of view, Monarch matrices can only describe permuted block matrices with blocks of ranks 1. In contrast $\mathcal{GS}$ -matrices with can describe matrices with different rank structure of blocks (including structures where rank of each block is equal to arbitrary $r$ ). From practical point of view, due to this constraint Monarch matrices are often unable to form a desirable block structure of matrices $L$ and $R$ . For demonstration of this phenomena, consider a case of square matrices with square blocks – the structure needed in Orthogonal fine-tuning paradigm. Formally, we have $b_L^1 = b_L^2 = b_L$ , $b_R^1 = b_R^2 = b_R$ , $m = n$ . Additional Monarch constraint would mean that $b_R = k_L$ ; $b_L = k_R$ . This in turn means that $k_L \cdot k_R = n$ . As we can see, it makes impossible to stack two matrices with small number of blocks (say, 4) or large number of blocks, which is required in situations with low parameter budget. In contrast, $\mathcal{GS}$ parametrization allows for both of these structures, which we use in our experiments.

Note, that the work [Fu et al., 2023] provides a slightly different definition for monarch matrices, introducing order- $p$ Monarch matrices. These matrices are also a special case of $\mathcal{GS}$ class, however they are very restrictive as they can only parametrize matrices with both sides equal to $a^p$ for some integers $a, p$ .

# D Proof of Theorem 2

We use information transition framework from [Liu et al., 2024b], representing product of m sparse $d \times d$ matrices as an transmitting information in a grid with $d \times (m + 1)$ nodes. Edges between nodes j and i represent that element i, j in sparse matrix is non-zero. Element i, j from final matrix can only be non-zero if there exists a path from the j-th node from the right column to the i-th node in the left column (see Figure 5).

Proof. Consider an information transmission graph for the matrix $B_{i}P_{(r,br)}$ . In this graph, the first node connects with b first edges, the second node connects with the edges from $b+1$ to 2b and so on. Now consider a graph for the product of m such matrices. As shown in Figure 5, now each node from the first level has paths to $b^{k}$ unique nodes from the kth-th level. It means that using $m=\lceil\log_{b}(d)\rceil=\lceil\log_{b}(br)\rceil=1+\lceil\log_{b}(r)\rceil$ matrices is sufficient to reach all nodes and therefore form a dense matrix. Note that the number of paths for each node is always equal to $b^{m}$ regardless of permutation choice. This observation shows that it is impossible to reach d unique elements on the final level with $m<1+\lceil\log_{b}(r)\rceil$ . □

# E Subject-driven generation

Training details All the models are trained using Adam optimizer with batch size = 4, learning rate = 0.00002, betas = (0.9, 0.999) and weight decay = 0.01. The Stable Diffusion-2-base model is used for all experiments.

Evaluation details We use the DreamBooth dataset for evaluation. The dataset contains 25 different contextual prompts for 30 various objects including pets, toys and furnishings. For each concept

![](images/94d6ef2734375e47b82adadc3cc5cb32a5531cfe8388796d43c775876aa93fcf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    b^k --> b^2
    b^2 --> b
    b --> 1
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b --> ...b
    b <--> b^2
    b <--> b
    b <--> 1
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
    b <--> ...b
```
```
</details>

Figure 5: Demonstration of information transition through a block structure. Each node is connected to exactly b consecutive nodes from the next level.

we generate 10 images per contextual prompt and 30 images per base prompt “a photo of an S\*”, resulting in 780 unique concept-prompt pairs and a total of 8400 images for fair evaluation.

To measure concept fidelity, we use the average pairwise cosine similarity (IS) between CLIP ViTB/32 embeddings of real and generated images as in [Gal et al., 2022]. This means that the image similarity is calculated using only the base prompt, i.e. “a photo of an S\*”. Higher values of this metric usually indicate better subject fidelity, while keeping this evaluation scene-independent. To evaluate the correspondence between generated images and contextual prompts (TS), the average cosine similarity between CLIP ViTB/32 embeddings of the prompt and generated images [Ruiz et al., 2023, Gal et al., 2022].

Additional results In Figures 6 we show a graphical representation of the metrics for 1000 and 3000 iterations. Examples of generation for different methods are presented in Figure 7, 8.

# F GS Orthogonal Convolution

In this section, we provide some details and insights about the choice of the ChShuffle permutation and the activation function.

![](images/341c3da021c59a1f9e617a0de2d15132745b378cb765914d94ae436d876d7498.jpg)

<details>
<summary>scatter</summary>

| Method | Num Steps | TS Value | BaseIS |
| --- | --- | --- | --- |
| LoRA rank=4 | 1000 | 0.255 | 0.818 |
| LoRA rank=4 | 1000 | 0.260 | 0.795 |
| LoRA rank=4 | 1000 | 0.265 | 0.798 |
| LoRA rank=32 | 1000 | 0.245 | 0.825 |
| LoRA rank=32 | 1000 | 0.255 | 0.818 |
| LoRA rank=32 | 1000 | 0.265 | 0.792 |
| LoRA rank=128 | 1000 | 0.245 | 0.825 |
| LoRA rank=128 | 1000 | 0.255 | 0.818 |
| GSOFT r=8 | 1000 | 0.245 | 0.820 |
| GSOFT r=8 | 1000 | 0.255 | 0.818 |
| GSOFT r=16 | 1000 | 0.245 | 0.818 |
| GSOFT r=16 | 1000 | 0.265 | 0.805 |
| GSOFT r=32 | 1000 | 0.265 | 0.792 |
| GSOFT r=32 | 1000 | 0.265 | 0.795 |
| Double GSOFT r=16 | 1000 | 0.225 | 0.812 |
| Double GSOFT r=16 | 1000 | 0.235 | 0.795 |
| Double GSOFT r=32 | 1000 | 0.245 | 0.815 |
| Double GSOFT r=32 | 1000 | 0.255 | 0.818 |
| Double GSOFT r=64 | 1000 | 0.245 | 0.815 |
| Double GSOFT r=64 | 1000 | 0.255 | 0.818 |
| BOFT r=16, m=5 | 3000 | 0.225 | 0.795 |
| BOFT r=32, m=6 | 3000 | 0.235 | 0.798 |
| BOFT r=32, m=4 | 3000 | 0.245 | 0.815 |
| Full | 3000 | 0.215 | 0.815 |
The chart displays a scatter plot with error bars.
</details>

Figure 6: Image and text similarity visualisation for different methods on subject-driven generation.

![](images/3863cc79b0efc37952fece2dae03c9e0d173caf6402c962b367592fd0817a629.jpg)

<details>
<summary>text_image</summary>

Concept
Full
LoRA
r = 4
LoRA
r = 32
BOFT
r = 32, m = 4
BOFT
r = 32, m = 6
GSOFT
r = 32
Double GSOFT
r = 64
"a V* on a cobblestone street"
"a purple V*"
"a V* in a purple wizard outfit"
"a V* in the jungle"
"a V* in the snow"
"a V* with a blue house in the background"
"a V* in a chef outfit"
</details>

Figure 7: Subject-driven generation visual results on 3000 training iterations.

In experiments, we apply the ChShuffle operation right before grouped convolutional layers. Stacking several layers of that form resembles higher-order GS-matrices, which motivates the usage of permutations from Definition 5.2 for optimal information transition (see Appendix D). However, in the LipConvnet architecture, the activation function can also shuffle information between channels. Thus, this additional shuffling of information can negatively affect our information transition properties. In the original SOC paper [Singla and Feizi, 2021], the authors use MaxMin activation, firstly proposed in Anil et al. [2019].

Definition F.1. [Singla and Feizi, 2021] Given a feature tensor $X \in \mathbb{R}^{2m \times n \times n}$ , the $MaxMin(X)$ activation of a tensor $X$ is defined as follows:

$$
A = X _ {: m,:,:}, B = X _ {m,:,:,}
$$

$$
M a x M i n (X) _ {: m,:,:} = m a x (A, B),
$$

$$
\operatorname{MaxMin} (X) _ {m:,:,} = \min (A, B).
$$

This activation shuffles information between different groups in convolution which harms performance of our experiments, as permutations that we use in ChShuffle become sub-optimal in terms of information transmission. Thus, we introduce a modification of MaxMin activation, that splits channels into pairs in a different way. Rather than constructing pairs from different halves of input

![](images/89e4d7a9dea57d31bda51c41d5b66a3e035b0372543f59845dbb3a0a8b609520.jpg)

<details>
<summary>text_image</summary>

Concept
Full
LoRA
r = 4
LoRA
r = 32
BOFT
r = 32, m = 4
BOFT
r = 32, m = 6
GSOFT
r = 32
Double GSOFT
r = 64
"a V* on a cobblestone street"
"a purple V*"
"a V* in a purple wizzard outfit"
"a V* in the jungle"
"a V* in the snow"
"a V* with a blue house in the background"
"a V* in a chef outfit"
</details>

Figure 8: Subject-driven generation visual results on 1000 training iterations.

tensor, we use neighboring channels for forming of pairs (first channel pairs with second, third with fourth and so on). With this modification information does not transfer between groups during activations, which enables more optimal information transmission in-between layers with ChShuffle operator. In further experiments we denote this activation function as MaxMinPermuted and define it below:

Definition F.2. Given a feature map $X \in R^{2m \times n \times n}$ . MaxMinPermuted(X) is defined as follows:

$$
A = X _ {:: 2,:,:,}, B = X _ {1:: 2,:,:,},
$$

$$
M a x M i n P e r m u t e d (X) _ {:: 2, \therefore} = m a x (A, B),
$$

$$
M a x M i n P e r m u t e d (X) _ {1:: 2, \therefore ,:} = m i n (A, B)
$$

However, we also empirically find that it is crucial for the channels that interact within activations functions to also interact during convolutions. This means that they should always stay in the same group. This motivates us to use a slightly different permutation for the ChShuffle operation, which permutes channels in pairs. We use the following permutation

$$
\sigma (i) _ {(k, n)} ^ {\text { paired }} = \left(\left\lfloor \frac {i}{2} \right\rfloor \bmod k\right) \cdot \frac {n}{k} + 2 \cdot \left\lfloor \frac {i}{2 k} \right\rfloor + (i \bmod 2)
$$

This permutation can be seen as an adaptation of $P_{(k,n)}$ that operates on pairs of channels instead of single channels. This permutation is also optimal in terms of information transition. We call this permutation “paired”. Using this paired permutation as a ChShuffle with our modified activation saves connection between pairs while also transmitting information in the most efficient way. We provide the results of comparison of approaches with activations and permutations in Table 4.

Table 4: Comparison of activations on LipConvnet-15 architecture and CIFAR-100. $(a, b)$ in “Groups” column denotes that we have two grouped exponential convolutions (the first one with kernel\_size = 3, the second with kernel\_size = 1). If b is not mentioned, we have only one GS orthogonal convolutional layer. 

<table><tr><td>Conv. Layer</td><td># Params</td><td>Groups</td><td>Speedup</td><td>Activation</td><td>Permutation</td><td>Accuracy</td><td>Robust Accuracy</td></tr><tr><td>SOC</td><td>24.1M</td><td>-</td><td>1</td><td>MaxMin</td><td>-</td><td>43.15%</td><td>29.18%</td></tr><tr><td>GS-SOC</td><td>6.81M</td><td>(4, -)</td><td>1.64</td><td>MaxMinPermuted</td><td>paired</td><td>43.48%</td><td>29.26%</td></tr><tr><td>GS-SOC</td><td>6.81M</td><td>(4, -)</td><td>1.64</td><td>MaxMinPermuted</td><td>not paired</td><td>40.46%</td><td>26.18%</td></tr><tr><td>GS-SOC</td><td>6.81M</td><td>(4, -)</td><td>1.64</td><td>MaxMin</td><td>paired</td><td>37.99%</td><td>24.19%</td></tr><tr><td>GS-SOC</td><td>6.81M</td><td>(4, -)</td><td>1.64</td><td>MaxMin</td><td>not paired</td><td>39.72%</td><td>25.96%</td></tr><tr><td>GS-SOC</td><td>8.91M</td><td>(4, 1)</td><td>1.21</td><td>MaxMinPermuted</td><td>paired</td><td>43.42%</td><td>29.56%</td></tr><tr><td>GS-SOC</td><td>8.91M</td><td>(4, 1)</td><td>1.21</td><td>MaxMinPermuted</td><td>not paired</td><td>40.15%</td><td>26.4%</td></tr><tr><td>GS-SOC</td><td>8.91M</td><td>(4, 1)</td><td>1.21</td><td>MaxMin</td><td>paired</td><td>40.3%</td><td>26.74%</td></tr><tr><td>GS-SOC</td><td>8.91M</td><td>(4, 1)</td><td>1.21</td><td>MaxMin</td><td>not paired</td><td>41.7%</td><td>27.66%</td></tr><tr><td>GS-SOC</td><td>7.86M</td><td>(4, 2)</td><td>1.22</td><td>MaxMinPermuted</td><td>paired</td><td>42.86%</td><td>28.98%</td></tr><tr><td>GS-SOC</td><td>7.86M</td><td>(4, 2)</td><td>1.22</td><td>MaxMinPermuted</td><td>not paired</td><td>41.13%</td><td>27.53%</td></tr><tr><td>GS-SOC</td><td>7.86M</td><td>(4, 2)</td><td>1.22</td><td>MaxMin</td><td>paired</td><td>41.55%</td><td>27.45%</td></tr><tr><td>GS-SOC</td><td>7.86M</td><td>(4, 2)</td><td>1.22</td><td>MaxMin</td><td>not paired</td><td>41.25%</td><td>27.29%</td></tr><tr><td>GS-SOC</td><td>7.3M</td><td>(4, 4)</td><td>1.23</td><td>MaxMinPermuted</td><td>paired</td><td>42.75%</td><td>28.7%</td></tr><tr><td>GS-SOC</td><td>7.3M</td><td>(4, 4)</td><td>1.23</td><td>MaxMinPermuted</td><td>not paired</td><td>38.93%</td><td>25.59%</td></tr><tr><td>GS-SOC</td><td>7.3M</td><td>(4, 4)</td><td>1.23</td><td>MaxMin</td><td>paired</td><td>40.34%</td><td>27.06%</td></tr><tr><td>GS-SOC</td><td>7.3M</td><td>(4, 4)</td><td>1.23</td><td>MaxMin</td><td>not paired</td><td>41.57%</td><td>27.48%</td></tr></table>

It can be seen that using “paired” permutation used with MinMaxPermuted activation significantly improves quality metrics.