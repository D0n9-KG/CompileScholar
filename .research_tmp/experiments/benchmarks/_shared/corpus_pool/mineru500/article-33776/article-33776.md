# OneBatchPAM: A Fast and Frugal K-Medoids Algorithm

Antoine de Mathelin $^{1}$ , Nicolas Enrique Cecchi $^{1}$ , François Deheeger $^{2}$ , Mathilde Mougeot $^{1}$ , Nicolas Vayatis $^{1}$

$^{1}$ Centre Borelli, Université Paris-Saclay, CNRS, ENS Paris-Saclay

$^{2}$ Michelin

antoine.de\_mathelin@ens-paris-saclay.fr

# Abstract

This paper proposes a novel k-medoids approximation algorithm to handle large-scale datasets with reasonable computational time and memory complexity. We develop a local-search algorithm that iteratively improves the medoid selection based on the estimation of the k-medoids objective. A single batch of size $m \ll n$ provides the estimation, which reduces the required memory size and the number of pairwise dissimilarities computations to $\mathcal{O}(mn)$ , instead of $\mathcal{O}(n^{2})$ compared to most k-medoids baselines. We obtain theoretical results highlighting that a batch of size $m = \mathcal{O}(\log(n))$ is sufficient to guarantee, with strong probability, the same performance as the original local-search algorithm. Multiple experiments conducted on real datasets of various sizes and dimensions show that our algorithm provides similar performances as state-of-the-art methods such as FasterPAM and BanditPAM++ with a drastically reduced running time.

Code — https://github.com/antoinedemathelin/obpam

# Introduction

The k-medoids problem consists in choosing k medoids from a set of n points $X_{n}$ , minimizing the sum of the pairwise dissimilarities between the n points and their nearest medoid. This problem has many uses in machine learning, in particular for clustering, subset selection and active learning (Bhat 2014; Wei, Iyer, and Bilmes 2015; Kaushal et al. 2019; de Mathelin et al. 2021). The k-medoids problem is related to k-medians, k-means and facility location (Schubert and Rousseeuw 2021). One specificity of k-medoids is to consider generic dissimilarities (non-necessarily metric). In machine learning applications, the dissimilarity function can involve heavy computational costs, especially when computed between complex data types such as images, texts, or time series.

The k-medoids problem is a discrete optimization problem known to be NP-hard (Kariv and Hakimi 1979), for which a wide variety of approximation algorithms have been developed. Many k-medoids approximations are greedy or local-search algorithms, which improve a medoid selection sequentially by either adding or removing a medoid or swapping one medoid with another data point (Dohan, Karp, and Matejek 2015). The main local-search approach considered by the operations research communities is called PAM (Partitioning Around Medoid) (Kaufman and Rousseeuw 1987; Kaufman 1990). This algorithm starts from an initial choice of k points (potentially greedily selected) and then performs a series of "swaps". The state-of-the-art PAM algorithms are the FastPAM variants (Schubert and Rousseeuw 2021; Schubert and Lenssen 2022).

A major drawback of these approximation algorithms is the computational burden encountered for large values of n. Indeed, the main algorithms require the computation and in-memory conservation of pairwise dissimilarities between the n points, resulting in a complexity of $\mathcal{O}(n^{2})$ . Nowadays, with the rise of Big Data, and the focus on reducing computational resources, there is a strong incentive to build algorithms that overcome this $\mathcal{O}(n^{2})$ limitation.

Subsampling is a straightforward solution to reduce the number of dissimilarity calculations. The idea is to use an approximation algorithm (like PAM) on a subsample of size $m \ll n$ selected among the n data points, resulting in a reduction of the time and memory complexities from $\mathcal{O}(n^{2})$ to $\mathcal{O}(m^{2})$ . Previous works have proven that this simple approach yields appealing statistical guarantees over the approximation error for relatively small batch size m (Mishra, Oblinger, and Pitt 2001; Thorup 2005; Mettu and Plaxton 2004; Meyerson, O'callaghan, and Plotkin 2004; Huang, Jiang, and Lou 2023; Guha and Mishra 2016; Czumaj and Sohler 2007). In this category of methods, the CLARA algorithm (Clustering LARge Applications) (Kaufman 1986; Kaufman and Rousseeuw 2008) is the most commonly used. The main drawback of the subsampling approach is the loose approximation of considering only the medoid candidates in the m subsampled data points, resulting in worse clustering quality (Tiwari et al. 2020). A recent method, BanditPAM, leverages Bandit algorithms to deal with this limitation (Tiwari et al. 2020, 2023). BanditPAM keeps the n data points as potential medoid candidates but only computes the dissimilarities for data points with high medoid potential, thus reducing the number of pairwise dissimilarity computations to $\mathcal{O}(n \log(n))$ for one medoid selection or one swap step of the PAM algorithm. Although BanditPAM provides a medoid selection close to PAM (in terms of k-medoids objective), the Bandit-based framework requires the computation of new pairwise dissimilarities at each medoid selection,

which then results in computing $\mathcal{O}(Tn\log(n))$ pairwise dissimilarities, with T the number of iterations of the algorithm.

In this paper, we propose an alternative approach to address the $\mathcal{O}(n^{2})$ limitations of local-search k-medoids algorithms. To avoid computing new pairwise dissimilarities at each swap, we only compute the dissimilarities between the n data points and a single batch of size m. Our theoretical analysis shows that $m = \mathcal{O}(\log(n))$ is sufficient to guarantee similar performances as FasterPAM with strong probability. Our algorithm called OneBatchPAM provides a $\mathcal{O}(T)$ speedup of time complexity compared to BanditPAM and a $\mathcal{O}(n/\log(n))$ speedup compared to FasterPAM for similar performances. We show through several experiments, conducted on real datasets, that OneBatchPAM proposes an efficient time / objective trade-off compared to multiple k-medoids algorithms.

# Related Works

# Approximation algorithms for k-medoids

The k-medoids problem is related to facility locations, k-medians (or p-medians) and k-means problems. A detailed comparison of these problems is given in (Schubert and Rousseeuw 2021). In a nutshell, the main k-medoids particularity is to consider generic dissimilarities (non-necessarily metric as in k-medians) and to constrain the k medoids to belong to the dataset $X_{n}$ (unlike k-means). k-medoids can then be seen as a special case of the facility location problem, where at most k facilities, belonging to the set of clients, can be opened with cost zero. The metric k-medoids problem is often considered, in which case the problem is similar to k-medians over discrete metric space (Schubert and Rousseeuw 2021).

As solving the k-medoids problem is NP-hard, many algorithms have been developed to provide approximations in polynomial running time $^{1}$ (Kaufman 1990; Charikar et al. 1999; Li and Svensson 2013; Bhat 2014). A “naive” greedy approach selects the medoids sequentially by solving a 1-medoid problem at each iteration. This approach is simple to implement and yields relatively good results in practice, but its theoretical approximation error in $\mathcal{O}(n)$ is quite large (Dohan, Karp, and Matejek 2015). It can be improved to $\mathcal{O}(\log(n))$ by the reverse greedy approach that starts with n medoids and removes them one by one until reaching k medoids (Chrobak, Kenyon, and Young 2006). The most notable improvement to the greedy approach is the PAM algorithm (Kaufman and Rousseeuw 1987; Kaufman 1990). It greedily initializes the set of medoids and then performs a series of swaps from one medoid to one non-medoid that improve the total objective. In the metric case, this local search approach provides a constant approximation ratio of 5 which can be reduced to $3 + \epsilon$ when swapping multiple medoids at each iteration (Arya et al. 2001). Assuming pairwise dissimilarities are precomputed, the time complexity of the seminal PAM algorithm is $\mathcal{O}(Tkn^{2})$ , with T the number of swap steps. A notable recent improvement, called FastPAM (Schubert and Rousseeuw 2021; Schubert and Lenssen 2022), reduces the PAM's time complexity to $\mathcal{O}(Tn^2)$ by using a smart decomposition of the swap evaluation (Schubert and Rousseeuw 2021). We emphasize that the PAM algorithm and its variants are perhaps the most widespread approximation algorithms for $k$ -medoids. It provides an appealing trade-off between approximation error and time complexity. Although its theoretical approximation ratio is 5, the error is often much smaller in practical use-cases (less than $2\%$ (Schubert and Rousseeuw 2021)).

When the dissimilarity evaluation is costly, and/or when the available memory is restricted. All aforementioned algorithms are limited by the $\mathcal{O}(n^{2})$ pairwise dissimilarities computation cost and by the $\mathcal{O}(n^{2})$ memory requirement to store the computed dissimilarities. Our work then focuses on reducing the time complexity of PAM while keeping similar performance. We therefore do not consider algorithms that propose improvement over the PAM performance at the price of additional computational efforts, such as (Li and Svensson 2013; Byrka et al. 2017; Ren, Hua, and Cao 2022).

# Subsampling Methods

Subsampling consists in performing a $k$ -medoids algorithm on a subsample $\mathcal{X}_m$ of the original dataset $\mathcal{X}_n$ , of size $m \ll n$ . For instance, the CLARA algorithm (Kaufman and Rousseeuw 2008) uses PAM on a subsample $\mathcal{X}_m$ . It has been shown that a $k$ -medoids algorithm with constant approximation can be derived, with great probability, using a random uniform subsample of size $m \simeq \mathcal{O}(k \log(n))$ (Mishra, Oblinger, and Pitt 2001). The required size of the subsample has been further reduced to $\mathcal{O}(k \log(k))$ with deeper analysis (Meyerson, O'callaghan, and Plotkin 2004; Czumaj and Sohler 2007), which is independent of $n$ . In this perspective, the CLARA algorithm proposes the heuristic $m = 40 + 2k$ for the subsample's size (Kaufman and Rousseeuw 2008). With such a setting, this subsampling method can drastically reduce the number of pairwise dissimilarity computations from $\mathcal{O}(n^2)$ to $\mathcal{O}(k^2)$ . CLARA repeatedly computes a $k$ -medoids approximation over multiple random subsamples drawn uniformly from $\mathcal{X}_n$ and selects the best set of $k$ medoids based on the evaluation over the whole dataset $\mathcal{X}_n$ . Although only $\mathcal{O}(k^2)$ dissimilarity computations are needed to perform PAM over the subsample, one evaluation step requires to compute $nk$ dissimilarities, resulting in a $\mathcal{O}(Tpnk)$ time complexity, with $T$ the number of subsamples. The cost of the evaluation step can be mitigated by evaluating the medoid set over another subsample from $\mathcal{X}_n$ (Meyerson, O'callaghan, and Plotkin 2004), in the same spirit as holdout validation in machine learning.

The primary drawback of the subsampling approach is the approximation error, which is theoretically twice as large as that of performing the same $k$ -medoids approximation on the full dataset of $n$ data points (Mishra, Oblinger, and Pitt 2001; Meyerson, O'callaghan, and Plotkin 2004; Czumaj and Sohler 2007). In practice, this leads to a noticeable decline in performance.

A recent method BanditPAM (Tiwari et al. 2020, 2023) proposes an interesting idea based on Bandit evaluation of the swap and initialization steps in PAM. At each step, the best local improvement is estimated using multi-armed ban-

dit techniques. BanditPAM therefore does not need to compute all pairwise dissimilarities but only the ones useful to find the best swap. The drawback of such an approach is to compute new dissimilarities at each step resulting in $\mathcal{O}((T+k)n\log(n))$ dissimilarity computations, with T the number of swap evaluations. In this work, we propose to instead compute all pairwise distance between the n data points and a batch of size $m = \mathcal{O}(\log(n))$ . The same dissimilarities are used to evaluate all swap steps, resulting in $\mathcal{O}(n\log(n))$ dissimilarity computations.

# k-means++ as a proxy for k-medoids

k-means++ is first designed as a seeding algorithm for k-means. It iteratively samples data points from $X_{n}$ with a probability proportional to the distance raised to the power p to the already sampled points for any $\ell_{p}$ distance. As the output of k-means++ is a set of cluster centers in $X_{n}$ , and since the objective of k-means is the same as k-medoids when considering the Euclidean distance, this algorithm can be used as a natural proxy for k-medoids. The k-means++ algorithm provides a $\mathcal{O}(\log(k))$ -approximation for k-means and k-medians (Arthur, Vassilvitskii et al. 2007), which is generally worse than PAM but it only require $\mathcal{O}(kn)$ pairwise dissimilarity computations instead of $\mathcal{O}(n^{2})$ .

Since the seminal work of (Arthur, Vassilvitskii et al. 2007), two primary directions have been pursued to improve $k$ -means++: enhancing the approximation error and reducing the time complexity. To improve the approximation, local-search algorithms are employed. These algorithms typically involve a random selection process similar to $k$ -means++, followed by a swap if the new selection yields a better clustering outcome. For instance, single-swap local-search methods require $\mathcal{O}((Z + k)n)$ pairwise distance computations, with $Z$ the number of swap steps, and $\mathcal{O}(Zkn)$ additional operations (Lattanzi and Sohler 2019). Multiple-swap approaches can further refine the clustering but at a higher computational cost, involving $\mathcal{O}((Zt + k)n)$ distance computations and $\mathcal{O}(Znk^{2t - 1})$ additional operations, with $t$ the number of simultaneous swaps (Beretta et al. 2024; Huang et al. 2024). To accelerate the process, (Bachem et al. 2016) introduced kmc2, which speeds up $k$ -means++ to $\mathcal{O}(Lk^2)$ distance computations, with $L$ a method's specific parameter. Other methods leverage the specificity of Euclidean distance. For example, by projecting the data onto one dimension (Charikar et al. 2023), or leveraging specific nearest neighbor structure (Cohen-Addad et al. 2020) (Pelleg and Moore 1999).

# Coreset for k-medians

According to (Feldman 2020), a coreset is a data summarization technique that selects a subsample from a large dataset, preserving the information needed to perform specific tasks such as linear regression or clustering. Specifically, given a dataset $X_{n}$ , an objective function L, and a set of queries Q, a coreset $X_{m}$ is a subsample of $X_{n}$ for which any query $q \in Q$ yields a similar objective value when computed on the coreset as when computed on the entire dataset, i.e., $\mathcal{L}(q, \mathcal{X}_{n}) \simeq \mathcal{L}(q, \mathcal{X}_{m})$ . In the context of k-medians clustering, the clustering cost of any k centers computed on a coreset is approximately the same as the cost of these centers on the entire dataset.

While there is no consensual definition of a coreset, it generally refers to a “strong coreset” where the objective computed on the coreset is a $(1+\epsilon)$ -approximation of the objective computed over the entire dataset for any query (e.g., any set of k centers). Coreset is sometimes equated with subsampling when the set of queries is restricted to the coreset itself (Huang, Jiang, and Lou 2023; Har-Peled and Mazumdar 2004). When the set of queries includes any combination of k centers from the entire dataset, coresets are similar in spirit to OnebatchPAM. However, the coreset literature focuses on constructing sets that provide a $(1+\epsilon)$ -approximation for any query, whereas OnebatchPAM focuses on achieving results comparable to PAM.

Various coresets for $k$ -medians have been proposed, aiming to find the minimal size that guarantees the $(1 + \epsilon)$ -approximation for any set of $k$ centers (Har-Peled and Mazumdar 2004; Chen 2009). The best-known result for discrete metric $k$ -medians (similar to metric $k$ -medoids) is provided by (Feldman 2020), with $m = \mathcal{O}(k\log(n)\epsilon^{-2})$ . However, constructing such a coreset has a running time of $\mathcal{O}(pnk)$ , with $p$ the data dimension. This has been improved by (Cohen-Addad, Saulpic, and Schwiegelshohn 2021b), which provides a similar-sized coreset with a running time of $\mathcal{O}(nk)$ . The coreset size $m = \mathcal{O}(k\log(n)\epsilon^{-2})$ has been shown to be the minimal size to guarantee the $(1 + \epsilon)$ -approximation (Cohen-Addad et al. 2022).

This size can be reduced when the constraints are relaxed, leading to what is known as a weak coreset (Feldman and Langberg 2011). A weak coreset guarantees the $(1 + \epsilon)$ -approximation for only a subset of queries (Feldman 2020; Jaiswal and Kumar 2024). Other definitions of coresets include those with additive and multiplicative error approximations, such as lightweight coresets (Bachem, Lucic, and Krause 2018). Smaller coresets yielding the $(1 + \epsilon)$ -approximation guarantee can be constructed when considering Euclidean space (Cohen-Addad, Saulpic, and Schwiegelshohn 2021a; Feldman and Langberg 2011), or constrained problems, such as capacitated clustering (uniform distribution between clusters) (Huang, Jiang, and Lou 2023; Braverman et al. 2022) and fair constraint clustering (Schmidt, Schwiegelshohn, and Sohler 2020).

# From PAM to OneBatchPAM

# Notations

The four parameters $n, k, p, m \in \mathbb{N}^*$ respectively denote the number of data points, the number of medoids, the problem dimension and the batch size. We consider the space $\mathcal{X}$ with $\mathcal{X} \subset \mathbb{R}^p$ and $d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_+$ a measure of dissimilarity over $\mathcal{X}$ . We consider the set $\mathcal{X}_n = \{x_1, ..., x_n\}$ of $n$ data points in $\mathcal{X}$ . We denote $\mathcal{P}_k(\mathcal{X}_n)$ the set of all subsets of $\mathcal{X}_n$ of size $k$ . We aim at solving the $k$ -medoids selection problem:

$$
\min _ {\mathcal {M} \in \mathcal {P} _ {k} (\mathcal {X} _ {n})} \sum_ {i = 1} ^ {n} d (x _ {i}, \mathcal {M}), \tag {1}
$$

with $d(x_{i},\mathcal{M}) = \min_{\tilde{x}\in \mathcal{M}}d(x_{i},\tilde{x})$ . We denote by $\mathcal{L}$ the objective function, such that $\mathcal{L}(\mathcal{M}) = \frac{1}{n}\sum_{x\in \mathcal{X}_n}d(x,\mathcal{M})$

for any $\mathcal{M} \in \mathcal{P}_{k}(\mathcal{X}_{n})$ . In the following, we consider the common assumption that one dissimilarity computation requires $\mathcal{O}(p)$ time complexity (Tiwari et al. 2020; Schubert and Rousseeuw 2021).

# PAM, FastPAM and FasterPAM

The local-search approximation algorithm called PAM, performs several “swap” steps that progressively improve the medoid selection. This is formally described by the following recurrence equation:

$$
\mathcal{M}_{t + 1} = \operatorname *{argmin}_{\substack{x\in \mathcal{M}_{t},\\ x^{\prime}\in \mathcal{X}_{n}\setminus \mathcal{M}_{t}}}\sum_{i = 1}^{n}d\left(x_{i},(\mathcal{M}_{t}\setminus \{x\})\cup \{x^{\prime}\}\right). \tag{2}
$$

For any $t \in \{0, ..., T - 1\}$ , with T the number of iterations. In the original PAM algorithm, the initial medoid set $M_{0}$ is built using a greedy algorithm (Kaufman 1990). The swap step described in Equation (2) consists in removing one medoid from $M_{t}$ and adding a non-medoid from $X_{n} \setminus M_{t}$ . This algorithm theoretically provides a 5-approximation (Arya et al. 2001). In practical scenarios, however, the approximation error is often below 2% (Schubert and Rousseeuw 2021).

As highlighted by Equation (2) the “naive” approach to perform a swap step requires computing the sum of dissimilarities for every swap pair $(x, x') \in \mathcal{M}_t \times \mathcal{X}_n \setminus \mathcal{M}_t$ , which leads to $\mathcal{O}(kn^2)$ operations to perform one swap. The FastPAM algorithm introduced in (Schubert and Rousseeuw 2021) proposes a modification of PAM that yields a $\mathcal{O}(k)$ speed up. The main idea lies in the fact that for each $x_i$ , only the removal of the nearest medoid will modify the value of $d(x_i, \mathcal{M}_t)$ . Therefore, only one pass through $X_n$ is needed to compute the impact of removing one medoid for all k medoids. The complexity of one swap step then only requires $\mathcal{O}(n^2)$ operations. Moreover, (Schubert and Rousseeuw 2021) shows that random initializations of the medoids lead to similar results as the greedy initialization but save $\mathcal{O}(kn^2)$ operations. Finally, additional speedups are derived by eagerly swapping a medoid with a non-medoid as soon as an improvement is found. In theory, eager swapping still requires $\mathcal{O}(n^2)$ operations for one swap but, in practice, it significantly speeds up the algorithm. The FastPAM algorithm with these additional improvements is called FasterPAM.

As noticed by (Tiwari et al. 2020), the main drawback of FastPAM and FasterPAM is that they require to compute every dissimilarity between each pair of data points in $X_{n}$ , with complexity $\mathcal{O}(pn^{2})$ . A solution proposed by (Schubert and Rousseeuw 2021) is FasterCLARA which uses FasterPAM on subsamples of $X_{n}$ . However, this solution comes with large approximation error in practice. To overcome this issue, we propose the OneBatchPAM algorithm.

# OneBatchPAM

The OneBatchPAM idea is the following: for any x, it is not necessary to compute every distance to every $x_{i}$ to perform the exact same swaps as FasterPAM. An estimation of the objectives on a subsample is sufficient. Theorem 1 will show that only a subsample of size $m = \mathcal{O}(\log (n))$ is needed to find the same series of swaps as FasterPAM with great probability.

Formally, OneBatch involves choosing a subsample $\mathcal{X}_m = \{x_{\sigma(1)},\dots,x_{\sigma(m)}\}$ drawn from $\mathcal{X}_n$ , with $\sigma :[1,m]\to$ $[1,n]$ the mapping indice function. The subsample $\mathcal{X}_m$ is used to estimate the best swap to perform, such that:

$$
\mathcal {M} _ {t + 1} = \underset { \begin{array}{c} x \in \mathcal {M} _ {t}, \\ x ^ {\prime} \in \mathcal {X} _ {n} \backslash \mathcal {M} _ {t} \end{array} } {\operatorname{argmin}} \sum_ {j = 1} ^ {m} d \left(x _ {\sigma (j)}, \left(\mathcal {M} _ {t} \backslash \{x \}\right) \cup \left\{x ^ {\prime} \right\}\right). \tag {3}
$$

Compared to Equation (2), the sum is now only computed over $X_{m}$ . This modification drastically reduces the time complexity while keeping similar performances as Faster-PAM with high probability as proven in Theorem 1 and Corollary 2.

It must be underlined that Equation (3) is not equivalent to subsampling as the search space is still $X_{n}$ . In subsampling methods, such as CLARA, we would have $x' \in X_{m} \setminus M_{t}$ instead of $x' \in X_{n} \setminus M_{t}$ . This difference has a significant impact on the approximation error. By reducing the search space to $X_{m} \setminus M_{t}$ , subsampling methods multiply by two the theoretical approximation error and, in practice, degraded performances are indeed observed.

Theorem 1. Let $\mathcal{X}_m$ be a subsample uniformly drawn from $\mathcal{X}_n$ . Let $D = \max_{(x,x') \in \mathcal{X}_n} d(x, x')$ and $\Delta$ be the smallest difference between two objectives computed by FasterPAM. Then, for any $\delta \in]0,1]$ , the OneBatchPAM algorithm returns the same set of medoid as FasterPAM with probability at least $1 - \delta$ if:

$$
m \geq \frac {4 D ^ {2}}{\Delta^ {2}} \log \left(\frac {2 T n}{\delta}\right). \tag {4}
$$

Where $\Delta = \min_{t\in [[0,T]]}\min_{\substack{x\in \mathcal{M}_{t},\\ x^{\prime}\in \mathcal{X}_{n}\setminus \mathcal{M}_{t}}}| \mathcal{L}(\mathcal{M}_{t}) - \mathcal{L}(\mathcal{M}_{t}\setminus \{x\} \cup \{x^{\prime}\})|$

Proof. The proof follows the same framework as the proof of Theorem 1 in (Tiwari et al. 2020). It consists in finding the minimal sample size which guarantees that the statistical error on the objectives remains smaller than the smallest objective difference, $\Delta$ , with high probability. Consequently, OneBatchPAM performs the same swaps as FasterPAM. The detailed proof is reported in the supplementary materials.

As stated by Theorem 1, the dependence of m with respect to n is only $m = \mathcal{O}(\log(n))$ . This implies a drastic reduction of the time complexity as formally described in the following corollary.

Corollary 2. The OneBatch PAM algorithm returns the same set of medoids as FasterPAM with arbitrarily high probability with time complexity:

$$
\mathcal {O} \left((p + T) n \log (n)\right). \tag {5}
$$

Table 1 provides a detailed comparison of OneBatchPAM's complexity against other algorithms. OneBatchPAM achieves a complexity gain of $\mathcal{O}(n / \log (n))$ over FasterPAM due to subsampling, and at least a $\mathcal{O}(T)$ improvement

over BanditPAM++, as it avoids computing new dissimilarities at each swap step. While OneBatchPAM may require more computational time compared to subsampling and k-means++, it offers a superior approximation error factor. It is important to note that the values in Table 1 are theoretical; in practical scenarios, the performance comparison between methods can vary. For example, the approximation errors for PAM-based algorithms are often significantly lower than 5.

<table><tr><td>Algorithm</td><td>Complexity</td><td>Approximation</td></tr><tr><td>FasterPAM</td><td> $(p + T)n^{2}$ </td><td>5</td></tr><tr><td>BanditPAM++</td><td> $p(T + k)n\log(n)$ </td><td>5</td></tr><tr><td>OneBatchPAM</td><td> $(p + T)n\log(n)$ </td><td>5</td></tr><tr><td>FasterCLARA</td><td> $I((p + T)k^{2} + pkn)$ </td><td>10</td></tr><tr><td>k-means++</td><td> $pkn$ </td><td> $\log(k)$ </td></tr></table>

Table 1: Summary of theoretical time complexity and approximation error. T is the number of swaps iterations and I the number of subsamples.

How many iterations T are needed? Generally, the larger the value of T, the better the objective, but this also increases the time complexity. It is important to note that the algorithms may terminate before reaching T swaps if a local minimum is attained. According to (Tiwari et al. 2023) and (Schubert and Rousseeuw 2021), in practice, the required number of swaps is typically $\mathcal{O}(k)$ . If a threshold $\epsilon$ on the improvement is set instead of a maximum number of iterations, such that the algorithm terminates when no swap is $1 - \epsilon$ better than the current medoid selection, then the number of swaps is at most $T = \mathcal{O}(\log(n)/\epsilon)$ .

How $X_{m}$ should be sampled? Theorem 1 demonstrates that uniform sampling is sufficient to obtain good guarantees with a relatively small subset. However, a natural question arises: can we improve this with a more specific selection method? One initial approach consists in modifying the dissimilarity between the subsampled points and themselves as follows: $d(x_{\sigma(j)}, x_{\sigma(j)}) = +\infty$ for any $j \in \{1, \ldots, m\}$ . We empirically observed that this adjustment prevents the medoid selection from being biased toward the subsampled data points. A second approach is to reweight the uniform sample to correct any potential sample bias. Since all distances between $X_{n}$ and $X_{m}$ are computed to perform OneBatchPAM, we recommend using the nearest neighbor sample bias correction method from (Loog 2012). In this method, the importance of the data point $x_{\sigma(j)}$ is proportional to the number of data points in $X_{n}$ whose nearest neighbor in $X_{m}$ is $x_{\sigma(j)}$ . Additionally, specific sampling techniques, such as those used to build coresets, may also be considered (Bachem, Lucic, and Krause 2018).

# Discussion and Limitations

Minimum sample size of OneBatchPAM derived in Theorem 1. The factor $1/\Delta$ in the sample size lower bound also appears in the theoretical time complexity of BanditPAM. It is implicitly assumed that the minimum objective difference, $\Delta$ , is not null (Tiwari et al. 2020). The inverse proportionality between $m$ and $\Delta^2$ indicates that OneBatchPAM may require a large subsample to perform the exact same swaps as FasterPAM if two objectives are close. This can happen if two data points $x, x' \in \mathcal{X}_n$ are close. In that case, OneBatchPAM may estimate that adding $x$ to the set of medoids instead of $x'$ is more efficient while FatserPAM may do the opposite. However, as the difference between both objectives is small, OneBatchPAM will likely return a set of medoids with close performance to the one of FasterPAM. This is confirmed in our empirical experiments where OneBatchPAM provides close objectives compared to FasterPAM (around 2% error) but not exactly the same. We emphasize that the purpose of Theorem 1 is essentially to highlight the dependence of $m$ relative to $n$ . Indeed, many upper bound approximations are involved in the derivation of the Theorem's result, hence using the exact value of Equation (4) for $m$ may be disproportionate. In practice, we do not estimate the ratio $D / \Delta$ to set the sample size, but instead choose a value proportional to $\log(n)$ .

It is interesting to notice that the minimum sample size for OneBatchPAM does not directly depend on the number of medoids k. However, this dependence is somehow hidden in the number of swap steps T. As highlighted by (Schubert and Rousseeuw 2021), when starting with a random medoid selection, one can expect at least k swaps before reaching a local minimum.

Comparison to BanditPAM and memory limitations of OneBatchPAM. Both BanditPAM and OneBatchPAM rely on the estimation of the k-medoids objective to determine which swap to perform. However, they consider two different approaches for estimating this objective. BanditPAM gradually improves the objective's estimation of swap pair candidates using mini-batches while reducing the set of candidates as the estimation becomes more accurate. The process is repeated after each swap, as the update of the medoid set modifies the swap pairs' evaluation. This leads to a linear increase in pairwise dissimilarity computations relative to the number of iterations. In contrast, OneBatchPAM computes all pairwise dissimilarities between the entire dataset $X_{n}$ and a subsample $X_{m}$ only once, using these precomputed values for each swap step. Consequently, it avoids the linear scaling of dissimilarity computations with the number of iterations. It should be noted that this computational load reduction comes with an increase in memory consumption. Indeed, BanditPAM only requires $\mathcal{O}(n)$ memory space while OneBatchPAM needs $\mathcal{O}(n\log(n))$ . Nevertheless, this memory usage is significantly more efficient than the $\mathcal{O}(n^{2})$ memory requirement of FasterPAM.

Comparison to coresets. As discussed in the related works section, OneBatchPAM is closely associated with coresets used in the context of k-medians. The coresets literature essentially focuses on constructing subsets that provide a $(1 + \epsilon)$ -approximation for any k-medoids selection. This imposes a stronger constraint compared to OneBatchPAM, which focuses on achieving results similar to those of the PAM algorithm. This explains why the minimal sample size for OneBatchPAM $m = \mathcal{O}(\log(n))$ is smaller than the minimal size for coresets for k-medians clustering with discrete metric spaces, $m = \mathcal{O}(k \log(n)\epsilon^{-2})$ (Cohen-Addad

et al. 2022). It is important to note, however, that the sample size for OneBatchPAM is derived from a uniform sample $X_{m}$ . Leveraging coreset construction techniques could potentially further reduce the required sample size and, consequently, the time complexity of OneBatchPAM.

Overfitting for highly imbalanced datasets. Overfitting is a potential risk for OneBatchPAM, especially when the batch is not representative of the full dataset. Overfitting issues especially arise in situations involving highly imbalanced datasets. For instance, if a small subset of points are very far from all others. In such a case, there is a low probability that any neighbors of these distant points will be included in the batch, potentially leaving these points “not covered” by any medoid at the end of the OneBatchPAM algorithm. A potential future improvement to our approach could be to construct the batch progressively, leveraging the computed distances to identify imbalances in the dataset and mitigate the issue by selecting data points that improve the “representativeness” of the batch.

# Experiments

We conduct several experiments on real datasets to compare OneBatchPAM with state-of-the-art k-medoids algorithms in practical scenarios. Our implementation of OneBatchPAM is coded in Python with the Cython module. The experiments are run on a 8G RAM computer with 4 cores. The source code of the experiments is available on GitHub $^{2}$ .

# Datasets and settings

We conduct the experiments on the MNIST and CIFAR10 image datasets (LeCun, Cortes, and Burges 1994; Krizhevsky, Hinton et al. 2009) and 8 UCI datasets (Dua and Graff 2017), arbitrarily selected, with various sizes and dimensions (cf. Table 2). The $\ell_1$ distance is used as the dissimilarity function. Experiments are performed for different values of $k$ in $\{10, 50, 100\}$ . Each experiment is repeated 5 times to compute the standard deviations.

We divide the datasets into two categories respectively called “small scale” and “large scale” to account for the fact that some algorithms cannot provide a medoid selection in reasonable time for datasets above $\sim$ 50000 instances. In particular, FasterPAM is not able to handle the size of the MNIST dataset (Schubert and Rousseeuw 2021).

<table><tr><td colspan="3">Small Scale</td><td colspan="3">Large Scale</td></tr><tr><td>Dataset</td><td>n</td><td>p</td><td>Dataset</td><td>n</td><td>p</td></tr><tr><td>abalone</td><td>4,176</td><td>8</td><td>CIFAR</td><td>50,000</td><td>3072</td></tr><tr><td>bankruptcy</td><td>6,819</td><td>96</td><td>MNIST</td><td>60,000</td><td>784</td></tr><tr><td>mapping</td><td>10,545</td><td>28</td><td>dota2</td><td>92,650</td><td>117</td></tr><tr><td>drybean</td><td>13,611</td><td>16</td><td>gas</td><td>416,153</td><td>9</td></tr><tr><td>letter</td><td>19,999</td><td>16</td><td>covertype</td><td>581,011</td><td>55</td></tr></table>

Table 2: Datasets Summary. n and p are respectively the dataset's size and dimension.

# Competitors and Hyper-parameters

The following two kinds of competitors are considered

- PAM Algorithms. we consider the PAM variants: FasterPAM, BanditPAM++ and FasterCLARA, as well as the Alternate approach (Park and Jun 2009) although it is not formally a PAM method. We use the official implementations of BanditPAM++ $^{3}$ (Tiwari et al. 2023). The other algorithms are found in the Python library kmedoids $^{4}$ , providing the official implementation of FasterPAM (Schubert and Lenssen 2022).   
- $k$ -means++ Algorithms. We consider the original $k$ -means++ algorithms and the two variants introduced in the related works: kmc2 (Bachem et al. 2016) and $k$ -means++ with local-search (LS-k-means++) (Lattanzi and Sohler 2019).

If nothing else is specified the default hyperparameters are selected for the method. For BanditPAM++, we consider the three different settings of swap iterations: $T \in \{0, 2, 5\}$ . We noticed that larger values of this parameter lead to excessive running time. For FasterCLARA we consider two different settings for the number of subsampling repetitions: $I \in \{5, 50\}$ . The sample size is set to $m = 80 + 4k$ as suggested in (Schubert and Rousseeuw 2021). Three different chain lengths are considered for kmc2: $L = \{20, 100, 200\}$ and two different number of local search iterations for LS-k-means++: $Z = \{5, 10\}$ . When different values of a parameter P are used for an algorithm Alg, we denote the corresponding variants by Alg-P.

For OneBatchPAM, we use a sample size of $m = 100 \log(kn)$ . The four following subsampling techniques introduced in Section are considered: Unif: uniform sampling; Debias: uniform sampling with $d(x_{\sigma(j)}, x_{\sigma(j)}) = +\infty$ for any $j \in \{1, \ldots, m\}$ ; NNIW: uniform sampling with nearest-neighbor importance weighting and LWCS: sample built through the “lightweight coreset” technique from (Bachem, Lucic, and Krause 2018).

# Results

The methods are compared in terms of both objective value and computational time. To provide a normalized measure between datasets, we consider the “delta relative objective” ( $\Delta RO$ ) and “relative time” (RT), defined for any algorithm A as follows:

$$
\Delta \mathrm{RO} (\mathcal {A}) = \frac {\mathcal {L} (\mathcal {M} ^ {\mathcal {A}})}{\mathcal {L} (\mathcal {M} ^ {\mathcal {A} ^ {*}})} - 1; \mathrm{RT} (\mathcal {A}) = \frac {T (\mathcal {A})}{T (\mathcal {A} ^ {*})}. \tag {6}
$$

Where $M^{A}$ is the set of medoids selected by algorithm A, $A^{*}$ refers to the algorithm providing the best objective.

Evolution of the objective and running time for different values of n and k. Figure 1 shows the objective and running time of five algorithms for different $(k, n)$ settings on the MNIST dataset. In each graph, OneBatchPAM ranks among the best methods both in terms of objective and running time. The time evolution of OneBatchPAM is similar

![](images/9bfe737b02acaa028423c67fbed7f15a983cacf73d5049b935c24167dc4488fc.jpg)

<details>
<summary>line</summary>

| n     | KM   | BP   | FP   | OBP  | FC   |
|-------|------|------|------|------|------|
| 0     | 0    | 0    | 0    | 0    | 0    |
| 10000 | 50   | 75   | 100  | 10   | 5    |
| 20000 | 150  | 175  | 300  | 25   | 10   |
</details>

![](images/feb95f299c0f29782421aa7f877b805c74bc24aa77e502135468229fd02f27c4.jpg)

<details>
<summary>line</summary>

| n      | Objective (Blue) | Objective (Green) | Objective (Red) |
| ------ | ---------------- | ----------------- | --------------- |
| 0      | 95               | 85                | 80              |
| 10000  | 90               | 85                | 80              |
| 20000  | 93               | 85                | 80              |
</details>

![](images/6231d29da4bcaeb344f9ec7842269e1d60d3468c48cdc0437ba2516fed8dbfe7.jpg)

<details>
<summary>line</summary>

| k   | Time (s) |
| --- | -------- |
| 0   | 100      |
| 50  | 75       |
| 100 | 75       |
| 150 | 75       |
| 200 | 75       |
</details>

![](images/7309e48b03dde1ba1028780836e7ca19bfe3ac6ccb5380e8c7b2f4d82be1c37e.jpg)

<details>
<summary>line</summary>

| k   | Series 1 | Series 2 | Series 3 | Series 4 |
| --- | -------- | -------- | -------- | -------- |
| 0   | 100      | 85       | 80       | 75       |
| 50  | 85       | 75       | 70       | 65       |
| 100 | 75       | 65       | 60       | 55       |
| 150 | 70       | 60       | 55       | 50       |
| 200 | 65       | 55       | 50       | 45       |
</details>

Figure 1: Evolution of the running time and objective on the MNIST dataset. Left: evolution as a function of n for k = 10. Right: evolution as a function of k for n = 10000. The results for five competitors are reported: k-means++ (KM), FasterPAM (FP), FasterCLARA-5 (FC), BanditPAM++-2 (BP), OneBatchPAM (OBP)

to the one of k-means++ and FasterCLARA-5, and significantly smaller than BanditPAM++ and FasterPAM, especially for large values of n. Additionally, the objective evolution of OneBatchPAM closely matches that of FasterPAM, while FasterCLARA-5 and k-means++ provide larger objective values.

Aggregated Results. Table 3 presents the averaged results over the three values of $k \in \{10, 50, 100\}$ , the five repetitions of the experiments and the respective five “small scale” and “large scale” datasets. The detailed results per dataset and value of k are reported in the supplementary materials. As expected, FasterPAM provides the best objective and Random the fastest medoid selection for the small scale experiments. The OneBatchPAM variants reduce the computational burden of FasterPAM by a factor of 7 on average (RT = 15%) for a small penalization of the objective value (1.7% compared to FasterPAM for the NNIW variant). This observation highlights the efficiency of OneBatchPAM to provide a fast and accurate medoid selection. Notice that the time reduction factor increases with the number of samples. The relative time for OneBatchPAM is equal to 8.5% for the letter dataset, which corresponds to a reduction factor of around 12 (cf. detailed results in supplementary materials). We observe that k-means++ and FasterCLARA-5 are faster than OneBatchPAM (by a factor of around 7 for FasterCLARA-5). However, the running time reduction comes with a significant penalization of the objective: respectively 13% and 30% for FasterCLARA-5 and k-means++. For large scale datasets, FasterPAM and BanditPAM++ fail to provide medoid selections within reasonable computational times, positioning OneBatchPAM as the method with the best objective ( $\Delta RO = 0$ ). Similar to the small-scale experiments, FasterCLARA-5 is 7 times faster than OneBatch but is 8% worse in terms of objective. kmc2 is even faster, however, its objective is close to the random selection’s objective.

Regarding the OneBatchPAM variants, we observe that debiasing offers a modest improvement compared to uniform sampling ( $\sim 0.2\%$ ). The gain is higher for large values of k (around 1%), as highlighted in the detailed results. While the LWCS method degrades performance (likely because LWCS is primarily designed to provide strong theoretical guarantees for k-means++ rather than PAM), the NNIW variant shows significant objective improvements (above 1.2%) over uniform sampling with comparable computational time. This observation supports the systematic use of nearest neighbor importance weighting in OneBatchPAM. Indeed, the pairwise dissimilarities needed to compute the importance weights are also required by the OneBatchPAM core algorithm, which explains why using NNIW has a negligible impact on the running time.

<table><tr><td rowspan="2">Method</td><td colspan="2">Small Scale</td><td colspan="2">Large Scale</td></tr><tr><td>RT</td><td> $\Delta RO$ </td><td>RT</td><td> $\Delta RO$ </td></tr><tr><td>Random</td><td>0.0</td><td>62.9</td><td>0.0</td><td>20.3</td></tr><tr><td>FasterPAM</td><td>100.0</td><td>0.0</td><td>Na</td><td>Na</td></tr><tr><td>Alternate</td><td>161.1</td><td>20.0</td><td>Na</td><td>Na</td></tr><tr><td>FasterCLARA-5</td><td>2.8</td><td>13.0</td><td>15.0</td><td>8.0</td></tr><tr><td>FasterCLARA-50</td><td>30.0</td><td>10.9</td><td>161.7</td><td>7.1</td></tr><tr><td>kmc2-20</td><td>14.5</td><td>31.3</td><td>0.5</td><td>18.2</td></tr><tr><td>kmc2-100</td><td>72.2</td><td>31.9</td><td>2.4</td><td>17.6</td></tr><tr><td>kmc2-200</td><td>153.6</td><td>33.0</td><td>5.2</td><td>18.6</td></tr><tr><td>k-means++</td><td>1.6</td><td>30.4</td><td>78.8</td><td>18.4</td></tr><tr><td>LS-k-means++-5</td><td>37.2</td><td>23.5</td><td>97.1</td><td>15.3</td></tr><tr><td>LS-k-means++-10</td><td>73.1</td><td>20.1</td><td>121.6</td><td>13.7</td></tr><tr><td>BanditPAM++-0</td><td>930.2</td><td>3.6</td><td>Na</td><td>Na</td></tr><tr><td>BanditPAM++-2</td><td>1670.1</td><td>2.8</td><td>Na</td><td>Na</td></tr><tr><td>BanditPAM++-5</td><td>2880.7</td><td>2.2</td><td>Na</td><td>Na</td></tr><tr><td>OneBatchPAM-lwcs</td><td>15.1</td><td>12.3</td><td>117.9</td><td>2.8</td></tr><tr><td>OneBatchPAM-unif</td><td>15.1</td><td>3.9</td><td>104.2</td><td>1.2</td></tr><tr><td>OneBatchPAM-debias</td><td>15.7</td><td>3.7</td><td>100.0</td><td>0.8</td></tr><tr><td>OneBatchPAM-nniw</td><td>15.5</td><td>1.7</td><td>100.0</td><td>0.0</td></tr></table>

Table 3: Results Summary. The scores are averaged over the five repetitions of the experiment, the three values of $k \in \{10, 50, 100\}$ and the five respective “small scale” and “large scale” datasets. RT and $\Delta RO$ are given in percentage. Standard deviations are reported in Appendix.

# Conclusion and Perspectives

This paper introduces OneBatchPAM, a novel k-medoids algorithm that accelerates FasterPAM by using a single batch of size $m = \mathcal{O}(\log(n))$ to estimate the objective. Our experiments demonstrate that OneBatchPAM is an efficient alternative to subsampling for handling large datasets within a reasonable running time while achieving performance similar to FasterPAM (with less than 2% error). Future work will focus on refining the subsampling process to further improve the running time and the accuracy of the medoid selection.

# References

Arthur, D.; Vassilvitskii, S.; et al. 2007. k-means++: The advantages of careful seeding. In Soda, volume 7, 1027–1035.   
Arya, V.; Garg, N.; Khandekar, R.; Meyerson, A.; Munagala, K.; and Pandit, V. 2001. Local search heuristic for k-median and facility location problems. In Proceedings of the thirty-third annual ACM symposium on Theory of computing, 21–29.   
Bachem, O.; Lucic, M.; Hassani, S. H.; and Krause, A. 2016. Approximate k-means++ in sublinear time. In Proceedings of the AAAI conference on artificial intelligence, volume 30.   
Bachem, O.; Lucic, M.; and Krause, A. 2018. Scalable k-means clustering via lightweight coresets. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1119–1127.   
Beretta, L.; Cohen-Addad, V.; Lattanzi, S.; and Parotsidis, N. 2024. Multi-Swap k-Means++. Advances in Neural Information Processing Systems, 36.   
Bhat, A. 2014. K-medoids clustering using partitioning around medoids for performing face recognition. International Journal of Soft Computing, Mathematics and Control, 3(3): 1–12.   
Braverman, V.; Cohen-Addad, V.; Jiang, H.-C. S.; Krauthgamer, R.; Schwiegelshohn, C.; Toftrup, M. B.; and Wu, X. 2022. The power of uniform sampling for coresets. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS), 462–473. IEEE.   
Byrka, J.; Pensyl, T.; Rybicki, B.; Srinivasan, A.; and Trinh, K. 2017. An improved approximation for k-median and positive correlation in budgeted optimization. ACM Transactions on Algorithms (TALG), 13(2): 1–31.   
Charikar, M.; Guha, S.; Tardos, É.; and Shmoys, D. B. 1999. A constant-factor approximation algorithm for the k-median problem. In Proceedings of the thirty-first annual ACM symposium on Theory of computing, 1–10.   
Charikar, M.; Henzinger, M.; Hu, L.; Vötsch, M.; and Waingarten, E. 2023. Simple, scalable and effective clustering via one-dimensional projections. Advances in Neural Information Processing Systems, 36: 64618–64649.   
Chen, K. 2009. On coresets for k-median and k-means clustering in metric and euclidean spaces and their applications. SIAM Journal on Computing, 39(3): 923–947.   
Chrobak, M.; Kenyon, C.; and Young, N. 2006. The reverse greedy algorithm for the metric k-median problem. Information Processing Letters, 97(2): 68–72.   
Cohen-Addad, V.; Larsen, K. G.; Saulpic, D.; and Schwiegelshohn, C. 2022. Towards optimal lower bounds for k-median and k-means coresets. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, 1038–1051.   
Cohen-Addad, V.; Lattanzi, S.; Norouzi-Fard, A.; Sohler, C.; and Svensson, O. 2020. Fast and accurate k-means++ via rejection sampling. Advances in Neural Information Processing Systems, 33: 16235–16245.

Cohen-Addad, V.; Saulpic, D.; and Schwiegelshohn, C. 2021a. Improved coresets and sublinear algorithms for power means in euclidean spaces. Advances in Neural Information Processing Systems, 34: 21085–21098.   
Cohen-Addad, V.; Saulpic, D.; and Schwiegelshohn, C. 2021b. A new coreset framework for clustering. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, 169–182.   
Czumaj, A.; and Sohler, C. 2007. Sublinear-time approximation algorithms for clustering via random sampling. Random Structures & Algorithms, 30(1-2): 226–256.   
de Mathelin, A.; Deheeger, F.; MOUGEOT, M.; and Vayatis, N. 2021. Discrepancy-Based Active Learning for Domain Adaptation. In International Conference on Learning Representations.   
Dohan, D.; Karp, S.; and Matejek, B. 2015. K-median algorithms: theory in practice. Princeton University.   
Dua, D.; and Graff, C. 2017. UCI Machine Learning Repository.   
Feldman, D. 2020. Core-sets: Updated survey. Sampling techniques for supervised or unsupervised tasks, 23–44.   
Feldman, D.; and Langberg, M. 2011. A unified framework for approximating and clustering data. In Proceedings of the forty-third annual ACM symposium on Theory of computing, 569–578.   
Guha, S.; and Mishra, N. 2016. Clustering data streams. In Data stream management: processing high-speed data streams, 169–187. Springer.   
Har-Peled, S.; and Mazumdar, S. 2004. On coresets for k-means and k-median clustering. In Proceedings of the thirty-sixth annual ACM symposium on Theory of computing, 291–300.   
Huang, J.; Feng, Q.; Huang, Z.; Xu, J.; and Wang, J. 2024. Linear Time Algorithms for k-means with Multi-Swap Local Search. Advances in Neural Information Processing Systems, 36.   
Huang, L.; Jiang, S. H.-C.; and Lou, J. 2023. The power of uniform sampling for k-median. In International Conference on Machine Learning, 13933–13956. PMLR.   
Jaiswal, R.; and Kumar, A. 2024. Universal Weak Coreset. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 12782–12789.   
Kariv, S.; and Hakimi, O. 1979. An algorithmic approach to network location problems. II: the p-medians. SIAM J. Appl. Math, 37(3): 539.   
Kaufman, L. 1986. Clustering large data sets. Pattern recognition in practice, 425–437.   
Kaufman, L. 1990. Partitioning around medoids (program pam). Finding groups in data, 344: 68–125.   
Kaufman, L.; and Rousseeuw, P. J. 1987. Clustering by means of Medoids. Statistical data analysis based on the L1-norm and related methods, edited by Y. Dodge.   
Kaufman, L.; and Rousseeuw, P. J. 2008. Clustering large applications (Program CLARA). Finding groups in data: an introduction to cluster analysis, 126–63.

Kaushal, V.; Iyer, R.; Kothawade, S.; Mahadev, R.; Doctor, K.; and Ramakrishnan, G. 2019. Learning from less data: A unified data subset selection and active learning framework for computer vision. In 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), 1289–1299. IEEE.   
Krizhevsky, A.; Hinton, G.; et al. 2009. Learning multiple layers of features from tiny images.   
Lattanzi, S.; and Sohler, C. 2019. A better k-means++ algorithm via local search. In International Conference on Machine Learning, 3662–3671. PMLR.   
LeCun, Y.; Cortes, C.; and Burges, C. J. 1994. The MNIST database of handwritten digits.   
Li, S.; and Svensson, O. 2013. Approximating k-median via pseudo-approximation. In proceedings of the forty-fifth annual ACM symposium on theory of computing, 901–910.   
Loog, M. 2012. Nearest neighbor-based importance weighting. In 2012 IEEE international workshop on machine learning for signal processing, 1–6. IEEE.   
Mettu, R. R.; and Plaxton, C. G. 2004. Optimal time bounds for approximate clustering. Machine Learning, 56(1): 35–60.   
Meyerson, A.; O'callaghan, L.; and Plotkin, S. 2004. A k-median algorithm with running time independent of data size. Machine Learning, 56(1): 61–87.   
Mishra, N.; Oblinger, D.; and Pitt, L. 2001. Sublinear time approximate clustering. In SODA, volume 1, 439–447.   
Park, H.-S.; and Jun, C.-H. 2009. A simple and fast algorithm for K-medoids clustering. Expert systems with applications, 36(2): 3336–3341.   
Pelleg, D.; and Moore, A. 1999. Accelerating exact k-means algorithms with geometric reasoning. In Proceedings of the fifth ACM SIGKDD international conference on Knowledge discovery and data mining, 277–281.   
Ren, J.; Hua, K.; and Cao, Y. 2022. Global optimal k-medoids clustering of one million samples. Advances in Neural Information Processing Systems, 35: 982–994.   
Schmidt, M.; Schwiegelshohn, C.; and Sohler, C. 2020. Fair coresets and streaming algorithms for fair k-means. In Approximation and Online Algorithms: 17th International Workshop, WAOA 2019, Munich, Germany, September 12–13, 2019, Revised Selected Papers 17, 232–251. Springer.   
Schubert, E.; and Lenssen, L. 2022. Fast k-medoids Clustering in Rust and Python. Journal of Open Source Software, 7(75): 4183.   
Schubert, E.; and Rousseeuw, P. J. 2021. Fast and eager k-medoids clustering: O (k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. Information Systems, 101: 101804.   
Thorup, M. 2005. Quick k-median, k-center, and facility location for sparse graphs. SIAM Journal on Computing, 34(2): 405–432.   
Tiwari, M.; Kang, R.; Lee, D.; Thrun, S.; Shomorony, I.; and Zhang, M. J. 2023. BanditPAM++: Faster k-medoids Clustering. Advances in Neural Information Processing Systems, 36: 73371–73382.

Tiwari, M.; Zhang, M. J.; Mayclin, J.; Thrun, S.; Piech, C.; and Shomorony, I. 2020. Banditpam: Almost linear time k-medoids clustering via multi-armed bandits. Advances in Neural Information Processing Systems, 33: 10211–10222. Wei, K.; Iyer, R.; and Bilmes, J. 2015. Submodularity in data subset selection and active learning. In International conference on machine learning, 1954–1963. PMLR.

# Appendix A. Algorithm

This section presents the pseudo-code of the OneBatchPAM algorithm. For simplicity, we consider the “Uniform” variant where the sample $X_{m}$ is selected uniformly at random in $X_{n}$ without reweighting and the two variants Debias and NNIW. For Algorithm 2, we respectively define $\text{near}(j)$ , $\text{sec}(j)$ as the indices in $[1, k]$ of the nearest and second nearest medoid to $x_{\sigma(j)}$ in M. We denote $d_{\text{near}(j)}$ , $d_{\text{sec}(j)}$ the corresponding dissimilarities between $x_{\sigma(j)}$ and its respective nearest and second nearest medoid in M. The Approximated-FasterPAM algorithm is close to FasterPAM (Schubert and Rousseeuw 2021). The difference lies in the loop of line 9, as the loop is performed only over the subsampled data points $x_{\sigma(j)}$ .

Algorithm 1: OneBatchPAM   
1: Inputs: Data $X_{n}$ , number of medoids k, maximal number of iteration T, batch size m
2: Outputs: Set of medoids M
3: Uniformly select $X_{m} \subset X_{n}$ of size m
4: Compute $d_{ij} = d(x_{i}, x_{\sigma(j)})$ for any $j \in [1, m]$ and any $i \in [1, n]$ 5: (For the NNIW variant) Compute $w_{j}$ according to (Loog 2012) and update $d_{ij} \leftarrow w_{j}d_{ij}$ 6: (For the Debias variant) Update $d_{jj} \leftarrow +\infty$ 7: Randomly select $\mathcal{M} \in \mathcal{P}_{k}(\mathcal{X}_{n})$ 8: Approximated-FasterPAM( $\{d_{ij}\}_{i \leq n,j \leq m}, M, T, k, n, m$ )

Algorithm 2: Approximated-FasterPAM   
1: Inputs: $\{d_{ij}\}_{i\leq n,j\leq m}$ , M, k, T, n, m
2: Outputs: Set of medoids M
3: For any $j\in[1,m]$ compute near(j), sec(j), $d_{\text{near}(j)}$ , $d_{\text{sec}(j)}$ 4: For any $l\in[1,k]$ , initialize $G_l=\sum_{j\in[1,m]}d_{\text{near}(j)}-d_{\text{sec}(j)}$ 5: for $1\leq t\leq T$ do
6:    for $1\leq i\leq n$ do
7:    Initialize $G_l^i\leftarrow G_l$ for any $l\in[1,k]$ 8:    Initialize $G^i\leftarrow0$ 9:    for $1\leq j\leq m$ do
10:    if $d_{ij}<d_{\text{near}(j)}$ then
11: $G^i\leftarrow G^i+d_{\text{near}(j)}-d_{ij}$ 12: $G_{\text{near}(j)}^i\leftarrow G_{\text{near}(j)}^i+d_{\text{sec}(j)}-d_{\text{near}(j)}$ 13:    else if $d_{ij}<d_{\text{sec}(j)}$ then
14: $G_{\text{near}(j)}^i\leftarrow G_{\text{near}(j)}^i+d_{\text{sec}(j)}-d_{\text{near}(j)}$ 15:    end if
16:    end for
17: $l^{*}=\arg\max_{l\in[1,k]}G_l^i$ 18: $G^i\leftarrow G^i+G_{l^{*}}^i$ 19:    if $G^i>0$ then
20:    Update M: swap role of medoid of indice $l^{*}$ in M with $x_i$ .
21:    Update near(j), sec(j), $d_{\text{near}(j)}$ , $d_{\text{sec}(j)}$ and $G_l$ for any $l\in[1,k]$ 22:    end if
23:    end for
24: end for

# Appendix B. Proof of Theorem 1

Theorem 1. Let $\mathcal{X}_m$ be a subsample uniformly drawn from $\mathcal{X}_n$ . Let $D = \max_{(x,x') \in \mathcal{X}_n} d(x, x')$ and $\Delta$ be the smallest difference between two objectives computed by FasterPAM. Then, for any $\delta \in]0,1]$ , the OneBatchPAM algorithm returns the same set of medoid as FasterPAM with probability at least $1 - \delta$ if:

$$
m \geq \frac {4 D ^ {2}}{\Delta^ {2}} \log \left(\frac {2 T n}{\delta}\right). \tag {7}
$$

Where $\Delta = \min_{t\in [0,T]}\min_{\substack{x\in \mathcal{M}_{t},\\ x^{\prime}\in \mathcal{X}_{n}\setminus \mathcal{M}_{t}}}| \mathcal{L}(\mathcal{M}_{t}) - \mathcal{L}(\mathcal{M}_{t}\setminus \{x\} \cup \{x^{\prime}\})|$

Proof. For any $t \in [0, T]$ , $\mathcal{M}_t \in \mathcal{P}_k(\mathcal{X}_n)$ denotes the medoid selection of FasterPAM after $t$ swaps.

We denote by $\widehat{\mathcal{L}}(\mathcal{M})$ the empirical risk for any $\mathcal{M} \in \mathcal{P}_k(\mathcal{X}_n)$ such that $\widehat{\mathcal{L}}(\mathcal{M}) = \frac{1}{m} \sum_{j=1}^{m} d(x_{\sigma(j)}, \mathcal{M})$ .

At each swap step $t \in [0, T]$ , the FasterPAM algorithm evaluates the objective of several pairs $(x, x') \in \mathcal{M}_t \times \mathcal{X}_n / \mathcal{M}_t$ . It compares it to the current objective $\mathcal{L}(\mathcal{M}_t)$ until finding a pair with a lower objective (if no such pair is found, the algorithm terminates). Let's denote $\mathcal{P}_t \subset \mathcal{M}_t \times \mathcal{X}_n$ the swap pairs evaluated by FasterPAM with a larger objective than $\mathcal{L}(\mathcal{M}_t)$ and $(x_t, x'_t) \in \mathcal{M}_t \times \mathcal{X}_n$ the swap pair selected by FasterPAM. Thus, for any $t \in [0, T]$ and any $(x, x') \in \mathcal{P}_t$ we have:

$$
\mathcal {L} \left(\mathcal {M} _ {t}\right) <   \mathcal {L} \left(\mathcal {M} ^ {\left(x, x ^ {\prime}\right)}\right) \tag {8}
$$

$$
\mathcal {L} \left(\mathcal {M} _ {t}\right) > \mathcal {L} \left(\mathcal {M} ^ {\left(x _ {t}, x _ {t} ^ {\prime}\right)}\right), \tag {9}
$$

where $\mathcal{M}_t^{(x,x')} = \mathcal{M}_t \setminus \{x\} \cup \{x'\}$ .

Let's consider $\delta \in ]0,1]$ , we define $\tilde{\delta} \in ]0,1]$ as follows:

$$
\tilde {\delta} = \frac {\delta}{2 T n ^ {2}} \tag {10}
$$

Let's consider a subsample size $m$ verifying Equation (7). It can be noticed that:

$$
m \geq \frac {2 D ^ {2}}{\Delta^ {2}} \log \left(\frac {1}{\tilde {\delta}}\right) \tag {11}
$$

To prove that OneBatchPAM selects the same swap pairs as FasterPAM, we have to show that, for any $t \in [0, T]$ , the objective estimation for any swap pairs in $P_{t}$ is lower than the current objective estimation, while the objective estimation for the pair $(x_{t}, x_{t}')$ is larger, i.e., for any $t \in [0, T]$ and any $(x, x') \in \mathcal{P}_{t}$

$$
\widehat {\mathcal {L}} (\mathcal {M} _ {t}) <   \widehat {\mathcal {L}} (\mathcal {M} ^ {(x, x ^ {\prime})}) \tag {12}
$$

$$
\widehat {\mathcal {L}} (\mathcal {M} _ {t}) > \widehat {\mathcal {L}} (\mathcal {M} ^ {(x _ {t}, x _ {t} ^ {\prime})}), \tag {13}
$$

For this purpose, we will show that the probability of the events $\widehat{\mathcal{L}}(\mathcal{M}_t) \geq \widehat{\mathcal{L}}(\mathcal{M}^{(x,x')})$ and $\widehat{\mathcal{L}}(\mathcal{M}_t) \leq \widehat{\mathcal{L}}(\mathcal{M}^{(x_t,x'_t)})$ is upperbounded by $\tilde{\delta}$ .

Let $t \in [0, T]$ and $\mathcal{M}_t \in \mathcal{P}_k(\mathcal{X}_n)$ , by the Hoeffding inequality, we have for any $(x, x') \in \mathcal{P}_t$ :

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t}) - \mathcal {L} (\mathcal {M} _ {t}) \geq D \sqrt {\frac {\log (1 / \tilde {\delta})}{2 m}}\right) \leq \tilde {\delta} \tag {14}
$$

$$
\mathbb {P} \left(\mathcal {L} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) - \widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) \geq D \sqrt {\frac {\log (1 / \tilde {\delta})}{2 m}}\right) \leq \tilde {\delta} \tag {15}
$$

And,

$$
\mathbb {P} \left(\mathcal {L} (\mathcal {M} _ {t}) - \widehat {\mathcal {L}} (\mathcal {M} _ {t}) \geq D \sqrt {\frac {\log (1 / \tilde {\delta})}{2 m}}\right) \leq \tilde {\delta} \tag {16}
$$

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x _ {t}, x _ {t} ^ {\prime})}) - \mathcal {L} (\mathcal {M} _ {t} ^ {(x _ {t}, x _ {t} ^ {\prime})}) \geq D \sqrt {\frac {\log (1 / \tilde {\delta})}{2 m}}\right) \leq \tilde {\delta} \tag {17}
$$

Let's consider $(x,x^{\prime})\in \mathcal{P}_{t}$ , to simplify the notations, we define the five quantities: $C = D\sqrt{\frac{\log(1 / \tilde{\delta})}{2m}}$ , $L = \mathcal{L}(\mathcal{M}_t)$ , $\widehat{L} = \widehat{\mathcal{L}} (\mathcal{M}_t)$ , $L_{x} = \mathcal{L}(\mathcal{M}_{t}^{(x,x^{\prime})})$ , $\widehat{L}_x = \widehat{\mathcal{L}} (\mathcal{M}_t^{(x,x')})$ ,

We have:

$$
\begin{array}{l} \mathbb {P} \left(\widehat {L} _ {x} \leq \widehat {L}\right) = \mathbb {P} \left(\left\{\widehat {L} _ {x} \leq \widehat {L} \right\} \cap \left\{\widehat {L} _ {x} > L _ {x} - C \right\}\right) + \mathbb {P} \left(\left\{\widehat {L} _ {x} \leq \widehat {L} \right\} \cap \left\{\widehat {L} _ {x} \leq L _ {x} - C \right\}\right) \\ \leq \mathbb {P} \left(L _ {x} - C <   \widehat {L}\right) + \mathbb {P} \left(\widehat {L} _ {x} \leq L _ {x} - C\right) \\ \leq \mathbb {P} (L _ {x} - C <   \widehat {L}) + \tilde {\delta} \tag {18} \\ \leq \mathbb {P} \left(\left\{L _ {x} - C <   \widehat {L} \right\} \cap \left\{\widehat {L} <   L + C \right\}\right) + \mathbb {P} \left(\left\{L _ {x} - C <   \widehat {L} \right\} \cap \left\{\widehat {L} \geq L + C \right\}\right) + \tilde {\delta} \\ \leq \mathbb {P} (L _ {x} - C <   L + C) + \mathbb {P} (\widehat {L} \geq L + C) + \tilde {\delta} \\ \leq \mathbb {P} (L _ {x} - L <   2 C) + 2 \tilde {\delta}, \\ \end{array}
$$

by using the respective Equations (15) and (14) for the third and sixth lines.

Therefore,

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) <   \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right) \leq \mathbb {P} \left(\mathcal {L} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) - \mathcal {L} (\mathcal {M} _ {t}) <   2 C\right) + 2 \tilde {\delta} \tag {19}
$$

The quantity $\mathcal{L}(\mathcal{M}_t^{(x,x'}) - \mathcal{L}(\mathcal{M}_t)$ is positive, as $(x,x')$ is not a swap pair. Then, by definition of $\Delta$ , we have:

$$
\mathcal {L} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) - \mathcal {L} (\mathcal {M} _ {t}) \geq \Delta \tag {20}
$$

Then:

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) \leq \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right) \leq \mathbb {P} \left(\Delta <   2 C\right) + 2 \delta \tag {21}
$$

If $m$ verifies Equation (7), we have:

$$
2 C \leq 2 D \sqrt {\frac {\log (1 / \tilde {\delta})}{4 \frac {D ^ {2}}{\Delta^ {2}} \log (1 / \tilde {\delta})}} \leq \Delta . \tag {22}
$$

Then, $\mathbb{P}(\Delta < 2C) = 0$ and we conclude that:

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) \leq \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right) \leq 2 \tilde {\delta} \tag {23}
$$

By using Equations (16) and (17), a similar proof can be derived to show that:

$$
\mathbb {P} \left(\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x _ {t}, x _ {t} ^ {\prime})}) \geq \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right) \leq 2 \tilde {\delta}, \tag {24}
$$

Let's denote $A$ the event: "OneBatchPAM performs a different swap as FasterPAM":

$$
A = \bigcup_ {t \in [ 0, T - 1 ]} \left(\bigcup_ {(x, x ^ {\prime}) \in \mathcal {P} _ {t}} \left\{\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}) \leq \widehat {\mathcal {L}} (\mathcal {M} _ {t}) \right\} \cup \left\{\widehat {\mathcal {L}} (\mathcal {M} _ {t} ^ {(x _ {t}, x _ {t} ^ {\prime})}) \geq \widehat {\mathcal {L}} (\mathcal {M} _ {t}) \right\}\right) \tag {25}
$$

Then,

$$
\begin{array}{l} \mathbb {P} (A) \leq \sum_ {t \in [ 0, T - 1 ]} \left(\sum_ {(x, x ^ {\prime}) \in \mathcal {P} _ {t}} \mathbb {P} \left(\widehat {\mathcal {L}} \left(\mathcal {M} _ {t} ^ {(x, x ^ {\prime})}\right) \leq \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right) + \mathbb {P} \left(\widehat {\mathcal {L}} \left(\mathcal {M} _ {t} ^ {(x _ {t}, x _ {t} ^ {\prime})}\right) \geq \widehat {\mathcal {L}} (\mathcal {M} _ {t})\right)\right) \tag {26} \\ \leq 2 T n ^ {2} \tilde {\delta} \\ \leq \delta \\ \end{array}
$$

Finally, it can be concluded that, if $m$ verifies Equation (7), then OneBatchPAM performs the same swaps as FasterPAM (and thus returns the same set of medoids) with probability at least $1 - \delta$ .

□

# Appendix C. Detailed Results

Table 4: Results Summary. The scores are averaged over the five repetitions of the experiment, the three values of $k \in [10, 50, 100]$ and the five respective “small scale” and “large scale” datasets. RT and $\Delta RO$ are given in percentage. The standard deviations computed over the five repetitions of the experiments and averaged over the five datasets and the three values of k are reported in brackets. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Small Scale</td><td colspan="2">Large Scale</td></tr><tr><td>RT</td><td>ΔRO</td><td>RT</td><td>ΔRO</td></tr><tr><td>Random</td><td>0.0 (0.0)</td><td>62.9 (16.4)</td><td>0.0 (0.0)</td><td>20.3 (2.4)</td></tr><tr><td>FasterPAM</td><td>100.0 (10.6)</td><td>0.0 (0.3)</td><td>NaN</td><td>NaN</td></tr><tr><td>Alternate</td><td>161.1 (43.2)</td><td>20.0 (7.4)</td><td>NaN</td><td>NaN</td></tr><tr><td>FasterCLARA-5</td><td>2.8 (0.4)</td><td>13.0 (1.5)</td><td>15.0 (0.5)</td><td>8.0 (0.6)</td></tr><tr><td>FasterCLARA-50</td><td>30.0 (1.3)</td><td>10.9 (0.8)</td><td>161.7 (3.2)</td><td>7.1 (0.4)</td></tr><tr><td>kmc2-20</td><td>14.5 (0.5)</td><td>31.3 (4.4)</td><td>0.5 (0.0)</td><td>18.2 (2.4)</td></tr><tr><td>kmc2-100</td><td>72.2 (1.0)</td><td>31.9 (4.9)</td><td>2.4 (0.2)</td><td>17.6 (2.3)</td></tr><tr><td>kmc2-200</td><td>153.6 (9.2)</td><td>33.0 (6.1)</td><td>5.2 (0.3)</td><td>18.6 (2.6)</td></tr><tr><td>k-means++</td><td>1.6 (0.1)</td><td>30.4 (4.8)</td><td>78.8 (4.1)</td><td>18.4 (2.7)</td></tr><tr><td>LS-k-means++-5</td><td>37.2 (0.5)</td><td>23.5 (3.3)</td><td>97.1 (2.1)</td><td>15.3 (1.8)</td></tr><tr><td>LS-k-means++-10</td><td>73.1 (2.2)</td><td>20.1 (2.9)</td><td>121.6 (2.9)</td><td>13.7 (1.7)</td></tr><tr><td>BanditPAM++-0</td><td>930.2 (40.3)</td><td>3.6 (0.3)</td><td>NaN</td><td>NaN</td></tr><tr><td>BanditPAM++-2</td><td>1670.1 (41.3)</td><td>2.8 (0.3)</td><td>NaN</td><td>NaN</td></tr><tr><td>BanditPAM++-5</td><td>2880.7 (65.5)</td><td>2.2 (0.2)</td><td>NaN</td><td>NaN</td></tr><tr><td>OneBatch-lwcs</td><td>15.1 (1.2)</td><td>12.3 (1.5)</td><td>118.2 (7.9)</td><td>2.7 (0.6)</td></tr><tr><td>OneBatch-unif</td><td>15.1 (1.8)</td><td>3.9 (0.7)</td><td>104.2 (8.8)</td><td>1.2 (0.4)</td></tr><tr><td>OneBatch-debias</td><td>15.7 (3.0)</td><td>3.7 (0.7)</td><td>100.0 (6.3)</td><td>0.8 (0.3)</td></tr><tr><td>OneBatch-nniw</td><td>15.5 (1.6)</td><td>1.7 (0.5)</td><td>100.0 (4.1)</td><td>0.0 (0.3)</td></tr></table>

# Appendix C.1. Detailed Results Small Scale

Table 5: Relative Time (RT) per dataset for the “small scale” experiments. The scores are averaged over the five repetitions of the experiment and the three values of $k \in [10, 50, 100]$ . RT is given in percentage. The standard deviations are reported in brackets. 

<table><tr><td>Datasets Methods</td><td>abalone</td><td>bankruptcy</td><td>drybean</td><td>letter</td><td>mapping</td></tr><tr><td>Random</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td></tr><tr><td>FasterPAM</td><td>100.0 (5.6)</td><td>100.0 (19.7)</td><td>100.0 (8.4)</td><td>100.0 (10.1)</td><td>100.0 (9.3)</td></tr><tr><td>Alternate</td><td>150.9 (34.7)</td><td>97.9 (23.8)</td><td>321.2 (124.6)</td><td>140.3 (23.7)</td><td>95.0 (9.4)</td></tr><tr><td>FasterCLARA-5</td><td>6.6 (1.2)</td><td>1.9 (0.1)</td><td>1.9 (0.5)</td><td>1.1 (0.1)</td><td>2.4 (0.1)</td></tr><tr><td>FasterCLARA-50</td><td>72.6 (1.7)</td><td>21.3 (1.8)</td><td>19.1 (1.3)</td><td>11.3 (0.2)</td><td>25.8 (1.6)</td></tr><tr><td>kmc2-20</td><td>59.4 (1.9)</td><td>2.0 (0.0)</td><td>4.6 (0.3)</td><td>1.9 (0.0)</td><td>4.5 (0.1)</td></tr><tr><td>kmc2-100</td><td>296.6 (3.5)</td><td>9.8 (0.1)</td><td>21.7 (0.3)</td><td>9.6 (0.1)</td><td>23.5 (1.1)</td></tr><tr><td>kmc2-200</td><td>634.0 (36.9)</td><td>20.4 (0.9)</td><td>45.9 (3.9)</td><td>19.8 (0.9)</td><td>47.9 (3.2)</td></tr><tr><td>k-means++</td><td>2.1 (0.1)</td><td>2.2 (0.1)</td><td>1.3 (0.0)</td><td>0.9 (0.0)</td><td>1.6 (0.1)</td></tr><tr><td>LS-k-means++-5</td><td>103.3 (1.6)</td><td>8.3 (0.1)</td><td>31.8 (0.3)</td><td>20.6 (0.3)</td><td>21.9 (0.1)</td></tr><tr><td>LS-k-means++-10</td><td>202.5 (4.3)</td><td>14.3 (0.2)</td><td>63.7 (1.1)</td><td>40.3 (0.2)</td><td>44.7 (5.4)</td></tr><tr><td>BanditPAM++-0</td><td>1388.5 (87.7)</td><td>722.7 (41.8)</td><td>858.0 (8.8)</td><td>733.2 (7.7)</td><td>948.8 (55.7)</td></tr><tr><td>BanditPAM++-2</td><td>2729.9 (52.8)</td><td>1073.0 (43.8)</td><td>1491.7 (30.3)</td><td>1270.7 (7.8)</td><td>1785.0 (71.9)</td></tr><tr><td>BanditPAM++-5</td><td>5394.9 (222.2)</td><td>1574.8 (28.8)</td><td>2434.9 (33.2)</td><td>2068.0 (5.9)</td><td>2930.8 (37.3)</td></tr><tr><td>OneBatch-lwcs</td><td>34.3 (3.6)</td><td>7.5 (0.6)</td><td>12.2 (0.5)</td><td>7.8 (0.5)</td><td>13.6 (1.0)</td></tr><tr><td>OneBatch-unif</td><td>31.3 (2.5)</td><td>6.8 (0.2)</td><td>13.3 (2.9)</td><td>8.8 (1.9)</td><td>15.4 (1.5)</td></tr><tr><td>OneBatch-debias</td><td>36.8 (9.4)</td><td>7.1 (0.2)</td><td>11.9 (0.5)</td><td>8.0 (1.2)</td><td>14.7 (3.6)</td></tr><tr><td>OneBatch-nniw</td><td>34.0 (3.7)</td><td>7.1 (0.4)</td><td>14.3 (3.0)</td><td>8.5 (0.5)</td><td>13.5 (0.7)</td></tr></table>

Table 6: Delta Relative Objective ( $\Delta$ RO) per dataset for the “small scale” experiments. The scores are averaged over the five repetitions of the experiment and the three values of $k \in [10, 50, 100]$ . $\Delta$ RO is given in percentage. The standard deviations are reported in brackets. 

<table><tr><td>Methods\Datasets</td><td>abalone</td><td>bankruptcy</td><td>drybean</td><td>letter</td><td>mapping</td></tr><tr><td>Random</td><td>82.5 (15.1)</td><td>32.4 (3.0)</td><td>154.6 (59.7)</td><td>26.4 (2.1)</td><td>18.6 (2.3)</td></tr><tr><td>FasterPAM</td><td>0.0 (0.2)</td><td>0.0 (0.1)</td><td>0.0 (0.9)</td><td>0.0 (0.1)</td><td>0.0 (0.1)</td></tr><tr><td>Alternate</td><td>30.1 (10.9)</td><td>12.1 (3.8)</td><td>41.9 (19.8)</td><td>9.4 (1.2)</td><td>6.2 (1.3)</td></tr><tr><td>FasterCLARA-5</td><td>13.5 (1.4)</td><td>12.2 (0.7)</td><td>16.3 (3.8)</td><td>13.5 (0.9)</td><td>9.5 (0.6)</td></tr><tr><td>FasterCLARA-50</td><td>10.8 (0.8)</td><td>11.2 (0.6)</td><td>12.1 (1.7)</td><td>11.9 (0.5)</td><td>8.4 (0.4)</td></tr><tr><td>kmc2-20</td><td>40.3 (5.7)</td><td>30.5 (3.9)</td><td>42.4 (8.4)</td><td>24.9 (2.1)</td><td>18.4 (1.7)</td></tr><tr><td>kmc2-100</td><td>41.2 (6.6)</td><td>31.3 (4.4)</td><td>41.0 (9.4)</td><td>26.0 (2.0)</td><td>19.8 (2.4)</td></tr><tr><td>kmc2-200</td><td>45.7 (10.2)</td><td>30.3 (4.7)</td><td>42.5 (11.0)</td><td>27.2 (1.7)</td><td>19.5 (2.7)</td></tr><tr><td>k-means++</td><td>39.5 (7.7)</td><td>31.7 (3.1)</td><td>35.0 (7.8)</td><td>26.1 (4.1)</td><td>19.4 (1.6)</td></tr><tr><td>LS-k-means++-5</td><td>30.9 (6.0)</td><td>23.7 (3.1)</td><td>22.5 (3.0)</td><td>22.6 (2.3)</td><td>17.5 (2.1)</td></tr><tr><td>LS-k-means++-10</td><td>25.6 (4.7)</td><td>21.1 (3.1)</td><td>18.3 (3.2)</td><td>20.5 (1.9)</td><td>14.9 (1.4)</td></tr><tr><td>BanditPAM++-0</td><td>5.2 (0.6)</td><td>3.6 (0.3)</td><td>4.7 (0.5)</td><td>2.1 (0.1)</td><td>2.3 (0.2)</td></tr><tr><td>BanditPAM++-2</td><td>3.7 (0.4)</td><td>2.6 (0.2)</td><td>4.1 (0.5)</td><td>1.8 (0.1)</td><td>1.8 (0.2)</td></tr><tr><td>BanditPAM++-5</td><td>2.9 (0.4)</td><td>2.1 (0.2)</td><td>3.1 (0.3)</td><td>1.6 (0.0)</td><td>1.3 (0.1)</td></tr><tr><td>OneBatch-lwcs</td><td>10.6 (1.2)</td><td>3.1 (0.5)</td><td>41.6 (5.0)</td><td>4.1 (0.5)</td><td>2.3 (0.4)</td></tr><tr><td>OneBatch-unif</td><td>3.5 (0.6)</td><td>3.6 (0.4)</td><td>6.8 (1.7)</td><td>3.3 (0.6)</td><td>2.6 (0.3)</td></tr><tr><td>OneBatch-debias</td><td>3.1 (0.6)</td><td>3.0 (0.4)</td><td>6.7 (1.7)</td><td>3.3 (0.6)</td><td>2.3 (0.3)</td></tr><tr><td>OneBatch-nniw</td><td>1.4 (0.4)</td><td>1.6 (0.2)</td><td>2.4 (1.2)</td><td>1.8 (0.2)</td><td>1.4 (0.3)</td></tr></table>

![](images/229c7fb433cf4ef6284188a0678f737e2f225c72d3505fda2596455c96989a54.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 (%) | 50 (%) | 100 (%) | Avg (%) |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| k-means++ | 0.1 | 0 | 6.1 | 2.07 |
| FasterCLARA-5 | 1.4 | 5.3 | 13.1 | 6.6 |
| OneBatch-unif | 34.5 | 29.9 | 29.5 | 31.3 |
| OneBatch-nniw | 33.9 | 35.9 | 32.1 | 34 |
| OneBatch-lwcs | 35.9 | 33.9 | 33.2 | 34.3 |
| OneBatch-debias | 45.4 | 32.5 | 32.6 | 36.8 |
| kmc2-20 | 9.9 | 58.8 | 109 | 59.4 |
| FasterCLARA-50 | 26.3 | 65.2 | 126 | 72.6 |
| FasterPAM | 100 | 100 | 100 | 100 |
| LS-k-means++-5 | 0.2 | 64.9 | 245 | 103 |
| Alternate | 142 | 167 | 144 | 151 |
| LS-k-means++-10 | 2 | 129 | 476 | 202 |
| kmc2-100 | 64.7 | 289 | 536 | 297 |
| kmc2-200 | 267 | 569 | 999 | 634 |
| BanditPAM++-0 | 773 | 999 | 999 | 999 |
| BanditPAM++-2 | 999 | 999 | 999 | 999 |
| BanditPAM++-5 | 999 | 999 | 999 | 999 |
The chart displays a color scale from purple (low RT) to yellow (high RT). The x-axis represents K values, and the y-axis lists the methods. The average RT across all methods is shown as a line between the top and bottom of the plot.
</details>

![](images/3738437732fdb7425f97ffdef489b710d67e0cf34d06229e053278887422dfd4.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | FasterPAM | OneBatch-nniw | BanditPAM++-5 | OneBatch-debias | OneBatch-unif | BanditPAM++-2 | BanditPAM++-0 | OneBatch-lwcs | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | Alternate | LS-k-means++-5 | k-means++ | kmc2-20 | kmc2-100 | kmc2-200 | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 11 | 3.7 | 6.6 | 9.9 | 9.5 | 16 | 31 | 48 | 43 | 44 | 50 |
| ΔRO (in %) | 0 | 0 | 0 | 0 | 0 | 0.1 | 3.4 | 5.3 | 2.9 | 0.8 | 0.8 | 1 | 4.5 | 5.7 | 7 | 10 | 18 | 33 |
| Avg |
| Color scale: 0 to 30
Color bar: purple = 0, dark purple = 0
Color bar: yellow = 30
Color bar: green = 25
Color bar: light green = 20
Color bar: cyan = 15
Color bar: teal = 10
Color bar: blue = 5
Color bar: dark blue = 0
Color bar: dark blue = 3
Color bar: dark blue = 2
Color bar: dark blue = 1
Color bar: dark blue = 0
Color bar: dark blue = -3
Color bar: dark blue = -2
Color bar: dark blue = -1
Color bar: dark blue = -2
Color bar: dark blue = -3
Color bar: dark blue = -4
Color bar: dark blue = -5
Color bar: dark blue = -6
Color bar: dark blue = -7
Color bar: dark blue = -8
Color bar: dark blue = -9
Color bar: dark blue = -10
Color bar: dark blue = -11
Color bar: dark blue = -12
Color bar: dark blue = -13
Color bar: dark blue = -14
Color bar: dark blue = -15
Color bar: dark blue = -16
Color bar: dark blue = -17
Color bar: dark blue = -18
Color bar: dark blue = -19
Color bar: dark blue = -20
Color bar: dark blue = -21
Color bar: dark blue = -22
Color bar: dark blue = -23
Color bar: dark blue = -24
Color bar: dark blue = -25
Color bar: dark blue = -26
Color bar: dark blue = -27
Color bar: dark blue = -28
Color bar: dark blue = -29
Color bar: dark blue = -30
Color bar: dark blue = -31
Color bar: dark blue = -32
Color bar: dark blue = -33
Color bar: dark blue = -34
Color bar: dark blue = -35
Color bar: dark blue = -36
Color bar: dark blue = -37
Color bar: dark blue = -38
Color bar: dark blue = -39
Color bar: dark blue = -40
Color bar: dark blue = -41
Color bar: dark blue = -42
Color bar: dark blue = -43
Color bar: dark blue = -44
Color bar: dark blue = -45
Color bar: dark blue = -46
Color bar: dark blue = -47
Color bar: dark blue = -48
Color bar: dark blue = -49
Color bar: dark blue = -50
Color bar: dark blue = -51
Color bar: dark blue = -52
Color bar: dark blue = -53
Color bar: dark blue = -54
Color bar: dark blue = -55
Color bar: dark blue = -56
Color bar: dark blue = -57
Color bar: dark blue = -58
Color bar: dark blue = -59
Color bar: dark blue = -60
Color bar: dark blue = -61
Color bar: dark blue = -62
Color bar: dark blue = -63
Color bar: dark blue = -64
Color bar: dark blue = -65
Color bar: dark blue = -66
Color bar: dark blue = -67
Color bar: dark blue = -68
Color bar: dark blue = -69
Color bar: dark blue = -70
Color bar: dark blue = -71
Color bar: dark blue = -72
Color bar: dark blue = -73
Color bar: dark blue = -74
Color bar: dark blue = -75
Color bar: dark blue = -76
Color bar: dark blue = -77
Color bar: dark blue = -78
Color bar: dark blue = -79
Color bar: dark blue = -80
Color bar: blackbody=0, blackbody=1, blackbody=2, blackbody=3, blackbody=4, blackbody=5, blackbody=6, blackbody=7, blackbody=8, blackbody=9, blackbody=10, blackbody=11, blackbody=12, blackbody=13, blackbody=14, blackbody=15, blackbody=16, blackbody=17, blackbody=18, blackbody=19, blackbody=20, blackbody=21, blackbody=22, blackbody=23, blackbody=24, blackbody=25, blackbody=26, blackbody=27, blackbody=28, blackbody=29, blackbody=30, Blackbody=31, Blackbody=32, Blackbody=33, Blackbody=34, Blackbody=35, Blackbody=36, Blackbody=37, Blackbody=38, Blackbody=39, Blackbody=40, Blackbody=41, Blackbody=42, Blackbody=43, Blackbody=44, Blackbody=45, Blackbody=46, Blackbody=47, Blackbody=48, Blackbody=49, Blackbody=50, Blackbody=51, Blackbody=52, Blackbody=53, Blackbody=54, Blackbody=55, Blackbody=56, Blackbody=57, Blackbody=58, Blackbody=59, Blackbody=60, Blackbody=61, Blackbody=62, Blackbody=63, Blackbody=64, Blackbody=65, Blackbody=66, Blackbody=67, Blackbody=68, Blackbody=69, Blackbody=70, Blackbody=71, Blackbody=72, Blackbody=73, Blackbody=74, Blackbody=75, Blackbody=76, Blackbody=77, Blackbody=78, Blackbody=79, Blackbody=80 |
KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2/Random KmC2-100 KmC2-200 KmC2-100 KmC2-200 KmC2-100 KmC2/Random KmC2-100 KmC2-200 KmC2-100 KmC2/200 KmC2/100 KmC2/200 KmC2/100 KmC2/200 KmC2/100 KmC2/200 KmC2/100 KmC2/200 KmC2/100 KmC2/100 KmC2/200 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 KmC2/100 Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans++ Kkmeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans++ LkMeans++ LkMeans++ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans + LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+ LkMeans+
KlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIcKlMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLHtMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeLhMnOaRbIeL h M n O a R b e l o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v i o l o v u<nl>
</details>

Figure 2: RT and $\Delta$ RO for Abalone

![](images/e7c5198c50b04228aa6538bf5eb302c7ec72c825743a580c9077ed1d10665b96.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| FasterCLARA-5 | 0.4 | 1.7 | 3.7 | 1.93 |
| kmc2-20 | 0.2 | 1.8 | 4 | 2 |
| k-means++ | 0.3 | 2 | 4.3 | 2.2 |
| OneBatch-unif | 5.1 | 7.3 | 8.1 | 6.83 |
| OneBatch-debias | 5.2 | 7.5 | 8.5 | 7.07 |
| OneBatch-nniw | 5.4 | 7.6 | 8.3 | 7.1 |
| OneBatch-lwcs | 5.4 | 7.8 | 9.2 | 7.47 |
| LS-k-means++-5 | 0.5 | 5.5 | 19 | 8.33 |
| kmc2-100 | 1.2 | 8.7 | 19.5 | 9.8 |
| LS-k-means++-10 | 0.7 | 8.8 | 33.5 | 14.3 |
| kmc2-200 | 2.6 | 17.3 | 41.2 | 20.4 |
| FasterCLARA-50 | 6.1 | 18.2 | 39.7 | 21.3 |
| Alternate | 78.9 | 92.3 | 123 | 97.9 |
| FasterPAM | 100 | 100 | 100 | 100 |
| BanditPAM++-0 | 101 | 651 | 999 | 723 |
| BanditPAM++-2 | 180 | 972 | 999 | 999 |
| BanditPAM++-5 | 309 | 999 | 999 | 999 |
</details>

![](images/8c53d9c0e3094c2a9ed200a6c7d254fb0093e6f8b5a1dfd253fc7392fea1d1f7.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | K=10 | K=50 | K=100 | Avg |
|---|---|---|---|---|
| FasterPAM | 0 | 0 | 0 | 0 |
| OneBatch-nniw | 0.4 | 1.5 | 2.9 | 1.6 |
| BanditPAM++-5 | 0.1 | 1.8 | 4.4 | 2.1 |
| BanditPAM++-2 | 0.4 | 2.4 | 5.1 | 2.6 |
| OneBatch-debias | 0.9 | 2.9 | 5.1 | 3 |
| OneBatch-lwcs | 0.7 | 3.8 | 4.9 | 3.1 |
| BanditPAM++-0 | 1.4 | 3.2 | 6.1 | 3.6 |
| OneBatch-unif | 0.8 | 3.3 | 6.6 | 3.6 |
| FasterCLARA-50 | 5.7 | 13 | 15 | 11 |
| Alternate | 12 | 12 | 13 | 12 |
| FasterCLARA-5 | 6.7 | 14 | 16 | 12 |
| LS-k-means++-10 | 17 | 22 | 24 | 21 |
| LS-k-means++-5 | 21 | 26 | 24 | 24 |
| kmc2-200 | 33 | 30 | 28 | 30 |
| kmc2-20 | 32 | 30 | 29 | 30 |
| kmc2-100 | 35 | 31 | 27 | 31 |
| k-means++ | 40 | 29 | 26 | 32 |
| Random | 34 | 32 | 30 | 32 |
Avg
</details>

Figure 3: RT and $\Delta$ RO for Bankruptcy

![](images/fff6320a7792dee2e262ccd87fd2648c1153eaa4b9cc194773b4d4d591680c81.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| k-means++ | 0 | 1.7 | 3.2 | 1.63 |
| FasterCLARA-5 | 0.6 | 2.2 | 4.3 | 2.37 |
| kmc2-20 | 0.5 | 4.4 | 8.7 | 4.53 |
| OneBatch-nniw | 12.6 | 13.6 | 14.4 | 13.5 |
| OneBatch-lwcs | 12.7 | 14.2 | 13.9 | 13.6 |
| OneBatch-debias | 16.7 | 13.5 | 13.9 | 14.7 |
| OneBatch-unif | 13.8 | 16.9 | 15.4 | 15.4 |
| LS-k-means++-5 | 0.5 | 14.2 | 50.9 | 21.9 |
| kmc2-100 | 5.2 | 21.8 | 43.4 | 23.5 |
| FasterCLARA-50 | 8.7 | 24.1 | 44.7 | 25.8 |
| LS-k-means++-10 | 1.4 | 27.8 | 105 | 44.7 |
| kmc2-200 | 12.6 | 43.3 | 87.9 | 47.9 |
| Alternate | 98.5 | 98.8 | 87.7 | 95 |
| FasterPAM | 100 | 100 | 100 | 100 |
| BanditPAM++-0 | 102 | 904 | 999 | 949 |
| BanditPAM++-2 | 292 | 999 | 999 | 999 |
| BanditPAM++-5 | 529 | 999 | 999 | 999 |
</details>

![](images/64d11b7c0d63ba8b4f68a5f4fe5389756840aa78779692a3d3eb00c28b85dca1.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | FasterPAM | BanditPAM++-5 | OneBatch-nniw | BanditPAM++-2 | BanditPAM++-0 | OneBatch-debias | OneBatch-lwcs | OneBatch-unif | Alternate | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | LS-k-means++-5 | kmc2-20 | Random | k-means++ | kmc2-200 | kmc2-100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 0 | 0 | 0 | 0 | 0 | 0.6 | 0.7 | 0.3 | 2.9 | 6.1 | 7.9 | 14 | 19 | 20 | 21 | 23 | 23 | 25 |
| ΔRO (in %) | 0 | 0 | 0 | 0 | 0 | 0.6 | 0.7 | 0.3 | 2.9 | 6.1 | 7.9 | 14 | 19 | 20 | 21 | 18 | 18 | 17 |
| Avg |
| Color scale: 0 to 30
Color bar: purple = 0, dark purple = 0
Color bar: green = 15, yellow = 25, green = 18, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = 17, green = nan |
K=50, K=100, K=150, K=180, K=210, K=240, K=270, K=300, K=330, K=360, K=390, K=420, K=450, K=480, K=510, K=540, K=570, K=600, K=630, K=660, K=690, K=720, K=750, K=780, K=810, K=840, K=870, K=900, K=930, K=960, K=990, K=1020, K=1050, K=1080, K=1110, K=1140, K=1170, K=1200, K=1230, K=1260, K=1290, K=1320, K=1350, K=1380, K=1410, K=1440, K=1470, K=1500, K=1530, K=1560, K=1590, K=1620, K=1650, K=1680, K=1710, K=1740, K=1770, K=1800, K=1830, K=1860, K=1890, K=2020 |
K=50, K=100, K=150, K=180, K=210, K=240, K=270, K=300, K=330, K=360, K=390, K=420, K=450, K=480, K=510, K=540, K=580, K=610, K=640, K=670, K=710, K=740, K=770, K=800, K=830, K=860, K=890, K=920, K=950, K=980, K=1010, K=1040, K=1070, K=1100 |
KmC2-20: 25
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
kmc2-20: 23
</details>

Figure 4: RT and $\Delta$ RO for Mapping

![](images/c9f930622ec8ca32bc776d09c7bb3680dd97d23bb01565ef1dfa9e149035a51a.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| k-means++ | 0 | 1.4 | 2.6 | 1.33 |
| FasterCLARA-5 | 0.7 | 1.8 | 3.2 | 1.9 |
| kmc2-20 | 1.1 | 4.3 | 8.3 | 4.57 |
| OneBatch-debias | 10.7 | 12.9 | 12.2 | 11.9 |
| OneBatch-lwcs | 11.5 | 12.5 | 12.5 | 12.2 |
| OneBatch-unif | 14.5 | 13.2 | 12.1 | 13.3 |
| OneBatch-nniw | 15 | 14.5 | 13.5 | 14.3 |
| FasterCLARA-50 | 8.2 | 17.5 | 31.6 | 19.1 |
| kmc2-100 | 4 | 21.1 | 39.9 | 21.7 |
| LS-k-means++-5 | 0.5 | 19.6 | 75.4 | 31.8 |
| kmc2-200 | 8.1 | 42 | 87.5 | 45.9 |
| LS-k-means++-10 | 1.2 | 40.2 | 150 | 63.7 |
| FasterPAM | 100 | 100 | 100 | 100 |
| Alternate | 470 | 302 | 192 | 321 |
| BanditPAM++-0 | 224 | 833 | 999 | 858 |
| BanditPAM++-2 | 476 | 999 | 999 | 999 |
| BanditPAM++-5 | 873 | 999 | 999 | 999 |
Avg
</details>

![](images/ebb4f445b2ef2b596421e538ed78c25b8bfb91648b246d25b922196e4b020be7.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | FasterPAM | OneBatch-nniw | BanditPAM++-5 | BanditPAM++-2 | BanditPAM++-0 | OneBatch-debias | OneBatch-unif | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | LS-k-means++-5 | k-means++ | kmc2-100 | OneBatch-lwcs | Alternate | kmc2-20 | kmc2-200 | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 0 | 0.1 | 3.3 | 3.5 | 3.8 | 2.1 | 2.2 | 2.7 | 5.7 | 6.9 | 8.5 | 35 | 41 | 51 | 1.9 | 41 | 50 | 99 |
| K | 0 | 1.8 | 2.1 | 3.2 | 3.6 | 7.7 | 7.2 | 14 | 16 | 19 | 27 | 33 | 39 | 38 | 58 | 42 | 38 | 99 |
| K | 0 | 5.3 | 2.1 | 3.9 | 5.6 | 10 | 11 | 20 | 28 | 29 | 32 | 37 | 43 | 36 | 66 | 44 | 40 | 99 |
| Avg | 0 | 2.4 | 3.1 | 4.1 | 4.7 | 6.7 | 6.8 | 12 | 16 | 18 | 22 | 35 | 41 | 42 | 42 | 42 | 42 | 99 |
</details>

Figure 5: RT and $\Delta$ RO for Drybean

![](images/c496ed62d66c1a160ad461aa243a4a9df5dd89efb32a98951c891969f8228477.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| k-means++ | 0 | 0.9 | 1.7 | 0.867 |
| FasterCLARA-5 | 0.4 | 1.1 | 1.8 | 1.1 |
| kmc2-20 | 0.3 | 1.9 | 3.6 | 1.93 |
| OneBatch-lwcs | 7.6 | 7.9 | 7.8 | 7.77 |
| OneBatch-debias | 8 | 8.4 | 7.5 | 7.97 |
| OneBatch-nniw | 8.4 | 8.9 | 8.1 | 8.47 |
| OneBatch-unif | 7.4 | 9.6 | 9.3 | 8.77 |
| kmc2-100 | 1.7 | 9.5 | 17.6 | 9.6 |
| FasterCLARA-50 | 3.9 | 11.1 | 19 | 11.3 |
| kmc2-200 | 4.5 | 18.9 | 35.9 | 19.8 |
| LS-k-means++-5 | 0.7 | 13.2 | 48 | 20.6 |
| LS-k-means++-10 | 1.2 | 26.4 | 93.4 | 40.3 |
| FasterPAM | 100 | 100 | 100 | 100 |
| Alternate | 152 | 138 | 131 | 140 |
| BanditPAM++-0 | 66.8 | 672 | 999 | 733 |
| BanditPAM++-2 | 154 | 999 | 999 | 999 |
| BanditPAM++-5 | 282 | 999 | 999 | 999 |
Avg
</details>

![](images/ea0ebadede8a3bd492276d0bf3c577fa1410d79bac33d2a493438258e011eab1.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | FasterPAM | BanditPAM++-5 | BanditPAM++-2 | OneBatch-nniw | BanditPAM++-0 | OneBatch-debias | OneBatch-unif | OneBatch-lwcs | Alternate | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | LS-k-means++-5 | kmc2-20 | kmc2-100 | k-means++ | Random | kmc2-200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 0 | 0.8 | 1.2 | 0.5 | 2 | 1 | 0.8 | 1.8 | 6.6 | 8.6 | 11 | 17 | 20 | 24 | 25 | 25 | 27 | 25 |
| 50 | 0 | 1.2 | 1.7 | 1.6 | 1.9 | 3.3 | 3.1 | 4.1 | 9.9 | 12 | 14 | 20 | 22 | 24 | 24 | 25 | 26 | 26 |
| 100 | 0 | 2.7 | 2.4 | 3.2 | 2.4 | 5.5 | 6 | 6.3 | 12 | 15 | 16 | 24 | 25 | 27 | 27 | 28 | 28 | 27 |
| Avg | 0 | 1.6 | 1.8 | 1.8 | 2.1 | 3.3 | 3.3 | 4.1 | 9.4 | 12 | 14 | 21 | 23 | 25 | 26 | 26 | 26 | 27 |
</details>

Figure 6: RT and $\Delta$ RO for Letter

Table 7: Relative Time (RT) per dataset for the “large scale” experiments. The scores are averaged over the five repetitions of the experiment and the three values of $k \in [10, 50, 100]$ . RT is given in percentage. The standard deviations are reported in brackets. 

<table><tr><td>Methods\Datasets</td><td>cifar</td><td>covertype</td><td>dota2</td><td>mnist</td><td>monitor-gas</td></tr><tr><td>Random</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td><td>0.0 (0.0)</td></tr><tr><td>FasterCLARA-5</td><td>19.8 (0.4)</td><td>14.3 (0.0)</td><td>12.4 (0.3)</td><td>15.1 (1.0)</td><td>13.4 (0.8)</td></tr><tr><td>FasterCLARA-50</td><td>193.7 (1.3)</td><td>169.6 (0.2)</td><td>144.8 (2.1)</td><td>159.4 (3.6)</td><td>140.9 (8.9)</td></tr><tr><td>kmc2-20</td><td>0.4 (0.0)</td><td>0.1 (0.0)</td><td>1.0 (0.0)</td><td>0.4 (0.0)</td><td>0.5 (0.0)</td></tr><tr><td>kmc2-100</td><td>2.1 (0.2)</td><td>0.5 (0.0)</td><td>5.0 (0.3)</td><td>1.7 (0.3)</td><td>2.4 (0.2)</td></tr><tr><td>kmc2-200</td><td>4.2 (0.2)</td><td>1.4 (0.0)</td><td>11.3 (0.9)</td><td>3.3 (0.1)</td><td>5.8 (0.3)</td></tr><tr><td>k-means++</td><td>21.0 (0.3)</td><td>76.0 (2.3)</td><td>17.8 (0.2)</td><td>17.2 (1.3)</td><td>261.9 (16.2)</td></tr><tr><td>LS-k-means++-5</td><td>24.8 (0.6)</td><td>94.8 (4.1)</td><td>53.7 (0.5)</td><td>26.0 (1.2)</td><td>286.3 (3.9)</td></tr><tr><td>LS-k-means++-10</td><td>29.1 (1.6)</td><td>105.8 (5.8)</td><td>88.9 (0.2)</td><td>35.3 (1.5)</td><td>348.7 (5.4)</td></tr><tr><td>OneBatch-lwcs</td><td>100.4 (0.4)</td><td>135.4 (9.7)</td><td>101.6 (2.0)</td><td>118.4 (22.7)</td><td>135.0 (5.0)</td></tr><tr><td>OneBatch-unif</td><td>116.9 (19.0)</td><td>117.5 (12.6)</td><td>95.2 (7.6)</td><td>92.1 (2.2)</td><td>99.1 (2.5)</td></tr><tr><td>OneBatch-debias</td><td>109.9 (6.4)</td><td>114.1 (12.8)</td><td>81.0 (3.8)</td><td>92.8 (2.1)</td><td>102.3 (6.6)</td></tr><tr><td>OneBatch-nniw</td><td>100.0 (0.3)</td><td>100.0 (2.9)</td><td>100.0 (6.1)</td><td>100.0 (7.5)</td><td>100.0 (3.8)</td></tr></table>

Table 8: Delta Relative Objective ( $\Delta$ RO) per dataset for the “large scale” experiments. The scores are averaged over the five repetitions of the experiment and the three values of $k \in [10, 50, 100]$ . $\Delta$ RO is given in percentage. The standard deviations are reported in brackets. 

<table><tr><td>Methods\Datasets</td><td>cifar</td><td>covertype</td><td>dota2</td><td>mnist</td><td>monitor-gas</td></tr><tr><td>Random</td><td>16.8 (1.4)</td><td>24.9 (3.4)</td><td>12.2 (1.9)</td><td>15.5 (1.6)</td><td>31.9 (3.5)</td></tr><tr><td>FasterCLARA-5</td><td>7.9 (0.6)</td><td>9.7 (0.8)</td><td>3.8 (0.3)</td><td>8.1 (0.5)</td><td>10.7 (1.0)</td></tr><tr><td>FasterCLARA-50</td><td>7.3 (0.5)</td><td>8.5 (0.8)</td><td>3.2 (0.1)</td><td>7.1 (0.3)</td><td>9.2 (0.6)</td></tr><tr><td>kmc2-20</td><td>18.1 (2.8)</td><td>22.2 (3.5)</td><td>8.9 (1.6)</td><td>16.0 (1.8)</td><td>25.8 (2.1)</td></tr><tr><td>kmc2-100</td><td>17.2 (1.5)</td><td>22.0 (2.3)</td><td>8.8 (2.2)</td><td>15.5 (2.1)</td><td>24.2 (3.1)</td></tr><tr><td>kmc2-200</td><td>19.2 (3.2)</td><td>22.5 (3.2)</td><td>8.6 (1.9)</td><td>16.4 (2.2)</td><td>26.1 (2.7)</td></tr><tr><td>k-means++</td><td>19.0 (2.5)</td><td>21.7 (2.9)</td><td>8.6 (2.4)</td><td>16.3 (1.4)</td><td>26.2 (4.4)</td></tr><tr><td>LS-k-means++-5</td><td>17.2 (2.4)</td><td>17.7 (2.0)</td><td>6.2 (0.9)</td><td>13.7 (1.4)</td><td>21.8 (2.5)</td></tr><tr><td>LS-k-means++-10</td><td>16.2 (1.9)</td><td>15.8 (1.7)</td><td>5.6 (0.6)</td><td>12.6 (1.8)</td><td>18.1 (2.7)</td></tr><tr><td>OneBatch-lwcs</td><td>-0.3 (0.2)</td><td>3.8 (1.1)</td><td>0.7 (0.3)</td><td>0.8 (0.3)</td><td>8.6 (1.3)</td></tr><tr><td>OneBatch-unif</td><td>0.3 (0.2)</td><td>1.7 (0.5)</td><td>0.6 (0.2)</td><td>0.9 (0.2)</td><td>2.4 (0.9)</td></tr><tr><td>OneBatch-debias</td><td>-0.7 (0.1)</td><td>1.9 (0.5)</td><td>0.2 (0.1)</td><td>0.6 (0.2)</td><td>2.1 (0.6)</td></tr><tr><td>OneBatch-nniw</td><td>0.0 (0.2)</td><td>0.0 (0.2)</td><td>0.0 (0.1)</td><td>0.0 (0.2)</td><td>0.0 (0.6)</td></tr></table>

![](images/7cf7003d08fa20263c30ed14c37bc8f64aabd69200c48bd10eb1e146a2288a66.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| kmc2-20 | 0 | 0.2 | 0.9 | 0.367 |
| kmc2-100 | 0.1 | 1 | 5.3 | 2.13 |
| kmc2-200 | 0.2 | 2 | 10.5 | 4.23 |
| FasterCLARA-5 | 4.9 | 19.2 | 35.2 | 19.8 |
| k-means++ | 4.2 | 20 | 38.8 | 21 |
| LS-k-means++-5 | 6.8 | 23.6 | 43.9 | 24.8 |
| LS-k-means++-10 | 9.3 | 26.6 | 51.4 | 29.1 |
| OneBatch-nniw | 100 | 100 | 100 | 100 |
| OneBatch-lwcs | 101 | 100 | 100 | 100 |
| OneBatch-debias | 110 | 110 | 110 | 110 |
| OneBatch-unif | 120 | 116 | 114 | 117 |
| FasterCLARA-50 | 51.2 | 187 | 343 | 194 |
Avg
</details>

![](images/27f11bcb83b2815afb619b60c0a79af75ecd60e70f9931122b380b463e6d7f06.jpg)  
Figure 7: RT and $\Delta$ RO for CIFAR

![](images/389723190366f576f6a6f036da4c0aa76fdfa53ddba93fb5f4182b2f7b7d46d0.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
| :--- | :--- | :--- | :--- | :--- |
| Random | 0 | 0 | 0 | 0 |
| kmc2-20 | 0.1 | 0.3 | 0.7 | 0.367 |
| kmc2-100 | 0.3 | 1.4 | 3.5 | 1.73 |
| kmc2-200 | 0.5 | 2.7 | 6.7 | 3.3 |
| FasterCLARA-5 | 3.9 | 13.7 | 27.7 | 15.1 |
| k-means++ | 3.5 | 16.5 | 31.6 | 17.2 |
| LS-k-means++-5 | 5.8 | 22.4 | 49.9 | 26 |
| LS-k-means++-10 | 7.8 | 28.8 | 69.3 | 35.3 |
| OneBatch-unif | 92.8 | 91 | 92.6 | 92.1 |
| OneBatch-debias | 95 | 90 | 93.5 | 92.8 |
| OneBatch-nniw | 100 | 100 | 100 | 100 |
| OneBatch-lwcs | 130 | 112 | 114 | 118 |
| FasterCLARA-50 | 45.3 | 149 | 284 | 159 |
Avg
</details>

![](images/be3295111f0722b72d509fb073051c0210a3400bfe632cc56301dafaecfde195.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | OneBatch-nniw | OneBatch-debias | OneBatch-lwcs | OneBatch-unif | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | LS-k-means++-5 | Random | kmc2-100 | kmc2-20 | k-means++ | kmc2-200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | 0 | 0.3 | 0.5 | 0.3 | 5.3 | 6.6 | 12 | 14 | 18 | 18 | 19 | 20 | 20 |
| ΔRO (in %) | 0 | 0.6 | 0.5 | 0.5 | 8.1 | 9 | 13 | 14 | 15 | 15 | 16 | 16 | 16 |
| Avg |
| :---: |
| OneBatch-nniw: 0:0, OneBatch-debias: 0.3, OneBatch-lwcs: 0.5, OneBatch-unif: 0.3, FasterCLARA-50: 5.3, FasterCLARA-5: 6.6, LS-k-means++-10: 12, LS-k-means++-5: 14, Random: 18, kmc2-100: 18, kmc2-20: 19, k-means++: 20, kmc2-200: 20 |
| :---: |
| OneBatch-nniw: 0:0, OneBatch-debias: 0.5, OneBatch-lwcs: 0.3, OneBatch-unif: 0.3, FasterCLARA-50: 5.3, FasterCLARA-5: 6.6, LS-k-means++-10: 12, LS-k-means++-5: 14, Random: 18, kmc2-100: 18, kmc2-20: 19, k-means++: 20, kmc3-200: 20 |
| :---: |
| OneBatch-nniw: 0:0, OneBatch-debias: 0.5, OneBatch-lwcs: 0.3, OneBatch-unif: 0.3, FasterCLARA-50: 5.3, FasterCLARA-5: 6.6, LS-k-means++-10: 12, LS-k-means++-5 : 14, Random: 18, kmc2-100: 18, kmc2-20: 19, k-means++: 20, kmc2-200: 20 |
| :---: |
| OneBatch-nniw: 0:0, OneBatch-debias: 0.8, OneBatch-lwcs: 1.3, OneBatch-unif: 2 | .0.57 | .77 | .93 | .71 | .8.6 | .8.1 | .12 | .14 | .15 | .15 | .16 | .16 |
| .---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93, FasterCLARA-50: .8.1, FasterCLARA-5: .8.1, LS-k-means++-10: .12, LS-k-means++-5: .14, Random: .15, kmc2-100: .15, kmc2-20: .16, k-means++: .16, kmc2-200: .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93, FasterCLARA-50: .8.1, FasterCLARA-5: .8.1, LS-k-means++-10: .12, LS-k-means++-5 : .14, Random: .15, kmc2-100: .15, kmc2-20: .16, k-means++: .16, kmc3-200: .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93, FasterCLARA-50: .8.1, FasterCLARA-5: .8.1, LS-k-means++-10: .12, LS-k-means++-5 : .14 , Random : .15 , kmc2-100 : .15 , kmc2-20 : .16 , k-means++ : .16 , kmc2-200 : .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93 , FasterCLARA-50: .8.1 , FasterCLARA-5: .8.1 , LS-k-means++ -10 : .12 , LS-k-means++ -5 : .14 , Random : .15 , kmc2-100 : .15 , kmc2-20 : .16 , k-means++ : .16 , kmc3-200 : .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93 , FasterCLARA-50: .8.1 , FasterCLARA-5: .8.1 , LS-k-means++ -10 : .12 , LS-k-means++ -5 : .14<fcel>.0.57 | .77 | .93 | .71 | .8.6 | .8.1 | .12 | .14 | .15 | .15 | .16 | .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93 , FasterCLARA-50: .8.1 , FasterCLARA-5: .8.1 , LS-k-means++ -10 : .12 , LS-k-means++ -5 : .14 , Random : .16 , kmc2-100 : .15 , kmc2-20 : .16 , k-means++ : .16 , kmc3-200 : .16 |
| :---: |
| OneBatch-nniw: .0, OneBatch-debias: .0.8, OneBatch-lwcs: .77, OneBatch-unif: .93 , FasterCLARA-50: .8.1, FasterCLARA-5: .8.1 , LS-k-means++ -10 : .12 , LS-k-means++ -5 : .14 , Random : .16 , kmc2-100 : .15 , kmc2-20 : .16 , k-means++ : .16 , kmc3-200 : .16 |
| :---: |
| OneBatch-nniw: .4, OneBatch-debias: 4.8, OneBatch-lwcs: 4.8, OneBatch-unif: 4.8, FasterCLARA-50: 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 | 4.8 |
| :---: |
| OneBatch-nniw: -4, OneBatch-debias: -4.8, OneBatch-lwcs: -4.8, OneBatch-unif: -4.8, FasterCLARA-50: -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 |
| :---: |
| OneBatch-nniw: -4, OneBatch-debias: -4.8, OneBatch-lwcs: -4.8, OneBatch-unif: -4.8, FasterCLARA-50: -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.8 | -4.<ecel><ecel><ecel><ecel><ecel><ecel><nl>
</details>

Figure 8: RT and $\Delta$ RO for MNIST

![](images/112b962fc2d9f914844d528253632b5f07d70b9be6cc4469ba62b0f7caaec17f.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
|---|---|---|---|---|
| Random | 0 | 0 | 0 | 0 |
| kmc2-20 | 0.2 | 1 | 1.9 | 1.03 |
| kmc2-100 | 1.5 | 4.4 | 9.2 | 5.03 |
| kmc2-200 | 1.9 | 8.9 | 23 | 11.3 |
| FasterCLARA-5 | 4 | 10.6 | 22.7 | 12.4 |
| k-means++ | 3.6 | 17.8 | 31.9 | 17.8 |
| LS-k-means++-5 | 7.4 | 41.7 | 112 | 53.7 |
| OneBatch-debias | 80.8 | 83.1 | 79.2 | 81 |
| LS-k-means++-10 | 10 | 65 | 192 | 88.9 |
| OneBatch-unif | 95.8 | 97.6 | 92.2 | 95.2 |
| OneBatch-nniw | 100 | 100 | 100 | 100 |
| OneBatch-lwcs | 102 | 104 | 99.2 | 102 |
| FasterCLARA-50 | 49.8 | 141 | 244 | 145 |
Avg
</details>

![](images/3bc330c2f2ebfba1f1de3dd1332cd9756ed33551ad2d80da3592fe18277e6f9d.jpg)  
Figure 9: RT and $\Delta$ RO for Dota2

![](images/16e1cb13f98478109e0078ed0439078e2f5a958c8e174cb604479ad59ab54ece.jpg)

![](images/35ce977ef3c65a732bafa1db78c1610f32c4f2f9b70f513536f92ae15fffae40.jpg)  
Figure 10: RT and $\Delta$ RO for Monitor-gas

![](images/de3021a13282cfd35a19b2d445d893ab65ccbb7101d664f6d4edad024534173a.jpg)

<details>
<summary>heatmap</summary>

RT (in %)
| method | 10 | 50 | 100 | Avg |
|---|---|---|---|---|
| Random | 0 | 0 | 0 | 0 |
| kmc2-20 | 0 | 0.1 | 0.3 | 0.133 |
| kmc2-100 | 0 | 0.4 | 1.2 | 0.533 |
| kmc2-200 | 0.1 | 0.7 | 3.4 | 1.4 |
| FasterCLARA-5 | 3 | 13.4 | 26.5 | 14.3 |
| k-means++ | 14.9 | 71 | 142 | 76 |
| LS-k-means++-5 | 24.7 | 84 | 176 | 94.8 |
| OneBatch-nniw | 100 | 100 | 100 | 100 |
| LS-k-means++-10 | 32.3 | 91.5 | 194 | 106 |
| OneBatch-debias | 111 | 113 | 118 | 114 |
| OneBatch-unif | 120 | 118 | 114 | 117 |
| OneBatch-lwcs | 133 | 135 | 138 | 135 |
| FasterCLARA-50 | 30.2 | 134 | 345 | 170 |
Avg
</details>

![](images/e7d086d6102002f8da08f974d8f24f45f65a8371ba6a1c915c862ab5d6f1b7ad.jpg)

<details>
<summary>heatmap</summary>

ΔRO (in %)
| | OneBatch-nniw | OneBatch-unif | OneBatch-debias | OneBatch-lwcs | FasterCLARA-50 | FasterCLARA-5 | LS-k-means++-10 | LS-k-means++-5 | k-means++ | kmc2-100 | kmc2-20 | kmc2-200 | Random |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | 0 | 0.9 | 1.1 | 4.5 | 6.6 | 8.8 | 16 | 18 | 27 | 28 | 30 | 28 | 31 |
| | 0 | 1.7 | 2 | 4.3 | 9.7 | 11 | 16 | 18 | 21 | 20 | 20 | 21 | 24 |
| | 0 | 2.4 | 2.5 | 4.3 | 9.3 | 9.4 | 15 | 17 | 18 | 18 | 17 | 18 | 20 |
| Avg |
| :---:0
| :---:0
OneBatch-nniw: 0, OneBatch-unif: 0.9
OneBatch-debias: 1.1
OneBatch-lwcs: 4.5
FasterCLARA-50: 6.6
FasterCLARA-5: 8.8
LS-k-means++-10: 16
LS-k-means++-5: 18
k-means++: 27
kmc2-100: 28
kmc2-20: 30
kmc2-200: 28
Random: 31
Color scale: 0 to 30
Yellow = >25
Green = >20
Purple = >15
Light green = >10
Dark purple = >5
</details>

Figure 11: RT and $\Delta$ RO for Covertype

# Appendix D. Pareto Front

This section presents the Pareto front (in red) for Objective vs Time graphs for each dataset and the two configurations k = 10 and k = 100. Algorithms belonging to the Pareto front are “optimal” for at least one objective/time trade-off. In contrast, the algorithms out of the Pareto front are “suboptimal” because another algorithm provides a better objective with less running time.

We observe that, for the small-scale datasets, k-means++, FasterCLARA-5, OneBatch-nniw and FasterPAM belong to the Pareto fronts. The Pareto fronts for the large-scale datasets include kmc2-20, FasterCLARA-5 and OneBatch-nniw.

![](images/8b3d4573b4864171d05cd533e6b4ee33d77db16888913646bf17d0ab2babdab5.jpg)

<details>
<summary>scatter</summary>

| Method              | Time (log sec) | Objective |
| ------------------- | -------------- | --------- |
| k-means++           | -3.5           | 1.15      |
| LS-k-means++-5      | -3.2           | 1.03      |
| FasterCLARA-5       | -2.5           | 0.94      |
| FasterCLARA-50      | -1.2           | 0.91      |
| OneBatch-nniw        | -0.8           | 0.88      |
| k-means++           | -1.0           | 1.21      |
| kmc2-100            | -0.5           | 1.21      |
| kmc2-20             | -0.3           | 1.20      |
| kmc2-200            | -0.1           | 1.20      |
</details>

Figure 12: Objective vs Time: Abalone (k = 10)

![](images/7dea7611f8917581586bf1cd98b707d71fc86e19bba3883fa5c197b62e900966.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| k-means++        | -1.5           | 0.27      |
| FasterCLARA-5    | -1.2           | 0.23      |
| FasterCLARA-50   | -0.5           | 0.22      |
| FasterPAM        | -0.8           | 0.19      |
| OneBatch-nniw     | -1.0           | 0.20      |
| LS-k-means++-10  | -0.2           | 0.26      |
| LS-k-means++-5   | -0.3           | 0.26      |
| OneBatch-nniw     | -0.4           | 0.20      |
| k-means++        | 0.5            | 0.28      |
| kmc2-100         | 0.3            | 0.27      |
| kmc2-20          | 0.4            | 0.28      |
| kmc2-200         | 0.5            | 0.28      |
| BanditPAM++-0    | 1.8            | 0.21      |
| BanditPAM++-2    | 2.1            | 0.21      |
| BanditPAM++-5    | 2.3            | 0.21      |
</details>

Figure 13: Objective vs Time: Abalone (k = 100)

![](images/c901db6560cf2cefcd3e7a33551ec2e9618b2b789fb477b86f8c69bc0e6c973c.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective     |
| ---------------- | -------------- | ------------- |
| kmc2-20          | -1.6           | 1.11e10       |
| FasterCLARA-5    | -1.4           | 0.90e10       |
| FasterCLARA-5    | -0.3           | 0.89e10       |
| OneBatch-nniw     | -0.3           | 0.84e10       |
| k-means++        | -1.5           | 1.17e10       |
| kmc2-100         | -0.9           | 1.14e10       |
| kmc2-20          | 1.0            | 0.84e10       |
| kmc2-200         | -0.5           | 1.12e10       |
| BanditPAM++-0    | 1.2            | 0.85e10       |
| BanditPAM++-2    | 1.5            | 0.84e10       |
| BanditPAM++-5    | 1.6            | 0.84e10       |
| LS-k-means++-10  | -1.3           | 1.02e10       |
| LS-k-means++-5   | -1.2           | 0.99e10       |
| OneBatch-nniw     | -0.3           | 0.84e10       |
</details>

Figure 14: Objective vs Time: Bankruptcy (k = 10)

![](images/d0672337748ac0d15f0f4e858a1ee0119bd21f1ab8d9f4dffbf20ab1f67e2b96.jpg)

<details>
<summary>scatter</summary>

| Method              | Time (log sec) | Objective |
| ------------------- | -------------- | --------- |
| FasterCLARA-5      | -0.5           | 5.17e9    |
| OneBatch-nniw       | -0.1           | 4.6e9     |
| FasterPAM           | 1.0            | 4.45e9    |
| kmc2-200            | 0.5            | 5.7e9     |
| kmc2-100            | 0.3            | 5.7e9     |
| k-means++           | -0.4           | 5.6e9     |
| LS-k-means++-10     | 0.5            | 5.5e9     |
| LS-k-means++-5      | 0.2            | 5.5e9     |
| Other methods      | 2.3            | 4.7e9     |
| BanditPAM++-0       | 2.4            | 4.7e9     |
| BanditPAM++-2       | 2.5            | 4.7e9     |
| BanditPAM++-5       | 2.6            | 4.65e9    |
| Alternative         | 1.0            | 5.05e9    |
</details>

Figure 15: Objective vs Time: Bankruptcy (k = 100)

![](images/3f0cd76b791dfca5e316e9433ee83bda32504ddeb9444a79cdb4da7b259caa08.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| k-means++        | -3.0           | 47000     |
| kmc2-20          | -1.8           | 46000     |
| LS-k-means++-5   | -1.9           | 45500     |
| FasterCLARA-5    | -1.7           | 41000     |
| FasterCLARA-50   | -0.5           | 40500     |
| OneBatch-nniw     | -0.3           | 38500     |
| Alternative       | 0.5            | 39500     |
| BanditPAM++-0    | 1.2            | 39000     |
| BanditPAM++-2    | 1.6            | 38500     |
| BanditPAM++-5    | 1.8            | 38200     |
| LS-k-means++-10  | -1.5           | 43500     |
| LS-k-means++-5   | -1.6           | 45500     |
| k-means++         | -3.2           | 47000     |
| kmc2-100         | -0.8           | 48000     |
| kmc2-20          | -1.9           | 46000     |
| kmc2-200         | -1.1           | 47000     |
</details>

Figure 16: Objective vs Time: Mapping (k = 10)

![](images/c31b5659ff2163dd0f9a403c4eddacd8a43f206157bf9b83297406095130e91f.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| k-means++        | -1.0           | 32200     |
| kmc2-100         | 0.5            | 32200     |
| kmc2-20          | 0.5            | 32200     |
| kmc2-200         | 0.5            | 32200     |
| LS-k-means++-10  | 0.5            | 31900     |
| LS-k-means++-5   | 0.5            | 32100     |
| OneBatch-nniw     | -0.5           | 28100     |
| BanditPAM++-2    | 2.7            | 28500     |
| BanditPAM++-5    | 2.9            | 28400     |
| Alternative       | 0.5            | 29800     |
| FasterCLARA-5    | -0.8           | 30300     |
| FasterCLARA-5    | 0.2            | 30200     |
| FasterCLARA-5    | 0.5            | 27600     |
</details>

Figure 17: Objective vs Time: Mapping (k = 100)

![](images/0aae013fbfa9ba42a6a6069b8aaf922e4b6d11f8489e049f57c4384f2e5536af.jpg)

<details>
<summary>scatter</summary>

| Method              | Time (log sec) | Objective |
| ------------------- | -------------- | --------- |
| k-means++           | -3.0           | 6500      |
| LS-k-means++-5      | -1.8           | 5200      |
| FasterCLARA-5       | -1.5           | 5100      |
| FasterCLARA-50      | -0.5           | 4900      |
| LS-k-means++-10     | -1.2           | 5200      |
| LS-k-means++-5      | -0.8           | 4800      |
| OneBatch-nniw        | -0.3           | 4800      |
| k-means++           | -3.0           | 6500      |
| kmc2-100            | -1.0           | 6700      |
| kmc2-20             | -1.5           | 6700      |
| kmc2-200            | -0.5           | 7200      |
| Alternative          | 1.2            | 4900      |
| BanditPAM++-0       | 1.8            | 5000      |
| BanditPAM++-2       | 2.1            | 5000      |
| BanditPAM++-5       | 2.3            | 5000      |
| FasterCLARA-5       | -1.7           | 5100      |
| FasterCLARA-50      | -0.7           | 4900      |
| FasterPAM           | 0.4            | 4800      |
</details>

Figure 18: Objective vs Time: Drybean (k = 10)

![](images/8445178cf2fe8df19edb88ceda7eeedeac7b3d12154ae28346a7749dfcc14f55.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| Alternate        | 0.8            | 950       |
| BanditPAM++-0    | 2.6            | 615       |
| BanditPAM++-2    | 2.8            | 610       |
| BanditPAM++-5    | 3.0            | 600       |
| FasterCLARA-5    | -0.9           | 735       |
| FasterCLARA-50   | 0.1            | 690       |
| FasterPAM        | 0.6            | 580       |
| LS-k-means++-10  | 0.7            | 745       |
| LS-k-means++-5   | 0.5            | 760       |
| OneBatch-nniw     | -0.3           | 605       |
| k-means++         | -1.0           | 790       |
| kmc2-100         | 0.2            | 825       |
| kmc2-20          | -0.5           | 830       |
| kmc2-200         | 0.5            | 805       |
</details>

Figure 19: Objective vs Time: Drybean (k = 100)

![](images/87fc9b25581512edb39a19138f795b2463d3fee8c8591d84a2c46c9c03608365.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| k-means++        | -3.0           | 24.8      |
| kmc2-20          | -1.5           | 24.1      |
| kmc2-20          | -0.5           | 24.9      |
| kmc2-20          | 0.5            | 19.7      |
| kmc2-20          | 1.0            | 20.7      |
| kmc2-20          | 1.5            | 19.8      |
| kmc2-20          | 2.0            | 19.6      |
| kmc2-20          | -1.0           | 22.7      |
| kmc2-20          | -0.5           | 21.1      |
| kmc2-20          | -1.5           | 23.3      |
| kmc2-20          | -0.5           | 21.5      |
| kmc2-20          | -1.0           | 22.7      |
| kmc2-20          | -0.5           | 21.1      |
| kmc2-20          | -1.5           | 23.3      |
| kmc2-20          | -0.5           | 24.9      |
| kmc2-20          | -1.0           | 24.3      |
| kmc2-20          | -0.5           | 21.5      |
| kmc2-20          | -1.5           | 23.3      |
| kmc2-20          | -0.5           | 24.9      |
| kmc2-20          | -1.0           | 24.3      |
| kmc2-20          | -0.5           | 21.5      |
| kmc2-20          | 0.5            | 19.7      |
| kmc2-20          | 1.0            | 19.7      |
| kmc2-20          | 1.5            | 19.7      |
| kmc2-20          | 2.0            | 19.7      |
| kmc2-20          | -1.0           | 24.3      |
| kmc2-20          | -0.5           | 24.3      |
| kmc2-20          | -1.5           | 24.3      |
| kmc2-20          | -0.5           | 24.3      |
| kmc2-20          | -1.5           | 24.3      |
| kmc2-20          | -0.5           | 24.3      |
| kmc2-20          | -1.5           | 24.3      |
| kmeans++         | -3.0           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.5           | 24.8      |
| kmeans++         | -0.8           | 24.8      |
| kmeans++         | -1.0           | 24.8      |
| kmeans++         | -0.5           | 24.8      |
| Kmc2-100         | -3.0           | 24.8      |
| Kmc2-100         | -1.0           | 24.8      |
| Kmc2-100         | -0.5           | 24.8      |
| Kmc2-100         | -1.5           | 24.8      |
| Kmc2-100         | -0.5           | 24.8      |
| Kmc2-100         | -1.5           | 24.8      |
| Kmc2-100         | -0.8           | 24.8      |
| Kmc2-100         | -1.5           | 24.8      |
| Kmc2-100         | -0.8           | 24.8      |
| Kmc2-100         | -1.5           | 24.8      |
| Kmc2-100         | -0.8           | 24.8                      |
| Kmc2-100         | -1.5           | 24.8      |
| Kmc2-100         | -0.8           | 24.8                      |
| Kmc2-100         | -1.5           | 24.8                      |
| Kmc2-100         | -0.8           | 24.8                      |
| Kmc2-100         | -1.5           | 24.8                      |
| Kmc2-100         | -0.8           | 24.8                      |
| Kmc2-100         | -1.5           | 24.8                      |
| Kmc2-15          | -3.0           | 24.8      |
| Kmc2-15          | -1.5           | 24.8      |
| Kmc2-15          | -1.5           | 24.8      |
| Kmc2-15          | -1.5           | 24.8      |
| Kmc2-15          | -1.5           | 24.8      |
| Kmc2-15          | -1.5           | 24.8      |
| Kmclx-marks+       +   | -3.0           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.0           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   | -1.5           | 24.8      |
| Kmclx-marks+       +   (Note: The objective value is estimated based on the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value (Note: The objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of each other (Note: The objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of the objective value of each other (Note : Note: The objective value in parentheses is not specified in this case).    Note: The example is a sample from a single data point (e.g., “k-means+”, “kmc-” or “kmc-”) but not included in the original table.
</details>

Figure 20: Objective vs Time: Letter (k = 10)

![](images/691a32bb44cb62e6a0c0aef60dae7ba3f15dea1beeab8b80efca45bd9caf62b5.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| k-means++        | -0.8           | 15.0      |
| FasterCLARA-5    | -0.7           | 13.8      |
| OneBatch-nniw     | -0.2           | 12.3      |
| FasterPAM        | 0.9            | 11.9      |
| kmc2-100         | 0.2            | 15.1      |
| kmc2-20          | 0.4            | 15.1      |
| kmc2-200         | 0.6            | 14.9      |
| LS-k-means++-10  | 0.6            | 14.9      |
| LS-k-means++-5   | 0.6            | 14.8      |
| LS-k-means++-10  | 0.6            | 14.7      |
| LS-k-means++-5   | 0.6            | 14.8      |
| BanditPAM++-0    | 2.8            | 12.2      |
| BanditPAM++-2    | 3.0            | 12.2      |
| BanditPAM++-5    | 3.2            | 12.2      |
| FasterCLARA-5    | -0.8           | 13.8      |
| FasterCLARA-5    | 0.2            | 13.6      |
| FasterCLARA-5    | 0.6            | 13.6      |
| FasterCLARA-5    | 0.6            | 13.6      |
| k-means++        | -0.8           | 15.0      |
| kmc2-100         | 0.2            | 15.1      |
| kmc2-20          | -0.5           | 15.0      |
| kmc2-200         | 0.4            | 15.1      |
| Alternative       | 1.0            | 13.3      |
| Other methods    | 3.0            | 12.2      |
</details>

Figure 21: Objective vs Time: Letter (k = 100)

![](images/89e89be9cab8468088cc1907fd36726a8262065c8d29226d8839b11c6088470e.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | -1.4           | 635       |
| kmc2-100         | -0.7           | 615       |
| FasterCLARA-5    | 0.9            | 540       |
| FasterCLARA-50   | 1.9            | 535       |
| LS-k-means++-10  | 1.1            | 605       |
| LS-k-means++-5   | 1.0            | 610       |
| OneBatch-nniw    | 2.1            | 505       |
| k-means++        | 0.8            | 630       |
| kmc2-100         | -0.6           | 640       |
</details>

Figure 22: Objective vs Time: CIFAR (k = 10)

![](images/15c7b0f3cdb8ec07ba6ac87c0ef57a5888aee7f3abffa2388206190a25ca7950.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | 0.3            | 516       |
| FasterCLARA-5    | 1.8            | 491       |
| FasterCLARA-50   | 2.7            | 490       |
| LS-k-means++-10  | 1.9            | 518       |
| LS-k-means++-5   | 1.9            | 519       |
| OneBatch-nniw    | 2.2            | 455       |
| k-means++        | 1.8            | 520       |
| kmc2-100         | 1.0            | 517       |
| kmc2-20          | 0.3            | 516       |
| kmc2-200         | 1.3            | 518       |
</details>

Figure 23: Objective vs Time: CIFAR (k = 100)

![](images/5a6c74e14d0501dea4b6c29f2f9038405d150fe231bc177f1ca28a13e17a2f86.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| FasterCLARA-5    | 0.3            | 84.0      |
| FasterCLARA-50   | 1.4            | 83.0      |
| LS-k-means++-10  | 0.6            | 88.5      |
| LS-k-means++-5   | 0.5            | 90.2      |
| OneBatch-nniw    | 1.7            | 79.0      |
| k-means++        | 0.3            | 94.5      |
| kmc2-20          | -1.1           | 94.0      |
| kmc2-100         | -0.8           | 93.5      |
| kmc2-200         | -0.6           | 94.8      |
</details>

Figure 24: Objective vs Time: MNIST (k = 10)

![](images/017932fceb1a0a47c1349d85f2f5dc8f3f0525b088445e13eba83345fff60ad8.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | 0.0            | 69.0      |
| FasterCLARA-5    | 1.3            | 66.5      |
| FasterCLARA-50   | 2.2            | 66.0      |
| LS-k-means++-10  | 1.7            | 69.0      |
| LS-k-means++-5   | 1.5            | 69.0      |
| OneBatch-nniw    | 1.8            | 61.0      |
| k-means++        | 1.3            | 69.5      |
| kmc2-100         | 0.4            | 69.5      |
| kmc2-20          | 0.0            | 69.0      |
| kmc2-200         | 0.6            | 69.0      |
</details>

Figure 25: Objective vs Time: MNIST (k = 100)

![](images/172558224a90ae7e0085e1838b8a2746e67258db9343642b64a07d9e4da52c3d.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | -1.5           | 24.0      |
| kmc2-100         | -0.5           | 23.8      |
| FasterCLARA-5    | -0.1           | 22.0      |
| FasterCLARA-50   | 1.0            | 21.7      |
| LS-k-means++-10  | 0.1            | 22.7      |
| LS-k-means++-5   | 0.3            | 22.4      |
| OneBatch-nniw     | 1.2            | 21.1      |
| k-means++        | -0.2           | 24.0      |
</details>

Figure 26: Objective vs Time: Dota2 (k = 10)

![](images/0c4b1b0cbb9b28c64010d2f46cfaff14b2f1a5a8170b28a67d7b5f4b875a4666.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | 0.0            | 18.2      |
| kmc2-100         | 0.3            | 18.15     |
| FasterCLARA-5    | 0.7            | 17.75     |
| FasterCLARA-5    | 1.7            | 17.7      |
| LS-k-means++-10  | 1.6            | 17.98     |
| LS-k-means++-5   | 1.4            | 18.02     |
| OneBatch-nniw    | 1.3            | 17.15     |
| k-means++        | 0.9            | 18.08     |
| kmc2-100         | 0.3            | 18.15     |
| kmc2-20          | 0.0            | 18.2      |
| kmc2-200         | 0.7            | 18.15     |
</details>

Figure 27: Objective vs Time: Dota2 (k = 100)

![](images/2c02cd8cc10ed6f5210c71435d12d424e59bfc955438c5e31944b6d492f4bf25.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | -1.8           | 117       |
| kmc2-100         | -1.2           | 116       |
| kmc2-50          | 1.6            | 107       |
| FasterCLARA-5    | 0.0            | 99        |
| FasterCLARA-50   | 1.0            | 97        |
| LS-k-means++-10  | 1.5            | 113       |
| LS-k-means++-5   | 1.5            | 113       |
| OneBatch-nniw    | 1.6            | 93        |
| k-means++        | 1.4            | 121       |
monitor_gas - K = 10
</details>

Figure 28: Objective vs Time: Monitor-gas (k = 10)

![](images/789e6649db5196848dca92ee578867d666c3c8367dc26b592e047b7447fc962c.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | 0.0            | 58.0      |
| kmc2-100         | 0.5            | 56.8      |
| FasterCLARA-5    | 1.1            | 52.5      |
| FasterCLARA-50   | 2.1            | 52.0      |
| LS-k-means++-10  | 2.4            | 56.5      |
| LS-k-means++-5   | 2.4            | 56.3      |
| OneBatch-nniw     | 1.7            | 47.0      |
| k-means++        | 2.4            | 57.2      |
| kmc2-20          | 0.0            | 58.0      |
| kmc2-200         | 0.9            | 57.8      |
</details>

Figure 29: Objective vs Time: Monitor-gas (k = 100)

![](images/896e93e440dc85a554ad32c39bfcd1dead973e647c10eb5ae3d9dc1f07163d9e.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | -1.8           | 1730      |
| kmc2-100         | -1.0           | 1715      |
| FasterCLARA-5    | 0.8            | 1455      |
| FasterCLARA-50   | 1.8            | 1425      |
| LS-k-means++-10  | 1.7            | 1580      |
| LS-k-means++-5   | 1.6            | 1590      |
| OneBatch-nniw    | 2.3            | 1340      |
| k-means++        | 1.5            | 1695      |
</details>

Figure 30: Objective vs Time: Covertype (k = 10)

![](images/85f5af9deeca8d81d3646629e695964d3d6ee5afc0ab954d7c38387907ea5822.jpg)

<details>
<summary>scatter</summary>

| Method           | Time (log sec) | Objective |
| ---------------- | -------------- | --------- |
| kmc2-20          | 0.0            | 890       |
| FasterCLARA-5    | 1.8            | 830       |
| FasterCLARA-50   | 2.9            | 828       |
| LS-k-means++-10  | 2.6            | 875       |
| LS-k-means++-5   | 2.6            | 885       |
| OneBatch-nniw    | 2.4            | 758       |
| k-means++        | 2.5            | 892       |
| kmc2-100         | 0.5            | 892       |
| kmc2-20          | 0.0            | 890       |
| kmc2-200         | 1.0            | 890       |
</details>

Figure 31: Objective vs Time: Covertype (k = 100)