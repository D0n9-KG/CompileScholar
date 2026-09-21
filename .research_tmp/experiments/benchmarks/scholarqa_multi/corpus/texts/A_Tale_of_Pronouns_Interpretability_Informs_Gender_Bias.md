# HARD LEFSCHETZ THEOREMS FOR FREE LINE BUNDLES

JIAJUN HU, SHIJIE SHANG, JIAN XIAO

ABSTRACT. We introduce a partial positivity notion for algebraic maps via the defect of semismallness. This positivity notion is modeled on m-positivity in the analytic setting and m-ampleness in the geometric setting. Using this positivity condition for algebraic maps, we establish Kähler packages, that is, Hard Lefschetz theorems and Hodge-Riemann bilinear relations, for the complete intersections of Chern classes of free line bundles.

# CONTENTS

1. Introduction 1   
2. Partial positivity for maps and free bundles 4   
3. Hironaka's principle of counting constants 8   
4. Proof of the main theorem 12   
References 14

# 1. INTRODUCTION

In this paper, we work on the field of complex numbers $\mathbb{C}$ .

1.1. Motivation. Given a mathematical object X of “dimension” n, a Kähler package on X usually consists of a triple $(A(X), P(X), K(X))$ , where $A(X)$ is a graded vector space (or even a graded algebra) constructing from X, $P(X)$ is a bilinear paring on $A(X)$ and $K(X)$ is a family of linear operators acting on $A(X)$ of degree one. The triple is called a Kähler package if it satisfies Poincaré duality, hard Lefschetz theorem and Hodge-Riemann bilinear relation. For example, given $q \leq n/2$ , the hard Lefschetz theorem for X means the following statement: for any $L_{1}, ..., L_{n-2q} \in K(X)$ , the linear map

$$
A ^ {q} (X) \to A ^ {n - q} (X), \xi \mapsto \left(\prod_ {k = 1} ^ {n - 2 q} L _ {k}\right) \cdot \xi
$$

is an isomorphism. In last decades, novel and exciting Kähler packages were discovered and have been playing key roles in algebra, combinatorics and geometry, see e.g. [Huh18, Huh22, Wil18] and the references therein.

As a sequel to [HX22], we are interested in the question: for which kinds of $L_{1}, \ldots, L_{n-2q}$ the hard Lefschetz theorem and Hodge-Riemann bilinear relation will hold. One motivation is that in certain problems one indeed needs to study Kähler packages without strong positivity assumption. Though in different context, see e.g. Adiprasito's recent breakthrough on the g-conjecture of McMullen in full generality [Adi18] where hard Lefschetz theorem beyond positivity is an essential ingredient.

We focus on the model and original case - the Kähler package on a compact Kähler manifold, which could be the prototype for later development. Let $X$ be a compact Kähler manifold of dimension $n$ and $\omega$ a Kähler class on $X$ . As a fundamental piece of Hodge theory, for any integers $0 \leq p, q \leq p + q \leq n$ , the complete intersection class

$$
\Omega = \omega^ {n - p - q} \in H ^ {n - p - q, n - p - q} (X, \mathbb {R}),
$$

has the following properties:

(HL): The linear map

$$
\Omega : H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q, n - p} (X, \mathbb {C}), \phi \mapsto \Omega \cdot \phi
$$

is an isomorphism.

(HR): The quadratic form $Q$ on $H^{p,q}(X,\mathbb{C})$ , defined by

$$
Q (\varphi_ {1}, \varphi_ {2}) = c _ {p, q} \Omega \cdot \varphi_ {1} \cdot \overline {{\varphi_ {2}}}, \text {   where   } c _ {p, q} = \mathrm{i} ^ {q - p} (- 1) ^ {(p + q) (p + q + 1) / 2},
$$

is positive definite on the primitive space $\mathrm{Prim}^{p,q}(X,\mathbb{C})$ with respect to $(\Omega ,\omega)$ :

$$
\operatorname{Prim} ^ {p, q} (X, \mathbb {C}) = \{\phi \in H ^ {p, q} (X, \mathbb {C}) | \Omega \cdot \omega \cdot \phi = 0. \}
$$

We call that the class $\Omega$ has HL property and the pair $(\Omega, \omega)$ has HR property.

By replacing the above $\Omega = \omega^{n - p - q}$ by an arbitrary cohomology class $\Omega \in H^{n - p - q,n - p - q}(X,\mathbb{R})$ and $\omega$ by an arbitrary (1,1) class $\eta \in H^{1,1}(X,\mathbb{R})$ , it is interesting to study:

When does the class $\Omega$ have HL property and when does the pair $(\Omega, \eta)$ have HR property? (In order to define the primitive space we need the class $\Omega$ coupling with an $(1, 1)$ class $\eta$ .)

A complete characterization of such $\Omega$ seems unreachable at this moment, nevertheless, in the same spirit of [HX22] we first study a subclass of $\Omega$ , coming from complete intersections:

Question 1.1. Let X be a compact Kähler manifold of dimension n, and let $\alpha_{1},\ldots,\alpha_{n-p-q},\eta\in H^{1,1}(X,\mathbb{R})$ , then under which assumptions does the complete intersection class

$$
\Omega = \alpha_ {1} \cdot \dots . \cdot \alpha_ {n - p - q}
$$

have HL property and does the pair $(\Omega,\eta)$ have HR property?

In [HX22], under a mild positivity assumption (nefness) on the $(1,1)$ classes, the first and third named authors gave a complete characterization of $\Omega$ on a compact complex torus:

Theorem 1.2 (Theorem A of [HX22]). Let $X = \mathbb{C}^n / \Gamma$ be a compact complex torus of dimension $n$ and $0 \leq p, q \leq p + q \leq n$ . Let $\alpha_1, \ldots, \alpha_{n - p - q}, \eta \in H^{1,1}(X, \mathbb{R})$ be nef classes on $X$ . Denote $\Omega = \alpha_1 \cdot \ldots \cdot \alpha_{n - p - q}$ . Then for HL property, the following statements are equivalent:

(1) the intersection class $\Omega$ has HL property;   
(2) for any subset $I \subset [n - p - q]$ , $\mathrm{nd}(\alpha_I) \geq |I| + p + q$ .

For HR property, the following statements are equivalent:

(1) the pair $(\Omega, \eta)$ has HR property for $\eta$ with $\mathrm{nd}(\eta) \geq p + q$ ;   
(2) for any subset $I \subset [n - p - q]$ , $\mathrm{nd}(\alpha_I) \geq |I| + p + q$ .

Here, $[k]$ is the finite set $\{1,2,\dots,k\}$ for a given positive integer $k$ , $\mathrm{nd}(-)$ is the numerical dimension of nef classes, $\alpha_{I} = \sum_{i\in I}\alpha_{i}$ and $|I|$ is the cardinality of $I$ .

As a consequence of Theorem 1.2, we obtain new kinds of cohomology classes on an arbitrary compact Kähler manifold, which have HL and HR properties.

Corollary 1.3 (Corollary A of [HX22]). Let X be a compact Kähler manifold of dimension n and $0 \leq p, q \leq p + q \leq n$ . Let $\alpha_{1}, \ldots, \alpha_{n-p-q}, \eta \in H^{1,1}(X, \mathbb{R})$ be nef classes on X. Denote $\Omega = \alpha_{1} \cdots \alpha_{n-p-q}$ . Assume that there exists a smooth semi-positive representative $\widehat{\alpha}_{i}$ in each class $\alpha_{i}$ , such that for any subset $I \subset [n - p - q]$ , $\widehat{\alpha}_{I}$ is $|I| + p + q$ positive in the sense of forms, and that there exists a smooth semi-positive representative $\widehat{\eta}$ in $\eta$ , such that $\widehat{\eta}$ is $p + q$ positive, then

- the complete intersection class $\Omega$ has HL property;   
- the pair $(\Omega, \eta)$ has HR property.

The above results greatly generalize [Xia21, DN06, Cat08, Tim98] by allowing degenerate positivity for each $(1,1)$ class. A smooth semi-positive $(1,1)$ form is m-positive if and only if its coefficient matrix has at least m positive eigenvalues. As a typical example, if $f: X \to Y$ is a submersion from X to a compact Kähler manifold Y of dimension m, then for any Kähler class $\omega_{Y}$ on Y, the pullback $f^{*}\omega_{Y}$ is m-positive on X.

From the geometric viewpoint of analytic/algebraic maps, in the above example $f$ being a submersion looks quite restrictive. Given a proper surjective holomorphic map $g: X \to Y$ , even if we assume that

$$
\dim g ^ {- 1} (y) \leq n - m
$$

for any $y \in Y$ , it is not sufficient to guarantee the $m$ -positivity of $g^{*}\omega_{Y}$ . On the other hand, it is easy to see that the requirement on numerical dimensions in Theorem 1.2 is not sufficient for a general

variety. Nevertheless, de Cataldo-Migliorini [dCM02] proved the following hard Lefschetz property: if $L$ is a line bundle on a complex projective manifold $X$ of dimension $n$ , such that a positive power of $L$ is generated by its global sections, then the class $\Omega = c_{1}(L)^{n - p - q}$ has HL property if and only if $L$ is lef, in the sense that $mL = f^{*}A$ for some projective semismall morphism $f: X \to Y$ , where $A$ is an ample line bundle on $Y$ . In particular, the pair $(\Omega, c_{1}(L))$ has HR property. The map $f: X \to Y$ is called semismall if for every $k \geq 0$ ,

$$
\dim Y ^ {k} + 2 k \leq \dim X,
$$

where $Y^{k} = \{y\in Y|\dim f^{-1}(y) = k\}$ . This result had been playing important roles in their geometric study of Hodge theory of algebraic maps [dCM02, dCM05].

1.2. The main result. Inspired by [dCM02] and a proposal proposed by the third named author [Xia21], from the viewpoint of Kähler packages, we make the following analog:

<table><tr><td>forms/classes</td><td>bundles</td><td>maps</td></tr><tr><td>m-positivity</td><td>(n-m)-ample</td><td>m-lef</td></tr></table>

The notion $m$ -positivity is frequently studied in geometric partial differential equations, and the $m$ -ampleness of a line bundle was studied by Sommese [Som78]. The $m$ -lefness for maps is defined via the defect of semismallness:

Definition 1.4. Let $X$ be an analytic variety and $L$ a free line bundle on $X$ , then we call the line bundle $L$ $m$ -lef if the Kodaira map

$$
\Phi_ {L}: X \to Y _ {L} \subset \mathbb {P} (H ^ {0} (X, L))
$$

is m-lef in the sense that the defect of semi-smallness of $\Phi_{L}$ ,

$$
r (\Phi_ {L}) = \max _ {i} \{\dim Y _ {L} ^ {i} + 2 i - \dim X \} \leq \dim X - m
$$

where $Y_{L}$ is the image of X, $Y_{L}^{i} = \{y \in Y | \dim f^{-1}(x) = i\}$ and we set $\dim Y_{L}^{i} = -\infty$ if $Y_{L}^{i} = \emptyset$ .

In particular, if $m = \dim X$ , then this is exactly the notion of lefness introduced by de Cataldo-Migliorini [dCM02]. By definitions, it is easy to see: if the free line bundle L is ( $\dim X - m$ )-ample, then it must be m-lef.

Remark 1.5. It is natural to extend Definition 1.4 to the analytic setting as follows. Let $X$ be a compact Kähler manifold and $\alpha \in H^{1,1}(X,\mathbb{R})$ , then we call that $\alpha$ is $m$ -lef if there is a proper surjective holomorphic map $f: X \to Y$ to a Kähler variety $Y$ such that

- $f$ is $m$ -lef;   
- $\alpha = f^{*}\omega_{Y}$ for some Kähler class $\omega_{Y}$ on $Y$ .

For notional simplicity, we use the following notation: let $L_{1}, \ldots, L_{k}$ be line bundles, then the complete intersection of their Chern classes $c_{1}(L_{1}) \cdot \ldots \cdot c(L_{k})$ is simply denoted by $L_{1} \cdot \ldots \cdot L_{k}$ .

Inspired by Corollary 1.3 and using the notion of $m$ -lefness, we prove the following result:

Theorem A. Let $X$ be a smooth projective variety of dimension $n$ and let $0 \leq p, q \leq p + q \leq n$ be integers. Assume that $L_{1}, \ldots, L_{n - p - q}, M$ are free line bundles such that

- $L_I$ is $|I| + p + q$ lef for any $I \subset [n - p - q]$ ,   
- $M$ is $p + q$ lef,

then the following statements hold:

(1) the complete intersection class $\Omega = L_{1} \cdot \ldots \cdot L_{n - p - q}$ has HL property, i.e., the linear map

$$
\Omega : H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q, n - p} (X, \mathbb {C})
$$

is an isomorphism.

(2) the pair $(\Omega, M)$ has HR property, i.e., the quadratic form $Q$ on $H^{p,q}(X, \mathbb{C})$ ,

$$
Q (\alpha , \beta) = c _ {p, q} \Omega \cdot \alpha \cdot \overline {{\beta}}, \alpha , \beta \in H ^ {p, q} (X, \mathbb {C}), c _ {p, q} = \mathrm{i} ^ {q - p} (- 1) ^ {(p + q) (p + q + 1) / 2},
$$

is positive definite on the primitive space

$$
\operatorname{Prim} ^ {p, q} (X) = \ker \{\Omega \cdot M: H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q + 1, n - p + 1} (X, \mathbb {C}) \}.
$$

Remark 1.6. When p = q = 1, by using Theorem A and the dually Lorentzian polynomials [RSW23], one can obtain a bunch of generalized Alexandrov-Fenchel inequalities.

As the converse to Question 1.1, analogous to Theorem 1.2, it is interesting to study the converse of Theorem A:

If the complete intersection class $\Omega = L_1 \cdot \ldots \cdot L_{n - p - q}$ has HL property, then what can we say about the positivity of the free bundles $L_i$ ?

Regarding this problem, we note that the HL property for certain degree is sufficient to characterize the m-lefness of a free line bundle, see Theorem 2.16 for details.

We expect that Theorem A extends to the analytic setting as mentioned in Remark 1.5. However, the geometric arguments cannot easily extend to this analytic setting. We hope to return to this issue in the future.

The paper is organized as follows. In Section 2, we introduce the notion m-lefness for maps and bundles, and study its basic properties. In Section 3, we study a variant of Hironaka's principle of counting constants for free line bundles and the restriction of m-lef line bundles on a hypersurface, which will be a key ingredient in the proof of the main result. Section 4 is devoted to the proof of Theorem A.

Acknowledgements. This work is supported by the National Key Research and Development Program of China (No. 2021YFA1002300) and National Natural Science Foundation of China (No. 11901336). We would like to thank Izzet Coskun and Zhiyu Tian for helpful discussions on Hironaka's original paper on the principle of counting constants, and thank Mark Andrea de Cataldo and Julius Ross for helpful comments.

# 2. PARTIAL POSITIVITY FOR MAPS AND FREE BUNDLES

The key notion in our study of partial positivity for maps is the defect of semismallness introduced by Goresky-MacPherson [GM88].

Definition 2.1. Let $f: X \to Y$ be a proper surjective holomorphic map between two complex analytic varieties, the defect of semi-smallness of $f$ is defined by

$$
r (f) = \max _ {i} \{\dim Y ^ {i} + 2 i - \dim X \}
$$

where $Y^{i}=\{y\in Y|\dim f^{-1}(x)=i\}$ and we set $\dim Y^{i}=-\infty$ if $Y^{i}=\emptyset$ .

The map $f$ is called semismall if $r(f) = 0$ . Therefore, the number $r(f)$ measures the deviation of $f$ from semismallness.

It is clear that each $Y^{i}$ is a locally closed analytic subvariety of Y, whose disjoint union is Y. By letting $i = \dim X/Y$ (the dimension of a general fiber), we see that

$$
r (f) \geq \dim X / Y \geq 0.
$$

By the definition of defect of semi-smallness, we have that

$$
r (f) = \max _ {T \subset X} \{2 \dim T - \dim f (T) - \dim X \}, \tag {1}
$$

where $T$ ranges over all irreducible analytic subvarieties (including $X$ itself) of $X$ .

In the study of Kähler packages with respect to degenerate positivity, the third named author [Xia21] proposed an analog of m-positivity for analytic maps. Here, we make it more precise:

Definition 2.2. Let $f: X \to Y$ be a proper surjective holomorphic map between two complex analytic varieties and $0 \leq m \leq \dim X$ , then $f$ is called $m$ -lef if

$$
r (f) \leq \dim X - m.
$$

It is called exact $m$ -lef if $r(f) = \dim X - m$ .

By definition, $f$ is semismall if and only if it is (dim $X$ )-lef.

Lemma 2.3. Let $f: X \to Y$ , $g: X \to Z$ and $\pi: Y \to Z$ be proper surjective holomorphic maps between complex analytic varieties such that $g = \pi \circ f$ , then $r(f) \leq r(g)$ . Moreover, if $\pi$ is a finite morphism, then $r(f) = r(g)$ .

Proof. Note that for any irreducible analytic subvariety $T$ in $X$ ,

$$
\dim g (T) = \dim \pi \circ f (T) \leq \dim f (T),
$$

therefore by (1) we have $r(f) \leq r(g)$ .

Furthermore, assume that $\pi$ is a finite morphism, then for any $i \geq 0$ , $Z^{i} \subset \pi(Y^{i})$ , which yields that $r(g) \leq r(f)$ . Thus, using the above inequality, $r(f) = r(g)$ . ☐

Let $X$ be a projective variety and $|W|$ a linear series on $X$ . We denote the Kodaira map corresponding to $W$ by

$$
\Phi_ {W}: X \dashrightarrow \mathbb {P} (W),
$$

and denote $Y_{W}$ the Zariski closure of $\Phi_{W}(X \setminus \operatorname{Bs}|W|)$ . In particular, if the linear series is given by a line bundle L, we shall use the notations $\Phi_{L}, Y_{L}$ .

Definition 2.4. Let $X$ be a smooth projective variety and $L$ a free line bundle on $X$ , then we call the line bundle $L$ $m$ -lef if the Kodaira map

$$
\Phi_ {L}: X \to Y _ {L} \subset \mathbb {P} (H ^ {0} (X, L))
$$

is $m$ -lef. The bundle $L$ is called exact $m$ -lef if $\Phi_L$ is exact $m$ -lef.

When $m = \dim X$ , this is exactly the notion lefness introduced by de Cataldo-Migliorini [dCM02] in their study of decomposition theorem for semismall maps.

Remark 2.5. By [EV89], Kodaira-Akizuki-Nakano type vanishing theorems hold with respect to m-lefness: let X be a smooth projective variety and let L be m-lef on X, then

$$
H ^ {p, q} (X, L ^ {- 1}) = 0
$$

whenever $p + q < m$ .

Proposition 2.6. Assume that the line bundle L is m-lef, then its numerical dimension $\mathrm{nd}(L) \geq m$ .

Proof. The follows by letting $i = \dim X / Y_L$ (the dimension of a general fiber of $\Phi_L$ ) in the formula of $r(\Phi_L)$ . Then we get $\dim Y_L \geq m$ , which means that the Kodaira dimension of $L$ is at least $m$ .

For a free line bundle, note that its Kodaira dimension and numerical dimension coincide.

![](images/535b31aeb64d0a91421e85df49e88262f23923e260d41928492cee678bd86d26.jpg)

Remark 2.7. By Proposition 2.6, the positivity assumption in Theorem A implies the requirement on numerical dimensions as in Theorem 1.2.

The following result is useful in the comparison of partial lefness for different free linear series.

Lemma 2.8. Let X be a smooth projective variety, and $|W|$ , $|V|$ two free linear series on X. Assume that $|W| \subset |V|$ , then the two morphisms $\Phi_{W}: X \to \mathbb{P}(W)$ , $\Phi_{V}: X \to \mathbb{P}(V)$ differ by a finite projection of $\Phi_{V}(X)$ , that is, $\Phi_{W} = \pi \circ \Phi_{V}$ where $\pi$ is a finite morphism.

Proof. This follows from [Laz04, Example 1.1.12].

![](images/2101bb7cf9dfa750e1ea52c519bcdefbae64d366f830a41f75c406c547ee5357.jpg)

For example, let $L$ be a free line bundle, by Lemma 2.8, it is clear that the following are equivalent:

- $L$ is $m$ -lef;   
- $kL$ is $m$ -lef for some positive integer $k$ ;   
- $kL$ is $m$ -lef for any positive integer $k$ .

Proposition 2.9. Any nontrivial free line bundle L is m-lef for some $m \geq 1$ .

Proof. Consider the Kodaira map

$$
\Phi_ {L}: X \to Y _ {L} \subset \mathbb {P} (H ^ {0} (X, L)).
$$

It is easy to see that

$$
2 \dim T - \dim \Phi_ {L} (T) \leq 2 \dim X - 1
$$

holds for any irreducible subvariety $T$ of $X$ . Therefore, any nontrivial free line bundle is 1-lef. Indeed, $L$ is exact $m$ -lef, where

$$
m = \min _ {T \subset X} \left\{2 \dim X - 2 \dim T + \dim \Phi_ {L} (T) \right\}.
$$

![](images/ef64ef73e163acfe7d04d6e2670db071e7b2bd6e3b730a9fe6d05be0cccbe738.jpg)

Therefore, the notion m-lefness gives a filtration on the positivity of free line bundles.

Lemma 2.10. If K is k-lef and L is l-lef, then $K + L$ is $\max(k, l)$ -lef.

Proof. By comparing the linear series $|K + L|$ and $|K| \otimes |L|$ and using Lemma 2.8, it suffices to show that the map

$$
f: X \to Y _ {K} \times Y _ {L}, x \mapsto (\Phi_ {K} (x), \Phi_ {L} (x)),
$$

is $\max (k,l)$ -lef onto its image. Applying Lemma 2.3, we have that

$$
r (f) \leq \min \{r (\Phi_ {K}), r (\Phi_ {L}) \}.
$$

This yields that $f$ is $\max(k, l)$ -lef.

![](images/9d76683adf20e1539d596eee44df0f26520e9af2e6e8df87f4cb8ac21b5217f6.jpg)

The following Lefschetz hyperplane theorem due to Goresky-MacPherson [GM88, Section 2.3] is useful for us.

Lemma 2.11. Let X be a smooth projective variety and let L be a r-lef free line bundle on X. Then for a general hypersurface $V \in |L|$ , the restriction map

$$
i ^ {*}: H ^ {l} (X, \mathbb {C}) \to H ^ {l} (V, \mathbb {C})
$$

is injective for $l \leq r - 1$ and isomorphic for $l \leq r - 2$ .

As a direct application, we get the following Bertini theorem for 2-lef line bundles.

Lemma 2.12. Let $X$ be a smooth projective variety and let $L$ be a 2-lef free line bundle on $X$ . Then a general hypersurface $V \in |L|$ is smooth and irreducible.

Remark 2.13. We give a brief discussion on the background for the analog:

$$
\left| \begin{array}{c c c} \text {forms / classes} & \text {bundles} & \text {maps} \\ \hline m \text {-positivity} & (n - m) \text {-ample} & m \text {-lef} \end{array} \right|,
$$

and the motivation for m-lefness. Let $\omega$ be a Kähler class on a compact Kähler manifold X of dimension n and let $\widehat{\omega}$ be a Kähler metric in the class $\omega$ . Then $\alpha \in H^{1,1}(X,\mathbb{R})$ is called m-positive with respect to $\widehat{\omega}$ , if $\alpha$ has a smooth representative $\widehat{\alpha}$ such that for any $1 \leq k \leq m$ ,

$$
\widehat {\alpha} ^ {k} \wedge \widehat {\omega} ^ {n - k} > 0 \tag {2}
$$

in the sense of forms. If we further assume that $\widehat{\alpha}$ is semipositive, (2) is equivalent to that $\widehat{\alpha}$ has at least m positive eigenvalues everywhere with respect to $\widehat{\omega}$ . In particular, a free line bundle L is called m-positive with respect to $\widehat{\omega}$ , if its Chern class $c_{1}(L)$ is m-positive with respect to $\widehat{\omega}$ . In [Xia21], the third named author proved the following result:

Let $0 \leq p, q \leq p + q \leq m \leq n$ be integers, and let $\alpha_{1}, \ldots, \alpha_{m - p - q + 1} \in H^{1,1}(X, \mathbb{R})$ be semipositive and $m$ -positive classes, then the class $\Omega = \omega^{n - m} \cdot \alpha_{1} \cdot \ldots \cdot \alpha_{m - p - q}$ has HL property and the pair $(\Omega, \alpha_{m - p - q + 1})$ has HR property.

Applying the result to the case when every $\alpha_{k}$ coming from a submersion $f: X \to Y$ motivates essentially the following notion of partial lefness for line bundles [Xia21, Section 4]: a free line bundle L is called m-lef if the Kodaira map

$$
\Phi_ {L}: X \to Y _ {L}
$$

satisfies that

- the numerical dimension of $L$ , $\operatorname{nd}(L) \geq m$ ;   
- the defect of semi-smallness of $f$ , $r(f) \leq n - \dim Y_L$ .

In general, this notion is stronger than Definition 2.4.

In [Som78], Sommese introduced the notion m-ampleness for line bundles. Let X be a smooth projective variety of dimension n and L a line bundle on X, the line bundle L is called m-ample if there exists some $k \in N$ such that $\dim \operatorname{Bs}_{|kL|} \leq m$ and the Kodaira map

$$
\Phi_ {k L}: X \setminus \operatorname{Bs} _ {| k L |} \to Y _ {k L} \subset \mathbb {P} (H ^ {0} (X, k L))
$$

satisfies that $\dim\Phi_{kL}^{-1}(x)\leq m$ for any x in the image. In particular, if L is free, then L is m-ample if and only if $\dim\Phi_{L}^{-1}(x)\leq m$ for any $x\in Y_{L}$ .

In the case when $\Phi_{kL}$ is a submersion, we have:

$$
L \text {   being   } (n - m) \text {-ample   } \Rightarrow c _ {1} (L) \text {   being   semipositive   and   } m \text {-positive. }
$$

By Definition 2.4, for free bundles we also have:

$$
L \text {   being   } (n - m) \text {-ample } \Rightarrow L \text {   being   } m \text {-lef. }
$$

Next, similar to [dCM02] we give a characterization of m-lefness by the hard Lefschetz property.

Proposition 2.14. Let X be a smooth projective variety of dimension n and L a free line bundle on X. Assume that L is m-lef, then for any ample line bundles $A_{1}, \ldots, A_{n-m}$ , the class $A_{1} \cdot \ldots \cdot A_{n-m} \cdot L^{m-p-q}$ has HL property for any $0 \leq p, q \leq p + q \leq m$ , that is, the map

$$
A _ {1} \cdot \ldots \cdot A _ {n - m} \cdot L ^ {m - p - q}: H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q, n - p} (X, \mathbb {C})
$$

is an isomorphism.

Proof. For $p + q = m$ , the result is clear, thus we need only to deal with the cases $p + q \leq m - 1$ . After taking multiples of the line bundles we can assume that $A_1, \ldots, A_{n - m}$ are very ample.

Take a general smooth hypersurface $H_{i} \in |A_{i}|$ for each $i \leq n - m$ , then by [dCM05, Section 4], the restriction $\Phi_{L}$ on $V = H_{1} \cap \ldots \cap H_{n-m}$ has vanishing defect of semismallness, in particular it must be semismall. This implies that the restriction of L on V, denoted by $L_{|V}$ , is lef.

Assume that $\varphi \in H^{p + q}(X,\mathbb{C})$ satisfies

$$
A _ {1} \cdot \ldots \cdot A _ {n - m} \cdot L ^ {m - p - q} \cdot \varphi = 0,
$$

we need to prove that $\varphi = 0$ . To this end, note that this is equivalent to that

$$
A _ {1} \cdot \dots \cdot A _ {n - m} \cdot L ^ {m - p - q} \cdot \varphi \cdot \psi = L _ {| V} ^ {m - p - q} \cdot \varphi_ {| V} \cdot \psi_ {| V} = 0 \tag {3}
$$

for any $\psi \in H^{p + q}(X,\mathbb{C})$ . By an inductive application of the Lefschetz hyperplane theorem, we obtain that the map

$$
i _ {V}: H ^ {p + q} (X, \mathbb {C}) \to H ^ {p + q} (V, \mathbb {C}), \psi \mapsto \psi_ {| V},
$$

is an isomorphism whenever $p + q \leq m - 1$ . Combining with (3), we get that

$$
L _ {| V} ^ {m - p - q} \cdot \varphi_ {| V} = 0
$$

on V.

By [dCM02], since $L_{|V}$ is lef on V, it has HL property. Therefore, $\varphi_{|V}=0$ . Another application of the Lefschetz hyperplane theorem implies that $\varphi=0$ , finishing the proof.

![](images/a4b9dee022b89472e86d0c8e5610e44809c47f1ed223d1d0302efc2fde37b8a7.jpg)

Proposition 2.15. Let X be a smooth projective variety and L a free line bundle on X. Let $A_{1}, \ldots, A_{n-m}$ be ample line bundles on X. Assume that the class $A_{1} \cdot \ldots \cdot A_{n-m} \cdot L^{m-p-q}$ has HL property for any $0 \leq p, q \leq p + q \leq m$ , then L is m-lef.

Proof. Since $L$ is free, we may assume that $L = \Phi_L^* A$ for some ample line bundle $A$ on $Y_L$ .

We argue by contradiction. Otherwise, $L$ is not $m$ -lef, which means that there is some irreducible subvariety $T$ in $X$ such that

$$
2 \dim T - 2 n + m > \dim f (T).
$$

This implies that there exists a cycle in the class $A^{2\dim T - 2n + m}$ , which is disjoint with $f(T)$ . Thus, there is a cycle in the class

$$
L ^ {2 \dim T - 2 n + m} = \Phi_ {L} ^ {*} A ^ {2 \dim T - 2 n + m}
$$

which is disjoint with $T$ , yielding that

$$
A _ {1} \cdot \ldots \cdot A _ {n - m} \cdot L ^ {m - 2 (n - \dim T)} \cdot [ T ] = 0.
$$

Therefore, $A_{1} \cdot \ldots \cdot A_{n - m} \cdot L^{m - p - q}$ does not have HL property for $p + q = 2(n - \dim T)$ .

This finishes the proof.

![](images/453787226139702a6b7a6b7a4e579dcdfbb179c6eec7ec5ba45c4eea6bedc6a8.jpg)

In summary, we obtain the following characterization:

Theorem 2.16. Let $X$ be a smooth projective variety of dimension $n$ and $L$ a free line bundle on $X$ , then the following statements are equivalent:

- $L$ is $m$ -lef;   
- for any ample line bundles $A_1, \ldots, A_{n-m}$ , the class $A_1 \cdot \ldots \cdot A_{n-m} \cdot L^{m-p-q}$ has HL property for any $0 \leq p, q \leq p + q \leq m$ ;   
- for some ample line bundles $A_1, \ldots, A_{n-m}$ , the class $A_1 \cdot \ldots \cdot A_{n-m} \cdot L^{m-p-q}$ has HL property for any $0 \leq p, q \leq p + q \leq m$ .

We expect that Theorem 2.16 also holds in the analytic setting (see Remark 1.5).

# 3. HIRONAKA'S PRINCIPLE OF COUNTING CONSTANTS

In [Hir68, Section 2], using the “counting constants” method, Hironaka proved results of the following form. We refer the reader to [SS85, Chapter 3] for a modern account.

Theorem 3.1 (Theorem 3.39 of [SS85]). Let X be a smooth projective variety and let L be an ample line bundle on X. Let $f: X \dashrightarrow P^{N}$ be a meromorphic map and let $X_{0}$ be the Zariski open set of X such that $f_{0} = f_{|X_{0}}$ is holomorphic. Then there is a positive integer k such that the generic element of $H^{0}(X, kL)$ does not vanish on any positive dimensional component of $f_{0}^{-1}(y)$ for all $y \in f_{0}(X_{0})$ .

In our setting, we need its extension to free bundles under certain assumption on the interaction between the map and the bundle.

Lemma 3.2. Let X be a smooth projective variety of dimension n, and let L, F be free line bundles on X with corresponding Kodaira maps $\Phi_{L}, \Phi_{F}$ . Then there exists $m_{0} \in N$ such that for any $m \geq m_{0}$ , a generic section of mF does not vanish on any positive dimensional irreducible component W of any fiber $\Phi_{L}^{-1}(y)$ with $\dim \Phi_{F}(W) > 0$ .

Proof. Let W be a positive dimensional irreducible component W of $\Phi_{L}^{-1}(y)$ with $\dim\Phi_{F}(W)>0$ , and denote $q:=\dim\Phi_{F}(W)\geq1$ . We may assume that $\Phi_{L}:X\to Y_{L}$ is the Iitaka fibration associated to L. The proof will be divided into three steps.

Step 1. We have the exact sequence

$$
0 \to H ^ {0} (X, \mathcal {I} _ {W / X} \otimes m F) \to H ^ {0} (X, m F) \to H ^ {0} (X | W, m F),
$$

where $\mathcal{I}_{W / X}$ is the ideal sheaf of $W$ in $X$ and

$$
H ^ {0} (X | W, m F) = \operatorname{Image} [ H ^ {0} (X, m F) \rightarrow H ^ {0} (W, m F | _ {W}) ]
$$

is the space of restricted sections of $mF$ on $W$ .

Following the proof of Pacienza-Takayama [PT11, Theorem 1.1], one can show that there exists an integer $m_0$ and a constant $c > 0$ such that

$$
h ^ {0} (X | W, m F) \geq c m ^ {q}
$$

holds for any $m \geq m_{0}$ .

For the convenience of readers, we include the details here. Let A be the very ample line bundle on $Y_{F}$ such that $\Phi_{F}^{*}A = F$ . Then we have

$$
H ^ {0} (Y _ {F}, m A) \cong H ^ {0} (X, m \Phi_ {F} ^ {*} A) = H ^ {0} (X, m F)
$$

for any $m > 0$ . Since $A$ is ample on $Y_{F}$ , then there exists an integer $m_0$ such that for any $m \geq m_0$ and any $i > 0$ , we have

$$
H ^ {i} (Y _ {F}, \mathcal {I} _ {\Phi_ {F} (W) / Y _ {F}} \otimes m A) = 0
$$

where $\mathcal{I}_{\Phi_F(W) / Y_F}$ is the ideal sheaf of $\Phi_F(W)$ in $Y_{F}$ . Thus, the restriction map

$$
H ^ {0} (Y _ {F}, m A) \rightarrow H ^ {0} (\Phi_ {F} (W), m A)
$$

is surjective for any $m \geq m_0$ . Then we have an inclusion

$$
(\Phi_ {F} | _ {W}) ^ {*} H ^ {0} (\Phi_ {F} (W), m A) \subset H ^ {0} (X | W, m F)
$$

for any $m \geq m_{0}$ . Hence there exists a constant c > 0 such that

$$
h ^ {0} (X | W, m F) \geq c m ^ {q}
$$

for any $m \geq m_{0}$ .

The following arguments are inspired by [Hir68, Section 2], [SS85, Chapter 3] and [Sta23, Tag 055A].

Step 2. Moreover, $c$ and $m_0$ can be chosen independently of $W$ and $y \in Y_L$ . This statement will be proved by induction on $\dim Y_L$ . If $\dim Y_L = 0$ holds, then the statement is trivial. Now we assume $\dim Y_L > 0$ .

Claim. We can find a morphism $v: Y \to Y_L$ with the following properties:

(i) $v$ is a finite open morphism,   
(ii) $Y$ is an integral affine scheme,   
(iii) $X \times_{Y_{L}} Y = \cup_{i=1}^{N} X_{i}$ is a decomposition of $X \times_{Y_{L}} Y$ , where the fibers of the morphism $X_{i} \to Y$ are all geometrically integral for any $i \in \{1, 2, \cdots, N\}$ .

Proof of the Claim. Consider the morphism $\Phi_L: X \to Y_L$ , and it follows from [Sta23, Tag 0551] that we have the following diagram

$$
\begin{array}{c} X ^ {\prime} \xrightarrow {g ^ {\prime}} X _ {V} \longrightarrow X \\ \Phi_ {L} ^ {\prime} \Big \downarrow \qquad \qquad \qquad \Big \downarrow \qquad \qquad \Big \downarrow \Phi_ {L} \\ Y _ {L} ^ {\prime} \xrightarrow {g} V \longrightarrow Y _ {L} \end{array}
$$

where

(i) $V$ is a nonempty open of $Y_{L}$ ,   
(ii) $X_{V} = V \times_{Y_{L}} X$ and $X' = Y_{L}' \times_{V} X_{V}$ ,   
(iii) both $g$ and $g'$ are surjective finite étale,   
(iv) $Y_{L}^{\prime}$ is an irreducible affine scheme,   
(v) all irreducible components of the generic fiber of $\Phi_L'$ are geometrically irreducible.

Denote by $\eta$ the generic point of $Y_L'$ . Then all irreducible components of the generic fiber $X_\eta'$ are geometrically irreducible over $\kappa(\eta)$ . Suppose that $X_\eta' = \cup_{i=1}^N X_{i,\eta}'$ is the decomposition of the generic fiber into (geometrically) irreducible components. For each $1 \leq i \leq N$ , let $X_i'$ be the closure of $X_{i,\eta}'$ in $X'$ and endow $X_i'$ with the induced reduced scheme structure. Note that the generic fiber of $X_i'$ is $X_{i,\eta}'$ . After shrinking $Y_L'$ we may assume that $X' = \cup_{i=1}^N X_i'$ by [Sta23, Tag 054Y]. After shrinking $Y_L'$ some more, it follows from [Sta23, Tag 0554] and [Sta23, Tag 0559] that $X_{i,y}'$ is geometrically irreducible for each $i$ and all $y \in Y_L'$ , and $X_y' = \cup_{i=1}^N X_{i,y}'$ is the decomposition of the fiber $X_y'$ into (geometrically) irreducible components for all $y \in Y_L'$ .

Fix any $i \in \{1,2,\ldots,N\}$ , we consider the morphism $\Phi_L' |_{X_i'} : X_i' \to Y_L$ . By abuse of notation, we will write $\Phi_L'$ instead of $\Phi_L' |_{X_i'}$ . It follows from [Sta23, Tag 0550] that we have the following diagram

$$
\begin{array}{c} X _ {i} ^ {\prime \prime} \xrightarrow {h ^ {\prime}} X _ {i, U} ^ {\prime} \longrightarrow X _ {i} ^ {\prime} \\ \Phi_ {L} ^ {\prime \prime} \Big \downarrow \qquad \qquad \qquad \Big \downarrow \qquad \qquad \Big \downarrow \Phi_ {L} ^ {\prime} \\ Y _ {L} ^ {\prime \prime} \xrightarrow {h} U \longrightarrow Y _ {L} ^ {\prime} \end{array}
$$

where

(i) $U$ is a nonempty open of $Y_L'$ ,   
(ii) $X_{i,U}^{\prime} = U\times_{Y_{L}^{\prime}}X$ and $X_{i}^{\prime \prime} = (Y_{L}^{\prime \prime}\times_{U}X_{i,U}^{\prime})_{red},$   
(iii) both $h$ and $\bar{h}'$ are finite universal homeomorphisms,   
(iv) $Y_{L}^{\prime \prime}$ is an integral affine scheme,   
(v) $\Phi_L^{\prime \prime}$ is flat and of finite presentation,   
(vi) the generic fiber of $\Phi_L^{\prime \prime}$ is geometrically reduced.

After shrinking $Y_{L}^{\prime\prime}$ , we can assume that for every point $y \in Y_{L}^{\prime\prime}$ , the fiber $X_{i,y}^{\prime\prime}$ is geometrically integral by [Gro65, Theorem 12.2.1]. Then the Claim follows immediately.

We will still use the notations introduced in the proof of the Claim. We denote by $\tilde{\Phi}_F$ the composition of the following morphisms

$$
X _ {i} ^ {\prime \prime} \xrightarrow {h ^ {\prime}} X _ {i, U} ^ {\prime} \longrightarrow X _ {i} ^ {\prime} \hookleftarrow X ^ {\prime} \xrightarrow {g ^ {\prime}} X _ {V} \longrightarrow X \xrightarrow {\Phi_ {F}} Y _ {F}.
$$

Since projective morphisms are preserved by base change, we can see that $\Phi_L'': X_i'' \to Y_L''$ is projective. Then it follows immediately that the morphism $\tilde{\Phi}_F \times \Phi_L'': X_i'' \to Y_F \times Y_L''$ is projective. In particular, the morphism $\tilde{\Phi}_F \times \Phi_L''$ is closed. Hence, $(\tilde{\Phi}_F \times \Phi_L'')(X_i'')$ is a closed subset in $Y_F \times Y_L''$ .

For any $i \in \{1, 2, \cdots, N\}$ , set $\Pi_i := (\tilde{\Phi}_F \times \Phi_L')(X_i'')$ with the induced reduced subscheme structure in $Y_F \times Y_L'$ , which is exactly the scheme-theoretic image of the morphism $\tilde{\Phi}_F \times \Phi_L'$ . Let $q_i : \Pi_i \to \Phi_L''(X_i'') = Y_L'$ be the projection for each i. One can easily see that $q_i$ is a projective morphism for each i. It follows from the Claim that there exists a open subvariety $\tilde{Y} \subseteq Y_L$ such that for any closed point $y \in \tilde{Y}$ , $\Phi_F(W)$ can be identified with $q_i^{-1}(y'')$ for some i where $y'' \in Y_L''$ . Then the set of such $\Phi_F(W)$ is contained in a finite number of components of the Hilbert scheme on $\mathbb{P}(H^0(X, F))$ by flattening stratification theorem (see e.g. [Mum66, Lecture 8]) and [Har77, Theorem III.9.9]. It follows from [ACG11, Chapter IX, Lemma 4.1] and the following commutative diagram (all rows and columns are exact)

$$
\begin{array}{c c c} 0 & 0 \\ \downarrow & \downarrow \\ 0 \longrightarrow \mathcal {I} _ {Y _ {F} / \mathbb {P} (H ^ {0} (X, F))} & \longrightarrow \mathcal {I} _ {Y _ {F} / \mathbb {P} (H ^ {0} (X, F))} & \longrightarrow 0 \\ \downarrow & \downarrow & \downarrow \\ 0 \longrightarrow \mathcal {I} _ {\Phi_ {F} (W) / \mathbb {P} (H ^ {0} (X, F))} & \longrightarrow \mathcal {O} _ {\mathbb {P} (H ^ {0} (X, F))} & \longrightarrow \mathcal {O} _ {\Phi_ {F} (W)} \\ \downarrow & \downarrow & \downarrow \\ 0 \longrightarrow \mathcal {I} _ {\Phi_ {F} (W) / Y _ {F}} & \longrightarrow \mathcal {O} _ {Y _ {F}} & \longrightarrow \mathcal {O} _ {\Phi_ {F} (W)} \\ \downarrow & \downarrow & \downarrow \\ 0 & 0 & 0 \end{array}
$$

that the Castelnuovo-Mumford regularity of $\mathcal{I}_{\Phi_F(W) / Y_F}$ with respect to the very ample line bundle $A$ only depends on $h^0 (X,F)$ and the Hilbert polynomial of $\mathcal{I}_{\Phi_F(W) / \mathbb{P}(H^0 (X,F))}$ .

By induction on the dimension of $Y_{L}$ , we can conclude that $c$ and $m_0$ can be chosen independently of $W$ and $y \in Y_{L}$ .

Step 3. Now we can choose a uniform $m_{0}$ such that for any $m \geq m_{0}$ ,

$$
\begin{array}{l} h ^ {0} (X, \mathcal {I} _ {W / X} \otimes m F) = h ^ {0} (X, m F) - h ^ {0} (X | W, m F) \\ \leq h ^ {0} (X, m F) - h ^ {0} (X, L) - 1. \\ \end{array}
$$

Consider the set $\Sigma$ of pairs

$$
(y, s) \in \mathbb {P} (H ^ {0} (X, L)) \times \mathbb {P} (H ^ {0} (X, m F))
$$

such that $s$ vanishes on some positive dimensional component of $\Phi_L^{-1}(y)$ . Then $\Sigma$ is a projective variety and

$$
\dim \Sigma \leq h ^ {0} (X, m F) - 2,.
$$

Therefore, any section $s \in \mathbb{P}(H^0(X, mF)) \backslash p(\Sigma)$ does not vanish identically on any positive dimensional irreducible component $W$ of any fiber of $\Phi_L^{-1}(y)$ with $\dim \Phi_F(W) > 0$ , where

$$
p: Y _ {L} \times \mathbb {P} (H ^ {0} (X, m F)) \to \mathbb {P} (H ^ {0} (X, m F))
$$

is the projection.

![](images/42f78367b95b2833b6cd24c87f1ee681fa3bbb5cb29e4e2e290ea40f73598df9.jpg)

As an application of Lemma 3.2, we obtain the following result on the restriction of $m$ -lef line bundles.

Proposition 3.3. Let $L, F$ be free line bundles. Assume that $F$ is 2-lef, $L$ is $k$ -lef and $L + F$ is $(k + 1)$ -lef, then there exists $m_0 \in \mathbb{N}$ such that for any $m \geq m_0$ and $V \in |mF|$ generic, the restriction $L_{|V}$ is $k$ -lef on the hypersurface $V$ .

Proof. Let $\Phi_L: X \to Y_L \subseteq \mathbb{P}(H^0(X, L))$ , $\Phi_F: X \to Y_F \subseteq \mathbb{P}(H^0(X, F))$ and

$$
\Phi_ {L + F}: X \to Y _ {L + F} \subseteq \mathbb {P} (H ^ {0} (X, L + F))
$$

be the morphisms induced by the linear systems $|L|, |F|$ and $|L + F|$ respectively.

For any $i\geq 0$ , set

$$
Y ^ {i} := \{y \in Y _ {L} | \dim \Phi_ {L} ^ {- 1} (y) = i \}
$$

and

$$
Z ^ {i} := \{(y, z) \in (\Phi_ {L} \times \Phi_ {F}) (X) | \dim \Phi_ {L} ^ {- 1} (y) \cap \Phi_ {F} ^ {- 1} (z) = i \} \subseteq Y _ {L} \times Y _ {F}
$$

Since $L$ is $k$ -lef, for any $i \geq 0$ we have

$$
\dim Y ^ {i} \leq 2 n - 2 i - k.
$$

Since $\Phi_{L + F}$ and $\Phi_F\times \Phi_L$ only differ by a finite morphism and $L + F$ is $(k + 1)$ -lef, then for any $i\geq 0$ we have

$$
\dim Z ^ {i} \leq 2 n - 2 i - k - 1.
$$

For any $i\geq 0$ , set

$Y^{i^{\prime}}:=\{y\in Y^{i}|\exists\text{a top dimensional irreducible component }W\subseteq\Phi_{L}^{-1}(y)\text{ such that dim}\Phi_{F}(W)=0\}$ and

$Y^{i^{\prime \prime}}:=\{y\in Y^{i}|\dim\Phi_{F}(W)>0\text{ holds for all top dimensional irreducible components }W\subseteq\Phi_{L}^{-1}(y)\}$ .

Then for any $i\geq 0$ we have

$$
Y ^ {i} = Y ^ {i ^ {\prime}} \sqcup Y ^ {i ^ {\prime \prime}}.
$$

It follows from Lemma 3.2 that there exists a positive integer $m_0$ such that general element $V \in |mF|$ has proper intersection with any positive dimensional irreducible component $W$ of any fiber $\Phi_L^{-1}(y)$ with $\dim \Phi_F(W) > 0$ . Set $V_L := \Phi_L(V)$ and

$$
V _ {L} ^ {i} := \{y \in V _ {L} | \dim (\Phi_ {L} | _ {V}) ^ {- 1} (y) = i \}.
$$

By the choice of $V$ , we have

$$
Y ^ {i ^ {\prime \prime}} \cap V _ {L} ^ {i} = \emptyset \tag {4}
$$

and

$$
Y ^ {i + 1 ^ {\prime \prime}} \cap V _ {L} \subseteq V _ {L} ^ {i}.
$$

Then we have

$$
\dim Y ^ {i + 1 ^ {\prime \prime}} \cap V _ {L} \leq \dim Y ^ {i + 1} \leq 2 n - 2 - 2 i - k = 2 \dim V - 2 i - k. \tag {5}
$$

Let $p: Y_L \times Y_F \to Y_L$ be the projection. It follows from the definitions of $Y^{i'}$ and $Z^i$ that for any $i \geq 0$ , the restriction of the projection

$$
p \mid_ {Z ^ {i} \cap p ^ {- 1} (Y ^ {i ^ {\prime}})}: Z ^ {i} \cap p ^ {- 1} (Y ^ {i ^ {\prime}}) \to Y ^ {i ^ {\prime}}
$$

is a surjective morphism. Hence, for any $i \geq 0$ we have

$$
\dim Y ^ {i ^ {\prime}} \leq \dim Z ^ {i} \leq 2 n - 2 i - k - 1 = 2 \dim V - 2 i - k + 1. \tag {6}
$$

Then we have

$$
\dim Y ^ {i + 1 ^ {\prime}} \cap V _ {L} \leq \dim Z ^ {i + 1} \leq 2 n - 2 i - k - 3 \leq 2 \dim V - 2 i - k. \tag {7}
$$

The set of divisors containing at least one irreducible component $W$ of $\Phi_L^{-1}(Y^{i'})$ for some $i$ is a finite union of linear proper subspaces of $|mF|$ . Then we can assume that the general element $V$ satisfies

$$
\dim W \cap V <   \dim W
$$

for any irreducible component $W$ of $\Phi_L^{-1}(Y^{i'})$ and any $i \geq 0$ . Meanwhile, we have

$$
\dim \Phi_ {L} (W \cap V) \leq \dim Y ^ {i ^ {\prime}}
$$

for any irreducible component $W$ of $\Phi_L^{-1}(Y^{i'})$ and any $i\geq 0$ .

If $\dim \Phi_L(W \cap V) < \dim Y^{i'}$ holds for any irreducible component $W$ of $\Phi_L^{-1}(Y^{i'})$ , then by

$$
\begin{array}{l} Y ^ {i ^ {\prime}} \cap V _ {L} = \Phi_ {L} | _ {V} ((\Phi_ {L} | _ {V}) ^ {- 1} (Y ^ {i ^ {\prime}} \cap V _ {L})) \\ = \Phi_ {L} | _ {V} (\Phi_ {L} ^ {- 1} (Y ^ {i ^ {\prime}}) \cap V) \\ = \bigcup_ {W \subset \Phi_ {L} ^ {- 1} (Y ^ {i ^ {\prime}})} \Phi_ {L} (W \cap V), \\ \end{array}
$$

it follows from (6) that

$$
\dim Y ^ {i ^ {\prime}} \cap V _ {L} \leq 2 \dim V - 2 i - k. \tag {8}
$$

If $\dim \Phi_L(W \cap V) = \dim Y^{i'}$ holds for some irreducible component $W$ of $\Phi_L^{-1}(Y^{i'})$ , then the dimension of general fiber of the morphism

$$
\Phi_ {L}: W \cap V \to \Phi_ {L} (W) \subseteq Y ^ {i ^ {\prime}} \cap V _ {L}
$$

is at most $i - 1$ . This holds since $\dim \Phi_L(W \cap V) = \dim Y^{i'}$ implies that $\dim \Phi_L(W \cap V) = \dim \Phi_L(W)$ , and the dimension of a general fiber of $\Phi_L: W \to \Phi_L(W)$ is at most $i$ by the definition of $Y^{i'} \subset Y^i$ , and $\dim W \cap V = \dim W - 1$ by the choice of $V$ . Hence, we have

$$
\dim Y ^ {i ^ {\prime}} \cap V _ {L} ^ {i} \leq \dim Y ^ {i ^ {\prime}} - 1 \leq 2 \dim V - 2 i - k. \tag {9}
$$

Combining (4), (5), (7), (8) and (9), we can conclude that

$$
\dim V _ {L} ^ {i} \leq 2 \dim V - 2 i - k
$$

holds for any $i\geq 0$

Since $\Phi_L|_V$ and $\Phi_{L|_V}$ only differ by a finite morphism, by Lemma 2.8 and Lemma 2.3, the proposition follows.

# 4. PROOF OF THE MAIN THEOREM

Recall that we are going to prove:

Theorem 4.1. Let $X$ be a smooth projective variety of dimension $n$ and let $0 \leq p, q \leq p + q \leq n$ be integers. Assume that $L_{1}, \ldots, L_{n - p - q}, M$ are free line bundles such that $M$ is $p + q$ lef and $L_{I}$ is $|I| + p + q$ lef for any $I \subset [n - p - q]$ , then the following statements hold:

(1) the complete intersection class $\Omega = L_{1} \cdot \ldots \cdot L_{n - p - q}$ has hard Lefschetz property, i.e., the linear map

$$
\Omega : H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q, n - p} (X, \mathbb {C})
$$

is an isomorphism.

(2) the pair $(\Omega, M)$ has Hodge-Riemann property, i.e., the quadratic form $Q$ on $H^{p,q}(X, \mathbb{C})$ ,

$$
Q (\alpha , \beta) = c _ {p, q} \Omega \cdot \alpha \cdot \overline {{\beta}}, \alpha , \beta \in H ^ {p, q} (X, \mathbb {C}),
$$

is positive definite on the primitive space

$$
\operatorname{Prim} ^ {p, q} (X) = \ker \{\Omega \cdot M: H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q + 1, n - p + 1} (X, \mathbb {C}) \}.
$$

Proof. For $p = q = 0$ , we need to show the following statement:

$$
L _ {1} \cdot \ldots \cdot L _ {n} > 0.
$$

Note that by Proposition 2.6, L being r-lef implies that $\mathrm{nd}(L) \geq r$ . Then the above statement is a consequence of the positivity criterion for the intersection of nef classes [HX22]:

$$
L _ {1} \cdot \dots \cdot L _ {n} > 0 \text {   if   and   only   if   } \mathrm{nd} (L _ {I}) \geq | I |, \forall I \subset [ n ].
$$

Hence we may suppose $p + q \geq 1$ . In this case, each $L_{i}$ is 2-lef.

We prove the result by induction on $n$ . Denote the first statement by $\mathrm{HL}_n$ and the second statement by $\mathrm{HR}_n$ .

We first show that $\mathrm{HR}_{n - 1}\Rightarrow \mathrm{HL}_n$

Assume that $\phi \in H^{p,q}(X)$ satisfies

$$
L _ {1} \cdot \ldots \cdot L _ {n - p - q} \cdot \phi = 0. \tag {10}
$$

Under the assumption $HR_{n-1}$ , we need to verify that $\phi = 0$ . To this end, we apply Proposition 3.3. By Proposition 3.3, we can take a smooth hypersurface $V \in |mL_{n-p-q}|$ such that the restrictions $L_{1|V}, ..., L_{n-1-p-q|V}, L_{n-p-q|V}$ satisfy the positivity condition for $HR_{n-1}$ on V, that is,

- for any $I \subset [n - 1 - p - q]$ , $L_{I|V}$ is $|I| + p + q$ lef on $V$ ;   
- $L_{n - p - q|V}$ is $p + q$ lef on $V$ .

By restricting (10) to $V$ , we get

$$
L _ {1 | V} \cdot \ldots \cdot L _ {n - 1 - p - q | V} \cdot L _ {n - p - q | V} \cdot \phi_ {| V} = 0. \tag {11}
$$

Using the positivity condition on V and $HR_{n-1}$ , we get that

$$
c _ {p, q} L _ {1 | V} \cdot \dots \cdot L _ {n - 1 - p - q | V} \cdot \phi_ {| V} \cdot \overline {{\phi}} _ {| V} \geq 0 \tag {12}
$$

with equality holds if and only if $\phi_{|V} = 0$ .

By the definition of V and (10), it is clear that (12) is an equality. Therefore, $\phi_{|V}=0$ . This implies $\phi=0$ since the restriction map

$$
H ^ {p + q} (X, \mathbb {C}) \to H ^ {p + q} (V, \mathbb {C})
$$

is injective by Lemma 2.11.

Next, we show that $\mathrm{HL}_n\Rightarrow \mathrm{HR}_n$

Recall that the primitive space is defined by

$$
\operatorname{Prim} ^ {p, q} (X) = \ker \{\Omega \cdot M: H ^ {p, q} (X, \mathbb {C}) \to H ^ {n - q + 1, n - p + 1} (X, \mathbb {C}) \}.
$$

We claim that $H^{p,q}(X,\mathbb{C})$ admits a $Q$ -orthogonal decomposition as follows:

$$
H ^ {p, q} (X, \mathbb {C}) = \operatorname{Prim} ^ {p, q} (X) \oplus M \cdot H ^ {p - 1, q - 1} (X, \mathbb {C}) \tag {13}
$$

with the convention that $H^{p - 1,q - 1}(X,\mathbb{C}) = \{0\}$ when $p = 0$ or $q = 0$ . To this end, we note that by the positivity assumption and Lemma 2.10,

- for any $I \subset [n - p - q]$ , $L_I$ is $|I| + p + q$ lef, which is automatically $|I| + p + q - 2$ lef.   
- for any $I_1 \subset [n - p - q]$ , $L_I := L_{I_1} + M$ is $\max\{|I_1| + p + q, p + q\}$ lef, which is automatically $|I| + p + q - 2$ lef.   
- for any $I_2 \subset [n - p - q]$ , $L_I := L_{I_2} + M + M$ is $\max\{|I_2| + p + q, p + q\}$ lef, which is also $|I| + p + q - 2$ lef.

Therefore, by $HL_{n}$ , the class $L_{1} \cdot \ldots \cdot L_{n-p-q} \cdot M^{2}$ has Hard Lefschetz property, i.e.,

$$
L _ {1} \cdot \dots \cdot L _ {n - p - q} \cdot M ^ {2}: H ^ {p - 1, q - 1} (X, \mathbb {C}) \to H ^ {n - q + 1, n - p + 1} (X, \mathbb {C})
$$

is an isomorphism. In particular,

$$
L _ {1} \cdot \dots \cdot L _ {n - p - q} \cdot M: M \cdot H ^ {p - 1, q - 1} (X, \mathbb {C}) \to H ^ {n - q + 1, n - p + 1} (X, \mathbb {C})
$$

is an isomorphism. Combining the definition of $\operatorname{Prim}^{p,q}$ , we complete the proof of the claimed $Q$ -decomposition for $H^{p,q}(X,\mathbb{C})$ .

As a consequence of the decomposition, we obtain that

$$
\dim \operatorname{Prim} ^ {p, q} (X) = h ^ {p, q} - h ^ {p - 1, q - 1}. \tag {14}
$$

For $t \geq 0$ and a fixed ample line bundle $A$ , consider $L_{1} + tA, \ldots, L_{n - p - q} + tA, M + tA$ and the corresponding $\mathrm{Prim}_t^{p,q}, Q_t$ . Then by the mixed Hodge-Riemann bilinear relation for Kähler classes [DN06, Cat08], we have

- $\dim \operatorname{Prim}_t^{p,q}(X) = h^{p,q} - h^{p-1,q-1} = \dim \operatorname{Prim}^{p,q}(X)$ for any $t > 0$ ;   
- $Q_{t}$ is positive definite on $\mathrm{Prim}_t^{p,q}$ for any $t > 0$ .

By $\mathrm{HL}_n$ , $Q_t$ is non-degenerate on $H^{p,q}(X)$ for any $t \geq 0$ . Therefore, $Q = \lim_{t \to 0} Q_t$ is positive definite on $\operatorname{Prim}^{p,q}(X)$ .

This finishes proof of the theorem.

![](images/f5af99a4c8a992a9dfc5afb72d82ae288091e7375c32fe1f196d4512fcadbca3.jpg)

# REFERENCES

[ACG11] Enrico Arbarello, Maurizio Cornalba, and Phillip A. Griffiths, Geometry of algebraic curves. Volume II with a contribution by Joseph Daniel Harris, Grundlehren der mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences], vol. 268, Springer-Verlag, Heidelberg, 2011. MR 2807457   
[Adi18] Karim Adiprasito, Combinatorial Lefschetz theorems beyond positivity, arXiv:1812.10454 (2018).   
[Cat08] Eduardo Cattani, Mixed Lefschetz theorems and Hodge-Riemann bilinear relations, Int. Math. Res. Not. IMRN (2008), no. 10, Art. ID rnn025, 20. MR 2429243   
[dCM02] Mark Andrea A. de Cataldo and Luca Migliorini, The hard Lefschetz theorem and the topology of semismall maps, Ann. Sci. École Norm. Sup. (4) 35 (2002), no. 5, 759–772. MR 1951443   
[dCM05] \_\_\_\_, The Hodge theory of algebraic maps, Ann. Sci. École Norm. Sup. (4) 38 (2005), no. 5, 693–750.
MR 2195257   
[DN06] Tien-Cuong Dinh and Viêt-Anh Nguyên, The mixed Hodge-Riemann bilinear relations for compact Kähler manifolds, Geom. Funct. Anal. 16 (2006), no. 4, 838–849.   
[EV89] Hélène Esnault and Eckart Viehweg, Vanishing and non-vanishing theorems, Théorie de Hodge - Luminy, Juin 1987 (Barlet D., Esnault H., Elzein F., Verdier Jean-Louis, and Viehweg E., eds.), Astérisque, no. 179-180, Société mathématique de France, 1989 (en). MR 1042803   
[GM88] Mark Goresky and Robert MacPherson, Stratified Morse theory, Ergebnisse der Mathematik und ihrer Grenzgebiete (3) [Results in Mathematics and Related Areas (3)], vol. 14, Springer-Verlag, Berlin, 1988. MR 932724   
[Gro65] Alexander Grothendieck, Éléments de géométrie algébrique. IV. étude locale des schémas et des morphismes de schémas. II, Inst. Hautes Études Sci. Publ. Math. 24 (1965), 231 pp. MR 0199181   
[Har77] Robin Hartshorne, Algebraic geometry, Graduate Texts in Mathematics, vol. 52, Springer-Verlag, New York-Heidelberg, 1977. MR 0463157   
[Hir68] Heisuke Hironaka, Smoothing of algebraic cycles of small dimensions, Amer. J. Math. 90 (1968), 1-54. MR 224611   
[Huh18] June Huh, Combinatorial applications of the Hodge-Riemann relations, Proceedings of the International Congress of Mathematicians—Rio de Janeiro 2018. Vol. IV. Invited lectures, World Sci. Publ., Hackensack, NJ, 2018, pp. 3093–3111. MR 3966524   
[Huh22] \_\_\_\_, Combinatorics and Hodge theory, Proceedings of the International Congress of Mathematicians, 2022.   
[HX22] Jiajun Hu and Jian Xiao, Hard Lefschetz properties, complete intersections and numerical dimensions, arXiv:2212.13548 (2022).   
[Laz04] Robert Lazarsfeld, Positivity in algebraic geometry. I, Ergebnisse der Mathematik und ihrer Grenzgebiete. 3. Folge. A Series of Modern Surveys in Mathematics [Results in Mathematics and Related Areas. 3rd Series. A Series of Modern Surveys in Mathematics], vol. 48, Springer-Verlag, Berlin, 2004, Classical setting: line bundles and linear series. MR 2095471   
[Mum66] David Mumford, Lectures on curves on an algebraic surface. With a section by G. M. Bergman, Annals of Mathematics Studies, vol. 59, Princeton, N.J., 1966. MR 0209285   
[PT11] Gianluca Pacienza and Shigeharu Takayama, On volumes along subvarieties of line bundles with nonnegative Kodaira-Iitaka dimension, Michigan Math. J. 60 (2011), no. 1, 35–49. MR 2785862   
[RSW23] Julius Ross, Hendrik Süss, and Thomas Wannerer, Dually Lorentzian Polynomials, arXiv:2304.08399 (2023).   
[Som78] Andrew John Sommese, Submanifolds of Abelian varieties, Math. Ann. 233 (1978), no. 3, 229-256. MR 466647   
[SS85] Bernard Shiffman and Andrew John Sommese, Vanishing theorems on complex manifolds, Progress in Mathematics, vol. 56, Birkhäuser Boston, Inc., Boston, MA, 1985. MR 782484   
[Sta23] The Stacks project authors, The Stacks Project, 2023, http://stacks.math.columbia.edu.   
[Tim98] V. A. Timorin, Mixed Hodge-Riemann bilinear relations in a linear context, Funktsional. Anal. i Prilozhen. 32 (1998), no. 4, 63–68, 96. MR 1678857   
[Wil18] Geordie Williamson, The Hodge theory of the Hecke category, European Congress of Mathematics, Eur. Math. Soc., Zürich, 2018, pp. 663–683. MR 3890447   
[Xia21] Jian Xiao, Mixed Hodge-Riemann bilinear relations and m-positivity, Sci. China Math. 64 (2021), no. 7, 1703–1714. MR 4280377

Tsinghua University, Beijing 100084, China

Email: hujj22@mails.tsinghua.edu.cn

Email: sjshang@mail.tsinghua.edu.cn

Email: jianxiao@tsinghua.edu.cn