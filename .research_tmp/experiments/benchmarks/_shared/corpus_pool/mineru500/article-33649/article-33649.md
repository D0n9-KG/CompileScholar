# On Action Theories with Iterable First-Order Progression

Daxin Liu $^{1*}$ , Jens Claßen $^{2}$

$^{1}$ State Key Laboratory for Novel Software Technology, Nanjing University, China

$^{2}$ Institute for People and Technology, Roskilde University, Denmark  
daxin.liu@nju.edu.cn, classen@ruc.dk

# Abstract

We study the first-order definability of progression for situation calculus action theories with a focus on the iterability of progression. Progression, the task of updating a knowledge base according to actions' effects so that proper information is retained, is notoriously challenging as it in general requires second-order logic. Exceptions where progression is first-order like local-effect actions and normal actions impose certain syntax constraints on action theories to eliminate second-order quantifiers in the progressed knowledge base. Unfortunately, the progressed result might not satisfy the constraints again, making it impossible to apply first-order progression iteratively. In this paper, we first lift the existing result on first-order progression for normal actions by allowing disjunctions in the knowledge base. As a result, we obtain an action theory whose type is called disjunctive normal, which is iteratively first-order progressable. Second, we propose a new class of action theories, called PANACK, that strictly subsumes the disjunctive normal ones, and we show that it remains iteratively first-order progressable as well.

# 1 Introduction

Intelligent agents acting in real-world scenarios need to be able to handle incomplete information, where in particular the number of objects they have to interact with is unbounded. Ideally, a representation of an agent's world model is hence given in terms of first-order logic. The Situation Calculus (McCarthy and Hayes 1969; Reiter 2001) is perhaps the most widely studied first-order formalism for reasoning about action and change. A central problem in this context is projection, where the task is to determine whether a given formula comes to hold after a given action sequence. Here, the specifics of the domain are encoded in terms of an action theory, which consists of axioms describing the initial situation as well as the pre- and postconditions of actions. Whereas regression solves projection by transforming the query formula to an equivalent one about the initial situation, progression updates the knowledge base to reflect the changes brought about by the action sequence. The latter is often preferable, in particular for longer sequences where regression can cause a significant blow-up, rendering it practically infeasible. Progression, on the other hand, comes with its own challenge: Lin and Reiter (1997) showed that for a first-order (FO) knowledge base (KB), the progression may in general require second-order (SO) logic.

Since Lin and Reiter's seminal work, efforts have been made to identify restricted classes of action theories where the existence of a FO progression can be guaranteed: In local-effect theories (Vassos, Lakemeyer, and Levesque 2008; Liu and Lakemeyer 2009), actions are required to name all objects they affect explicitly in their arguments. An action $move(x,z,y)$ for moving block $x$ from $y$ onto $z$ is thus local-effect, as only the truth values of $on(x,y)$ and $on(x,z)$ change due it. A classical example for an action that is not local-effect is that of exploding a bomb, which destroys all (unmentioned) objects in its vicinity. The class of normal actions due to Liu and Lakemeyer (2009) supports such global effects, as long as the affected fluent predicates only depend on ones that are only subject to local effects, and under additional restrictions on the initial KB. Liu and Claßen (2024) generalize this class to acyclic actions, where more complex interactions between non-local-effect fluents are allowed, as long as they do not contain cycles.

Unfortunately, as we will show in this paper, normal action theories (and hence also acyclic ones) suffer from the problem that progression is in general not iterable. The reason, roughly, is that a fluent predicate F subject to non-local effects is required to only occur in expressions of the form $\psi(\vec{x}) \supset F(\vec{x})$ or $F(\vec{x}) \supset \phi(\vec{x})$ in the initial theory. However, the outcome of progression then might not satisfy this requirement, so the result itself cannot be progressed anymore! The aforementioned works only consider progression through a single action, but arguably, for progression to be useful, it should work for all actions and all action sequences admitted by the theory. Therefore, in this paper, we study the iterability of FO progression. In particular, after presenting formal preliminaries (Section 2), we lift Liu and Lake-meyer's first-order result on normal actions by allowing disjunctions in the knowledge base, thus obtaining a class of action theories called the disjunctive normal ones that are iteratively first-order progressable (Section 3). Furthermore, we present a new class of action theories, called PANACK, that strictly subsumes the disjunctive normal ones and allows for more complex dependencies between fluents (including cycles), which we also prove to be iteratively FO progressable (Section 4). Finally, we discuss related work and conclude.

# 2 Preliminaries

In this section, we review some important notions such as forgetting, the situation calculus, and some recent results on FO progression with a focus on iterability.

We start with a FO language L with equality. For simplicity, we only consider predicates and ignore functions. The set of formulas of L is the least set that contains the atomic formulas, and if $\phi$ and $\psi$ are in the set and x is a variable, then $\neg\phi$ , $\phi\wedge\psi$ and $\forall x\phi$ are in the set. The connectives $\vee,\supset,\equiv$ , and $\exists$ are understood as the usual abbreviations. We will use parentheses around quantifiers to indicate the scopes, and “dot” to indicate that the quantifier preceding the dot has maximum scope, e.g., $\forall x.\phi(x)\supset\psi(x)$ stands for $\forall x(\phi(x)\supset\psi(x))$ . Leading universal quantifiers might be omitted in writing sentences, i.e., free variables are assumed implicitly $\forall$ -quantified from the outside, i.e., we identify $\phi(x)$ with $\forall x.\phi(x)$ . A theory is a set of sentences. We use $\phi\Leftrightarrow\psi$ to mean $\phi$ and $\psi$ are logically equivalent. Let $\psi$ be a formula, and let $\mu$ and $\mu'$ be two expressions (terms or formulas). We denote by $\phi(\mu/\mu')$ the result of simultaneously replacing every occurrence of $\mu$ in $\phi$ with $\mu'$ .

# 2.1 Forgetting

Intuitively, forgetting a ground atom (or predicate) in a theory leads to a weaker theory that entails the same set of sentences that are “irrelevant” to the atom (or predicate) (Lin and Reiter 1994).

Definition 1 (Forgetting). Let $T$ be a theory, and $\mu$ a ground atom or predicate symbol. A theory $T'$ is a result of forgetting $\mu$ in $T$ , denoted by $forget(T, \mu) \Leftrightarrow T'$ , if for any structure $M$ , $M \models T'$ iff there exists a model $M'$ of $T$ s.t. $M' \sim_{\mu} M$ , where $M' \sim_{\mu} M$ means that $M, M'$ agree on everything except maybe the interpretation of $\mu$ .

Trivially, if $T, T'$ are both the result of forgetting $\mu$ in $T$ , then $T \Leftrightarrow T'$ . Definition 1 naturally extends to forgetting a set of ground atoms or predicates. In this paper, we only consider finite theories, so henceforth, we only consider forgetting for sentences.

Now, we consider $L^{2}$ , the second-order extension of L. For a sentence $\phi$ and ground atom $P(\vec{t})$ , let $\phi[P(\vec{t})]$ be the formula obtained by replacing every occurrence of the form $P(\vec{t}')$ in $\phi$ with $[\vec{t} = \vec{t}' \wedge P(\vec{t})] \vee [\vec{t} \neq \vec{t}' \wedge P(\vec{t}')]$ , and let $\phi_{+}^{P(\vec{t})}$ and $\phi_{-}^{P(\vec{t})}$ be formulas obtained by replacing $P(\vec{t})$ in $\phi[P(\vec{t})]$ with TRUE and FALSE, respectively.

Theorem 2 (Lin and Reiter 1994). Let $P(\vec{t})$ be a ground atom, $P$ a predicate symbol, and $\phi$ a sentence. Then

- forget $(\phi, P(\vec{t})) \Leftrightarrow \phi_{+}^{P(\vec{t})} \vee \phi_{-}^{P(\vec{t})}$ ;   
- forget $(\phi, P) \Leftrightarrow \exists R. \phi(P / R)$ ,

where $R$ is a SO variable.

Likewise, we can define $\text{forget}(\phi, \Gamma)$ for a finite set of ground atoms $\Gamma$ by iteratively forgetting atoms in $\Gamma$ .

Let $\phi_1 := \text{broken}(A) \land \text{contains}(B, A)$ and $\phi_2 := \exists x$ . $\text{broken}(x) \land \exists y.\text{contains}(y, x)$ , then $\text{forget}(\phi_1, \text{broken}(A)) \Leftrightarrow \text{contains}(B, A)$ , and $\text{forget}(\phi_2, \text{broken}) \Leftrightarrow \exists R \exists x. R(x) \land \exists y.\text{contains}(y, x) \Leftrightarrow \exists x \exists y.\text{contains}(y, x)$ .

# 2.2 Basic Action Theories

The situation calculus (Reiter 2001) $L_{sc}$ is a many-sorted FO language (with some second-order features) for representing dynamic worlds. There are three sorts: action, situation, and object. $L_{sc}$ contains the following features: a distinct constant $S_{0}$ denoting the initial situation; a binary function $do(a,s)$ representing the new situation resulting from doing action a in situation s; a binary relation $Poss(a,s)$ expressing action a being executable in situation s; action functions, e.g. $drop(x,y)$ ; a finite set of fluent predicates, i.e., predicates whose last argument is a situation term, e.g., broken(x,s).

A formula $\phi$ is uniform in a situation term s if $\phi$ does not mention any other situation terms except s, does not quantify over situation variables, and does not mention Poss.

The dynamics of a domain is specified by a basic action theory (BAT) in $\mathcal{L}_{sc}$ as

$$
\mathcal {D} = \Sigma_ {i n d} \cup \mathcal {D} _ {a p} \cup \mathcal {D} _ {s s} \cup \mathcal {D} _ {u n a} \cup \mathcal {D} _ {S _ {0}}, \text { where }
$$

1. $\Sigma_{ind}$ is a set of domain-independent axioms that ensure situations are well-structured;   
2. $\mathcal{D}_{ap}$ is a set of action precondition axioms;   
3. $\mathcal{D}_{ss}$ is a set of successor state axioms (SSAs), one for each fluent predicate $F$ , of the form

$$
F (\vec {x}, d o (a, s)) \equiv \gamma_ {F} ^ {+} (\vec {x}, a, s) \vee \neg \gamma_ {F} ^ {-} (\vec {x}, a, s) \wedge F (\vec {x}, s),
$$

where $\gamma_{F}^{+}$ and $\gamma_{F}^{-}$ are uniform in $s$ ;

4. $\mathcal{D}_{una}$ is the set of unique names axioms for actions: $A(\vec{x}) \neq A'(\vec{y})$ , and $A(\vec{x}) = A(\vec{y}) \supset \vec{x} = \vec{y}$ ;   
5. $\mathcal{D}_{S_0}$ , the initial database (or initial KB), is a finite set of sentences uniform in $S_0$ .

Successor state axioms constitute Reiter's (1991) solution to the frame problem. In particular, it is required that for all fluent predicates $F$ , $\mathcal{D} \models \neg (\gamma_F^+ \land \gamma_F^-)$ . Henceforth, given a ground action $\alpha$ , we use $S_{\alpha}$ to refer to the situation $do(\alpha, S_0)$ .

# 2.3 Progression

For formalizing progression, we follow the definition by (Vassos and Levesque 2013), which is equivalent to the original model-theoretical one by (Lin and Reiter 1997).

Definition 3 (Progression). Let D be a BAT, $\alpha$ a ground action, and $D_{S_{\alpha}}$ a set of (first-order or second-order) sentences uniform in $S_{\alpha}$ . We say that $D_{S_{\alpha}}$ is a progression of $D_{S_{0}}$ w.r.t. $\alpha$ , D iff for every sentence $\phi$ uniform in $S_{\alpha}$ ,

$$
\mathcal {D} \models \phi \text {   iff   } (\mathcal {D} - \mathcal {D} _ {S _ {0}}) \cup \mathcal {D} _ {S _ {\alpha}} \models \phi .
$$

Namely, a progression retains all the logical entailments in terms of the future of the initial KB. By applying progression iteratively, one obtains a progression for sequences of ground actions naturally.

Lin and Reiter (1997) proved that progression is always second-order definable as follows. We write the instantiation of $D_{ss}$ w.r.t. $\alpha$ and $S_{0}$ as $D_{ss}[\alpha, S_{0}]$ , i.e. $D_{ss}[\alpha, S_{0}]$ is the set of sentences $F(\vec{x}, do(\alpha, S_{0})) \equiv \Phi_{F}(\vec{x}, \alpha, S_{0})$ , where $\Phi_{F}$ denotes the right-hand side (RHS) of the SSA for F. Let

$F_{1}, \ldots, F_{n}$ be the set of all fluents. For each $F_{i}$ , we introduce a new predicate symbol $P_{i}$ . We use $\phi \uparrow S_{0}$ to denote the result of replacing every $F_{i}(\vec{t}, S_{0})$ in $\phi$ by $P_{i}(\vec{t})$ and call $P_{i}$ the lifting predicate for $F_{i}$ . For a finite set of formulas $\Sigma$ , we also use $\Sigma$ to denote the conjunctions of its elements. Using this notation, the following is a progression of $\mathcal{D}_{S_{0}}$ w.r.t. $\alpha$ :

$$
\exists \vec {R}. \{(\mathcal {D} _ {u n a} \cup \mathcal {D} _ {S _ {0}} \cup \mathcal {D} _ {s s} [ \alpha , S _ {0} ]) \uparrow S _ {0} \} (\vec {P} / \vec {R}) \tag {1}
$$

where $\vec{R}=\{R_{1},\ldots,R_{n}\}$ are second-order predicate variables. By Theorem 2, the progression of an initial theory $D_{S_{0}}$ w.r.t. $\alpha$ and $D_{ss}$ can be obtained by adding the effects of $\alpha$ (the union of $D_{ss}[\alpha,S_{0}]$ ) and forgetting the past (by means of the SO existential quantifiers in the head of (1)).

# 2.4 Iterative First-Order Progression

Efforts have been made to identify fragments of the situation calculus where progression is FO definable, i.e. conditions under which Eq. (1) is equivalent to a FO theory. For instance, Lin and Reiter (1997) showed that this is the case if the initial KB is relatively complete, i.e., for every sentence $\phi$ uniform in $S_{0}$ , KB entails either $\phi$ or its negation, or if the basic action theory is context-free, i.e. actions' effects are independent of situations. Notably, these two types of action theories are all iteratively first-order progressable.

Definition 4. A BAT D is called iteratively first-order progressable if for all action sequences $\vec{\alpha}$ , the progression of $D_{S_{0}}$ w.r.t. $\vec{\alpha}$ and $D_{ss}$ is first-order definable.

Liu and Lakemeyer (2009) (LL09 for short) proved that if the basic action theory is local-effect, then progression is always FO definable. Intuitively, a ground action has local effects if it only affects the truth of ground fluent atoms that mention only the action's parameters.

Definition 5. An SSA is local-effect if both $\gamma_F^+(\vec{x}, a, s)$ and $\gamma_F^-(\vec{x}, a, s)$ are disjunctions of formulas of the form $\exists \vec{z}[a = A(\vec{u}) \wedge \phi(\vec{u}, s)]$ , where $A$ is an action function, $\vec{u}$ contains $\vec{x}$ , $\vec{z}$ is the remaining variables of $\vec{u}$ , and $\phi$ is called the context. An action theory is local-effect if each SSA is local-effect.

LL09 observed that for a local-effect action theory, every ground action $\alpha$ affects only finitely many fluent atoms, the so-called characteristic set $\Omega$ , determined by the action's parameters. Hence, forgetting the lifting predicates can be reduced to forgetting these finitely many instances, resulting in a FO theory. Namely,

$$
\operatorname{forget} \left(\mathcal {D} _ {u n a} \cup \mathcal {D} _ {S _ {0}} \cup \mathcal {D} _ {s s} [ \Omega ], \Omega\right) \left(S _ {0} / S _ {\alpha}\right) \tag {2}
$$

is a progression of $D_{S_{0}}$ w.r.t. $\alpha$ , $D_{ss}$ where $D_{ss}[\Omega]$ is the instantiation of $D_{ss}$ w.r.t. $\Omega$ . Moreover, this process is iterable, yielding an iteratively FO progressable action theory.

LL09 also extended their FO progression result on local-effect actions to normal actions, where actions might have local effects on some fluents while having non-local effects on others, with additional restrictions on the initial KB. More recently, Liu and Claßen (2024) extended these results to the so-called acyclic actions, where the affected non-local fluents might be mutually dependent, as long as their dependencies do not form cycles. Unfortunately, while these results are significant, the results are about the FO progression through a single action, not action sequences or all actions admitted by the theory. As a result, there are instances of action theories where an action is normal or acyclic, yet the same action is no longer normal or acrylic w.r.t. the progression result, and so it is impossible to apply the same progression method again, let alone this being the case for multiple actions at the agent's disposal. We discuss this in Section 5.

# 3 Disjunctive Normal Action Theories

The lack of guarantee for normal actions to admit iterated progression motivates us to identify action theories that are iteratively FO progressable, here called disjunctive normal action theories. To obtain the result, we first introduce some necessary notation. We defer a comparison between our result and the result on normal actions (also acyclic actions) to Section 5.

Definition 6 (Semi-definitional). A finite theory $T$ is semi-definitional (SDEF) w.r.t. a predicate $P$ if the only occurrences of $P$ in $T$ are of the form $P(\vec{x}) \supset \phi (\vec{x})$ or $\psi (\vec{x}) \supset P(\vec{x})$ , where $\phi$ and $\psi$ do not mention $P$ . $\phi$ is called a necessary condition of $P$ , and $\psi$ a sufficient condition.

An example for a (non-trivial) formula that is not semi-definitional is $\forall x, y, z. P(x, y) \vee \neg P(y, z)$ . Note that “⊃” is an abbreviation in terms of “∨”, meaning formulas $\neg P(x) \vee \phi(x)$ and $P(x) \vee \psi(x)$ are also SDEF w.r.t. P. We use $WSC_{P}$ (weakest sufficient condition) to denote the disjunction of formulas $\psi(\vec{x})$ such that $\psi(\vec{x}) \supset P(\vec{x})$ is in T, and $SNC_{P}$ (strongest necessary condition) to denote the conjunction of formulas $\phi(\vec{x})$ with $P(\vec{x}) \supset \phi(\vec{x})$ in T.

Theorem 7 (Liu and Lakemeyer 2009). Let $T$ be finite and semi-definitional w.r.t. $P$ , and $T'$ the set of sentences in $T$ not mentioning $P$ . Then forget $(T, P) \Leftrightarrow T' \land \forall \vec{x}. \mathrm{WSC}_P(\vec{x}) \supset \mathrm{SNC}_P(\vec{x})$ .

The theorem is a direct application of the well-known Ackermann (1935) lemma for second-order quantifier elimination. We first note that alternatively, we can express the result of forgetting P as a set of implications obtained by determining all “resolvents” over P:

Proposition 8. Let $T$ be finite and semi-definitional w.r.t. $P$ , and $T'$ the set of sentences in $T$ not mentioning $P$ . Then

$$
forget(T,P)\Leftrightarrow T^{\prime}\wedge \bigwedge_{\substack{\psi \in \operatorname{WSC}_{P}\\ \phi \in \operatorname{SNC}_{P}}}\forall \vec{x} .  \psi (\vec{x})\supset \phi (\vec{x}).
$$

Next, we show that this theorem can be extended to allow disjunctions in the theory:

Lemma 9. Given theories $T_1, \ldots, T_k$ and predicate $P$ , forget( $\bigvee_i T_i, P$ ) $\Leftrightarrow \bigvee_i$ forget( $T_i, P$ ).

Proof. By Theorem 2, $\text{forget}(\bigvee_{i} T_i, P) \Leftrightarrow \exists R. (\bigvee_{i} T_i)_R^P$ which is equivalent to $\bigvee_{i} \exists R. (T_i)_R^P$ . By Theorem 2 again, the result is equivalent to $\bigvee_{i} \text{forget}(T_i, P)$ .

Definition 10 (Disjunctive semi-definitional). A theory T is said to be disjunctive semi-definitional (DSDEF) w.r.t. predicate P if each sentence in T is of them form $\bigvee_{i}\psi_{i}$ where all $\psi_{i}$ are SDEF w.r.t. P.

Proposition 11. Let $T, T'$ be theories that are DSDEF w.r.t. predicate $P$ , and $Q(\vec{t})$ a ground atom where $Q$ is distinct from $P$ . Then we have:

1. forget(T, P) is FO definable;   
2. forget $(T\wedge T',P)$ , forget $(T\vee T',P)$ are FO definable;   
3. forget $(T, Q(\vec{t}))$ can be rewritten to be DSDEF w.r.t. $P$ ;   
4. if $T$ is also DSDEF w.r.t. predicate $P'$ , then forget $(T, P)$ can be equivalently rewritten to be DSDEF w.r.t. $P'$ .

The proofs for items 1 and 2 are similar. Simply distributing conjunctions over disjunctions in the theories and then applying Lemma 9 and Theorem 7, we obtain a FO result. For Item 3, it is easy to see, $T_{+}^{Q(\vec{t})}$ and $T_{-}^{Q(\vec{t})}$ are both DSDEF w.r.t. $P$ . By Theorem 2, $forget(T, Q(\vec{t})) \Leftrightarrow T_{+}^{Q(\vec{t})} \vee T_{-}^{Q(\vec{t})}$ . Distributing the disjunction over conjunctions in $T_{+}^{Q(\vec{t})}$ and $T_{-}^{Q(\vec{t})}$ , we obtain a theory of the desired form. Item 4 is less obvious. First, we distribute disjunctions over conjunctions in $T$ and obtain a sentence of the form $\bigvee_{i} T_{i}$ , where each $T_{i}$ is the conjunction of sentences that are SDEF with both $P$ and $P'$ , namely, $T_{i} = \bigcup_{j} \phi_{i,j}$ so that the $\phi_{i,j}$ are of the form $\psi(x) \vee (\neg)P(\vec{x}) \vee (\neg)P'(\vec{x})$ , where $(\neg)P$ means either $P$ or $\neg P$ , and $\psi$ contains no $P$ and $P'$ . Now forgetting $P$ in $T_{i}$ via Proposition 8, the result is equivalent to the conjunctions of all the resolvents for sentences in $T_{i}$ w.r.t. $P(\vec{x})$ . Clearly, the resolvents are again SDEF w.r.t. $P'$ . Therefore, $forget(T_{i}, P)$ can be equivalently rewritten to be SDEF w.r.t. $P'$ . Let $Re(T_{i})$ denote the rewritten result. Using $forget(T, P) \Leftrightarrow \bigvee_{i} forget(T_{i}, P) \Leftrightarrow \bigvee_{i} Re(T_{i})$ and distributing conjunctions over disjunction in $\bigvee_{i} Re(T_{i})$ , we obtain the desired theory.

Proposition 12. In any model of D, the sentence $F(\vec{x}, S_{\alpha}) \equiv \gamma_{F}^{+}(\vec{x}, \alpha, S_{0}) \vee \neg \gamma_{F}^{-}(\vec{x}, \alpha, S_{0}) \wedge F(\vec{x}, S_{0})$ is equivalent to the conjunction of following sentences: $^{1}$

$$
\gamma_ {F} ^ {+} \vee \neg F (\vec {x}, S _ {\alpha}) \vee F (\vec {x}, S _ {0}) \tag {3a}
$$

$$
\neg F (\vec {x}, S _ {0}) \vee \gamma_ {F} ^ {-} \vee F (\vec {x}, S _ {\alpha}) \tag {3b}
$$

$$
\neg \gamma_ {F} ^ {+} \vee F (\vec {x}, S _ {\alpha}) \tag {3c}
$$

$$
\neg \gamma_ {F} ^ {-} \vee \neg F (\vec {x}, S _ {\alpha}). \tag {3d}
$$

Given a BAT D, we call a fluent local-effect if its SSA is local-effect. We denote the set of all local-effect fluents in D as $\mathrm{LE}(\mathcal{D})$ and the other fluents as $\mathrm{NLE}(\mathcal{D})$ .

Definition 13 (Disjunctive Normal Action Theory). A BAT $\mathcal{D}$ is said to be a disjunctive normal action theory, if

1. $\mathcal{D}_{S_0}$ is DSDEF w.r.t. all fluents in NLE(D);   
2. for each fluent $F \in \mathrm{NLE}(\mathcal{D})$ , in its SSA, all fluents appearing in $\gamma_F^+$ and $\gamma_F^-$ are in $\mathrm{LE}(\mathcal{D})$ .

Theorem 14. Disjunctive normal action theories are iteratively first-order progressable.

We only need to show that for any ground action $\alpha$ , the progression of $\mathcal{D}_{S_0}$ w.r.t. $\alpha, \mathcal{D}_{ss}$ is FO and DSDEF w.r.t. $F(\vec{x}, S_{\alpha})$ for fluents $F \in \mathrm{NLE}(\mathcal{D})$ .

Proof. Let $D_{ss}^{F}[\alpha, S_{0}]$ denote the instantiation of the SSA for fluent F w.r.t. $\alpha$ and $S_{0}$ . To compute the progression, by Eq. (1), we only need to forget the lifting predicates of fluents in $(\mathcal{D}_{una} \cup \mathcal{D}_{S_{0}} \bigcup_{F} \mathcal{D}_{ss}^{F}[\alpha, S_{0}]) \uparrow S_{0}$ . This can be achieved by first iterating over the lifting predicates for fluents in $\mathrm{NLE}(\mathcal{D})$ (in any order) and afterwards the ones in $\mathrm{LE}(\mathcal{D})$ (in any order). We show the results are in the right form in every intermediate step.

For fluents $F \in \mathrm{NLE}(\mathcal{D})$ , we replace $\mathcal{D}_{ss}^{F}[\alpha, S_{0}]$ with formulas (3a)-(3d). Clearly, after lifting, (3a) and (3b) are the only places the lifting predicate $P$ of $F$ occurs (recall that $P(\vec{x})$ is substituted for $F(\vec{x}, S_{0})$ ) and they are SDEF w.r.t. $P$ . In addition, (3a)-(3d) are SDEF w.r.t. $F(\vec{x}, S_{\alpha})$ . Now, we need to forget $P$ in $\mathcal{D}_{una} \cup \mathcal{D}_{S_0} \cup \{(3a), \ldots, (3d)\} \uparrow S_0$ . By items 2 and 4 in Prop. 11, the result is FO definable and can be rewritten to be SDEF w.r.t. the remaining lifting predicates and $F(\vec{x}, S_{\alpha})$ . Iterating this process for all remaining fluents in NLE(D), we obtain a theory $T$ that is FO and DSDEF w.r.t. $F(\vec{x}, S_{\alpha})$ for all fluents $F \in \mathrm{NLE}(\mathcal{D})$ .

Now, for fluents $F \in \mathrm{LE}(\mathcal{D})$ , we forget their lifting predicate in $T$ by Eq. (2). By item 3 in Prop. 11, the result is FO and DSDEF w.r.t. $F(\vec{x}, S_{\alpha})$ for all fluents $F \in \mathrm{NLE}(\mathcal{D})$ .

Example 15. Consider the domain that is described by the two fluents broken(x, s) and shielded(x, s) that say, respectively, that object x is broken and shielded in situation s. The action explode will destroy everything that is unshielded and the action cover(x) will make x shielded. The following SSAs express such a domain:

$$
\begin{array}{c} \text {broken} (x, d o (a, s)) \equiv a = \text {explode} \land \neg \text {shielded} (x, s) \lor \\ \text {broken} (x, s) \end{array}
$$

$$
\text { shielded } (x, d o (a, s)) \equiv a = \text { cover } (x) \vee \text { shielded } (x, s)
$$

Let $\mathcal{D}$ be a BAT where SSAs are as above and $\mathcal{D}_{S_0}$ is $\{shielded(x, S_0) \supset \neg broken(x, S_0)\}$ , then $\mathcal{D}$ is a disjunctive normal action theory with $\mathrm{LE}(\mathcal{D}) = \{shielded\}$ and $\mathrm{NLE}(\mathcal{D}) = \{broken\}$ . For the ground action $\alpha = cover(A)$ , $\alpha$ has no effects on broken and local effects on shielded. To progress $\mathcal{D}_{S_0}$ w.r.t. $\alpha$ , we only need to forget $\Omega = \{shielded(A, S_0)\}$ in $\mathcal{D}_{S_0} \cup \{shielded(A, S_\alpha)\}$ according to Eq. (2), resulting in $\mathcal{D}_{S_\alpha}$ :

$$
\{s h i e l d e d (A, S _ {\alpha}), \tag {4a}
$$

$$
\forall x. [ x = A \lor s h i e l d e d (x, S _ {\alpha}) ] \supset \neg b r o k e n (x, S _ {\alpha}) \vee (4 b)
$$

$$
\forall x. [ x \neq A \land s h i e l d e d (x, S _ {\alpha}) ] \supset \neg b r o k e n (x, S _ {\alpha}) \}. \tag {4c}
$$

Namely, A is shielded in $S_{\alpha}$ , and depending on if A is broken in $S_{0}$ , there are two cases: everything that is shielded, including A (4b) or excluding A (4c), is not broken in $S_{\alpha}$ . The result is DSDEF w.r.t. broken( $x, S_{\alpha}$ ).

Now, consider the action $\beta = explode$ . It has non-local effects on broken and no effects on shielded. We progress $D_{S_{\alpha}}$ w.r.t. $\beta$ . Let $S_{\beta} = do(\beta, S_{\alpha})$ . By Prop. 12, $D_{ss}^{broken}[\beta, S_{\alpha}]$ is equivalent to the conjunction of:

$$
\neg \text { shielded } (x, S _ {\alpha}) \vee \neg \text { broken } (x, S _ {\beta}) \vee \text { broken } (x, S _ {\alpha}) \tag {5a}
$$

$$
\neg \text { broken } (x, S _ {\alpha}) \vee \text { broken } (x, S _ {\beta}), \tag {5b}
$$

$$
\text { shielded } (x, S _ {\alpha}) \vee \text { broken } (x, S _ {\beta}). \tag {5c}
$$

Let $P$ be the lifting predicate of broken $(x, S_{\alpha})$ . Clearly,

$$
\text { forget } (\mathcal {D} _ {S _ {\alpha}} \cup \mathcal {D} _ {s s} ^ {b r o k e n} \uparrow S _ {\alpha}, P) \Leftrightarrow
$$

$$
\operatorname{forget} \left(\{(4 a), (4 b) \} \cup \{(5 a), (5 b), (5 c) \} \uparrow S _ {\alpha}, P\right) \vee \tag {6a}
$$

$$
\text { forget } (\{(4 \mathrm{a}), (4 \mathrm{c}) \} \cup \{(5 \mathrm{a}), (5 \mathrm{b}), (5 \mathrm{c}) \} \uparrow S _ {\alpha}, P) \tag {6b}
$$

Applying Theorem 7 to (6a) and (6b) and substituting $S_{\alpha}$ with $S_{\beta}$ , we have $\mathcal{D}_{S_{\beta}}$ as (after simplification)

$$
\{s h i e l d e d (A, S _ {\beta}),
$$

$$
\operatorname{shielded} (x, S _ {\beta}) \vee \operatorname{broken} (x, S _ {\beta}),
$$

$$
\forall x. \neg \text { shielded } (x, S _ {\beta}) \vee \neg \text { broken } (x, S _ {\beta}) \vee \tag {7a}
$$

$$
\forall x. x = A \vee \neg \text {   shielded } (x, S _ {\beta}) \vee \neg \text {   broken } (x, S _ {\beta}) \}. \tag {7b}
$$

Namely, A is shielded, all unshielded objects are broken, and if A was not broken in $S_{0}$ , all shielded objects are now not broken (7a), otherwise all shielded objects except for A are now not broken (7b). Obviously, $D_{S_{\beta}}$ is DSDEF w.r.t. broken( $x, S_{\beta}$ ) as well.

# 4 PANACK Action Theories

Although the above disjunctive normal action theories can capture global effects and are iteratively FO progressable, they have limitations as well. The most obvious one is that actions' effects on non-local-effect fluents cannot rely on other non-local-effect fluents. Here, we present a class of action theories called Pan-Ackermann (PANACK for short) $^{2}$ that subsumes the class of disjunctive normal action theories and overcomes this drawback. To begin, we enlarge the class of theories that remain FO after forgetting predicates.

Definition 16 (Pan-semi-definitional). A sentence is pan-semi-definitional (PANSDEF) w.r.t. a predicate P if

1. $P$ appears as ground atoms in it; or   
2. it occurs in the form $P(\vec{x}) \supset \phi(\vec{x})$ or $\psi(\vec{x}) \supset P(\vec{x})$ , and $\phi, \psi$ either contain no $P$ or only as ground atoms.

E.g. the following are all PANSDEF w.r.t. $P: P(A) \vee Q(x), \forall x. Q(x) \supset P(x), \forall x. P(A) \wedge Q(x) \supset P(x)$ .

Proposition 17. If a sentence is PANSDEF w.r.t. a predicate P, then, it can be equivalently rewritten to a theory that is DSDEF w.r.t. P.

For an assignment $\theta$ over ground atoms $P(\vec{t}_{1}),\ldots,P(\vec{t}_{k})$ , we also use $\theta$ to refer to the set of literals it satisfies and $\phi[\theta]$ to refer to the formula obtained from $\phi$ by replacing every atom $P(\vec{t}_{i})$ in $\phi$ with its respective truth values TRUE or FALSE in $\theta$ . We prove the proposition by cases.

Proof. 1. In case $P$ appears only as ground atoms, wlog assume $P(\vec{t}_1), \ldots, P(\vec{t}_k)$ are all the ground atoms in $\phi$ , i.e. $\phi := \phi[P(\vec{t}_1), \ldots, P(\vec{t}_k)]$ . Let $\Theta$ be the set of all possible truth assignments over $P(\vec{t}_1), \ldots, P(\vec{t}_k)$ . It is easy to see that $\phi \Leftrightarrow \bigvee_{\theta \in \Theta} \theta \wedge \phi[\theta]$ . Clearly, $\phi[\theta]$ contains no $P$ . On the other hand, $\theta$ only contains ground literals of $P$ . Since $P(\vec{t}_i) \Leftrightarrow \forall \vec{x}. \vec{x} = \vec{t}_i \supset P(\vec{x})$ (likewise for $\neg P(\vec{t}_i)$ ), $\theta$ can be rewritten to be a theory that is semi-definitional

w.r.t. P. Let $RE[\theta]$ be the rewriting result. Distributing the disjunction of $\Theta$ over conjunctions in $RE[\theta]$ , one obtains the desired DSDEF theory.

2. This case is very similar in spirit to the above. When $\phi(x)$ contains no P, the sentence is semi-definitional w.r.t. P by definition. If $\phi(x)$ contains P as ground atoms, we extract these atoms outside and rewrite them. After properly distributing disjunctions over conjunctions, one obtains the desired result. □

E.g. the formula $\forall x.P(A) \land Q(x) \supset P(x)$ is PANSDEF w.r.t. $P$ . It is equivalent to $P(A) \land (\forall x.Q(x) \supset P(x)) \lor \neg P(A) \land \text{TRUE}$ , which can be rewritten as $(\forall x.Q(x) \supset P(x)) \lor (\forall x.x = A \supset \neg P(x))$ , a theory that is disjunctive semi-definitional w.r.t. $P$ .

Definition 18 (Disjunctive pan-semi-definitional). A theory T is said to be disjunctive pan-semi-definitional (DPANSDEF) w.r.t. a predicate P if each sentence in T is of the form $\bigvee_{i}\psi_{i}$ , where the $\psi_{i}$ are pan-semi-definitional (PANSDEF) w.r.t. P.

Proposition 19. Every DPANSDEF theory can be equivalently rewritten to be a DSDEF theory and vice versa.

$(\Leftarrow)$ is trivial. For $(\Rightarrow)$ , one can use the techniques in the proof of Prop. 17 to rewrite $\psi_{i}$ and distribute disjunctions over conjunctions, which results in a DSDEF theory.

In the remaining paper, we assume $\gamma_F^+$ and $\gamma_F^-$ are disjunctions of the form $\exists \vec{y} \cdot (a = A(\vec{v}) \wedge \phi_A \wedge \phi'_A)$ where $A(\vec{v})$ is an action term and $\vec{v}$ contains $\vec{y}$ ; all free variables of $\phi_A$ are among $\vec{v}$ ; and $\phi'_A$ might contain free variables not in $\vec{v}$ (such as variables in $\vec{x}$ ). $\phi_A$ is called the context condition and $\phi'_A$ is called the effect descriptor (Zarrieß and Claßen 2016) in the sense that $\phi_A$ specifies if action $A(\vec{v})$ will have an effect on instances of $F$ , and $\phi'_A$ specifies which instances of $F$ are affected by the action.

Definition 20 (PANACK Action Theories). A BAT D is said to be a Pan-Ackermann (PANACK) action theory, if

1. $\mathcal{D}_{S_0}$ is DPANSDEF w.r.t. all fluents in NLE(D);   
2. for each fluent $F \in \mathrm{NLE}(\mathcal{D})$ , all fluents in $\mathrm{NLE}(\mathcal{D})$ mentioned in $\gamma_F^+$ or $\gamma_F^-$ only appear in $\phi_A$ , but not in $\phi_A'$ , for all action symbols $A$ .

Clearly, the definition above generalizes disjunctive normal action theories. For the latter, $D_{S_{0}}$ has to be DSDEF, while here $D_{S_{0}}$ is required to be DPANSDEF. Also, disjunctive normal action theories cannot contain non-local-effect fluents in the RHS of an SSA for a non-local-effect fluent, yet, here this is allowed as long as non-local-effect fluents only occur in context conditions.

Theorem 21. PANACK action theories are iteratively first-order progressable.

We sketch the idea of the proof and only focus on non-local-effect fluents. Local-effect fluents can be handled the same as in Theorem 14. The key ideas of the proof are that: (1) we rewrite $\mathcal{D}_{S_0}$ into one that is DSDEF w.r.t. all fluents in NLE(D) as in Prop. 19 and call the result $RE(\mathcal{D}_{S_0})$ ; (2) we equivalently replace $\mathcal{D}_{ss}^F[\alpha, S_0]$ by the formulas given in Prop. 12, which, after instantiating the theory by the ground action $\alpha = A(\vec{t})$ , are equivalent to

$$
\begin{array}{l} \neg (\phi_ {A} ^ {+} \land \phi_ {A} ^ {\prime +}) \lor \neg F (\vec {x}, S _ {\alpha}) \lor F (\vec {x}, S _ {0}) (8a) \\ \neg F (\vec {x}, S _ {0}) \vee \phi_ {A} ^ {-} \wedge \phi_ {A} ^ {\prime -} \vee F (\vec {x}, S _ {\alpha}) (8b) \\ \neg \left(\phi_ {A} ^ {+} \wedge \phi_ {A} ^ {\prime +}\right) \vee F (\vec {x}, S _ {\alpha}) (8c) \\ \neg \left(\phi_ {A} ^ {-} \wedge \phi_ {A} ^ {\prime -}\right) \vee \neg F (\vec {x}, S _ {\alpha}) (8d) \\ \end{array}
$$

where $\phi_A^+$ and $\phi_A^{\prime +}$ (likewise for $\phi_A^-$ and $\phi_A^{\prime -}$ ) are the corresponding positive context conditions and effect descriptors of $A(\vec{t})$ . The key observation is that non-local-effect fluents can only appear in $\phi_A^+$ but not $\phi_A^{\prime +}$ . Since the only free variables in $\phi_A^+$ (likewise for $\phi_A^-$ ) are from $\vec{v}$ , once grounded by $\vec{t}$ , non-local-effect fluents in $\phi_A^+$ occur as ground atoms in formulas (8a)-(8d). By Prop. 17, formulas (8a)-(8d) can all be rewritten to be a theory that is DSDEF w.r.t. the non-local-fluent in $\phi_A^+$ or $\phi_A^-$ . More importantly, the rewritten theory (denoted by $RE(\mathcal{D}_{ss}^{F}[\alpha ,S_{0}])$ ) is also semi-definitional w.r.t. $F(\vec{x},S_{\alpha})$ and $F(\vec{x},S_0)$ . Now, we only need to forget the lifting predicates in $RE(\mathcal{D}_{S_0})\cup$ $\bigcup_F RE(\mathcal{D}_{ss}^F [\alpha ,S_0])\uparrow S_0$ . By Prop. 11 Item 4, the result is FO and can be rewritten to be DSDEF (hence also DPANSDEF) w.r.t. $F(\vec{x},S_{\alpha})$ for $F\in \mathrm{NLE}(\mathcal{D})$ . Hence, the progression is FO and iterable.

Example 22. Consider a box domain with three fluents, adapted from (Claßen and Zarrieß 2017): contains(x, y, s) says that x contains y, on(x, y, s) says that x is on y, and broken(x, s) says that x is broken in situation s. The action drop(x, y) denotes dropping container x from shelf y, causing all things in x to become broken and no longer be positioned on y. The SSAs $D_{ss}$ are given by

$$
\gamma_ {b r o k e n} ^ {+} := \exists y, z. a = d r o p (y, z) \wedge \underline {{o n (y , z , s)}}
$$

$$
\wedge \underline {{\text { contains }}} (y, x, s).
$$

$$
\gamma_ {o n} ^ {-} := \exists z. a = d r o p (z, y) \wedge (\underline {{z = x \vee c o n t a i n s (z , x , s)}})
$$

$$
\gamma_ {b r o k e n} ^ {-} \equiv \gamma_ {o n} ^ {+} \equiv \gamma_ {c o n t a i n s} ^ {+} \equiv \gamma_ {c o n t a i n s} ^ {-} \equiv \mathrm{FALSE}
$$

where drop has no effect on contains. Context conditions and effect descriptors are underlined by solid and dashed lines, respectively. Consider a BAT D with $D_{S_{0}}$ as

$$
\{c o n t a i n s (B o x, V a s e, S _ {0}), \tag {9a}
$$

$$
o n (B o x, S h e l f, S _ {0}), \tag {9b}
$$

$$
\text { broken } (x, S _ {0}) \supset \neg \exists y. \text { contains } (y, x, S _ {0}) \}. \tag {9c}
$$

Namely, Box contains Vase, Box is on Shelf, and every broken object is not contained in anything.

$\mathcal{D}$ is a PANACK action theory: it is easy to see that $\mathrm{LE}(\mathcal{D}) = \{\text{contains}\}$ and $\mathrm{NLE}(\mathcal{D}) = \{\text{on, broken}\}$ . For $\mathcal{D}_{S_0}$ , (9b) and (9c) are PANSDEF w.r.t. on and broken, respectively, hence $\mathcal{D}_{S_0}$ is disjunctive pan-semi-definitional w.r.t. $\mathrm{NLE}(\mathcal{D})$ . Furthermore, only the SSA for broken mentions some fluent from $\mathrm{NLE}(\mathcal{D})$ , namely on, however notice that on only appears in the context condition. Hence, $\mathcal{D}$ is indeed a PANACK action theory.

Now, to progress D w.r.t. $\alpha = drop(Box, Shelf)$ , first, we rewrite $D_{S_{0}}$ to be disjunctive semi-definitional w.r.t.

NLE(D), which yields (call this $RE(\mathcal{D}_{S_{0}})$ )

$$
\{c o n t a i n s (B o x, V a s e, S _ {0}), \tag {10a}
$$

$$
x = B o x \wedge y = S h e l f \supset o n (x, y, S _ {0}), \tag {10b}
$$

$$
\text { broken } (x, S _ {0}) \supset \neg \exists y. \text { contains } (y, x, S _ {0}) \}. \tag {10c}
$$

$\mathcal{D}_{ss}[\alpha, S_0]$ , by Prop. 12, is equivalent to

$$
\begin{array}{l} \left\{\left[ o n (B o x, S h e l f, S _ {0}) \wedge c o n t a i n s (B o x, x, S _ {0}) \right] \right. \\ \lor \neg \text { broken } (x, S _ {\alpha}) \lor \text { broken } (x, S _ {0}), \tag {11a} \\ \end{array}
$$

$$
\neg \text { broken } (x, S _ {0}) \vee \text { broken } (x, S _ {\alpha}), \tag {11b}
$$

$$
\neg o n (B o x, S h e l f, S _ {0}) \vee \neg c o n t a i n s (B o x, x, S _ {0})
$$

$$
\vee \text {   broken } (x, S _ {\alpha}) \tag {11c}
$$

$$
\neg o n (x, y, S _ {\alpha}) \vee o n (x, y, S _ {0}), \tag {11d}
$$

$$
\neg o n (x, y, S _ {0}) \vee y = \text { Shelf } \wedge [ B o x = x
$$

$$
\vee \text { contains } (B o x, x, S _ {0}) ] \vee \text { on } (x, y, S _ {\alpha}), \tag {11e}
$$

$$
y = \text { Shelf } \land (x = B o x \lor
$$

$$
\left. \text { contains } (B o x, x, S _ {0})) \supset \neg o n (x, y, S _ {\alpha}) \right\} \tag {11f}
$$

Clearly both sets of formulas above are PANSDEF w.r.t. NLE(D) ∪ {on(x, y, Sα), broken(x, Sα)} (let us denote the set by NLE\*(D)). In fact, except (11a) and (11c), the formulas are SDEF w.r.t. NLE\*(D). For (11a), we replace it according to Prop. 17 (after distributing conjunctions over disjunctions) by the disjunction of the two sets:

$$
\{x = B o x \land y = S h e l f \supset P (x, y), \tag {12a}
$$

$$
\text { contains } (B o x, x, S _ {0}) \vee \neg \text { broken } (x, S _ {\alpha}) \vee P ^ {\prime} (x) \}, \tag {12b}
$$

$$
\{x = B o x \land y = S h e l f \supset \neg P (x, y), \tag {12c}
$$

$$
\neg \text { broken } (x, S _ {\alpha}) \vee P ^ {\prime} (x) \}. \tag {12d}
$$

Likewise, (11c) is replaced by the disjunction of

$$
x = \text { Box } \land y = \text { Shelf } \supset \neg P (x, y) \tag {13a}
$$

$$
\neg \text { contains } (B o x, x, S _ {\alpha}) \vee \neg \text { broken } (x, S _ {\alpha}) \tag {13b}
$$

Now, each formula in (12a)–(12d) and (13a)–(13b) is semi-definitional w.r.t. NLE $^{\star}$ (D). Distributing disjunctions over conjunctions in the processed $\mathcal{D}_{ss}[\alpha, S_0]$ , we obtain a rewrite $RE(\mathcal{D}_{ss}^F[\alpha, S_0])$ that is DSDEF w.r.t. NLE $^{\star}$ (D). Now forgetting the lifting predicates in $(RE(\mathcal{D}_{S_0}) \cup RE(\mathcal{D}_{ss}^F[\alpha, S_0])) \uparrow S_0$ (this can be done in the same way as in Theorem 14 since the rewritten sets are both DSDEF w.r.t. NLE $^{\star}$ (D)), one obtains, with simplifications:

$$
\{c o n t a i n s (B o x, V a s e, S _ {\alpha}), \tag {14a}
$$

$$
y = \text { Shelf } \land (x = B o x \lor \text { contains } (B o x, x, S _ {\alpha}))
$$

$$
\supset \neg o n (x, y, S _ {\alpha}), \tag {14b}
$$

$$
\neg \text { contains } (B o x, x, S _ {\alpha}) \vee \text { broken } (x, S _ {\alpha}), \tag {14c}
$$

$$
\neg \text { broken } (x, S _ {\alpha}) \vee \text { contains } (\text { Box }, x, S _ {\alpha}) \vee
$$

$$
\neg \exists y. \text { contains } (y, x, S _ {\alpha}) \} \tag {14d}
$$

That is Box is still in Vase (14a), Box and everything contained in it are no longer on Shelf (14b), all things contained in Box are broken (14c), and all broken objects are either contained in Box, or were among the previously broken objects not contained in anything (14d). It is easy to check that the progressed KB is FO and again DSDEF (in fact, semi-definitional) w.r.t. $on(x,y,S_{\alpha})$ and $broken(x,S_{\alpha})$ . It would now be possible to iteratively progress through another action, say $drop(Box2,Shelf)$ .

# 5 Discussion

Here we compare our results with the FO progression results on normal actions and acyclic actions. As mentioned before, the progression for normal and acyclic actions is only defined in terms of a single ground action, rather than an action theory that admits an unbounded number of actions. More formally, according to LL09, a ground action $\alpha$ is said to have local effects on a fluent $F(\vec{x}, s)$ , if by using $D_{una}$ , $\gamma_{F}^{+}(\vec{x}, \alpha, s)$ and $\gamma_{F}^{-}(\vec{x}, \alpha, s)$ can be simplified to a disjunction of formulas of the form $\vec{x} = \vec{t} \wedge \psi(s)$ , where $\vec{t}$ is a vector of ground terms, and $\psi$ is a formula whose only free variable is s. Let $\mathrm{LE}(\alpha)$ be the set of all fluents $\alpha$ has local effects on, and $\mathrm{NLE}(\alpha)$ be the other fluents. Then:

Definition 23. A ground action $\alpha$ is normal if for each fluent $F$ , all the fluents that appear in $\gamma_F^+$ and $\gamma_F^-$ are in $\mathrm{LE}(\alpha)$ .

Theorem 24. For a BAT D where $D_{S_{0}}$ is SDEF w.r.t. NLE( $\alpha$ ), progression of $D_{S_{0}}$ w.r.t. $\alpha$ is FO definable and computable.

Our definition of disjunctive normal theories is stricter in the sense that is based on $\mathrm{LE}(\mathcal{D})$ instead of $\mathrm{LE}(\alpha)$ : A fluent being in $\mathrm{LE}(\mathcal{D})$ means it has to be in $\mathrm{LE}(\alpha)$ for every action $\alpha$ . However, disjunctive normal theories are also less restrictive in the sense that $\mathcal{D}_{S_0}$ is only required to be disjunctive SDEF w.r.t. $\mathrm{NLE}(\mathcal{D})$ , whereas normal theories require it to be SDEF (w.r.t. $\mathrm{NLE}(\alpha)$ ) without allowing for disjunctions.

Acyclic actions (Liu and Claßen 2024) generalize normal actions by allowing fluents in NLE( $\alpha$ ) to depend on each other, but the dependency graph has to be acyclic, and fluents appearing in $\gamma_{F}^{+}$ or $\gamma_{F}^{-}$ have to be in a specific form to ensure that one can apply Theorem 7 to forget the fluents' lifting predicates in an order that follows the structure of the graph. Again, our PANACK action theories are more restrictive in one sense, but less restrictive in another. On the one hand, acyclic theories allow NLE fluents to appear in effect descriptors, albeit in a limited form. On the other hand, PANACK theories allow cyclic dependencies among NLE fluents, as long as they only appear in context conditions.

We want to emphasize that our main contribution is on the iterability of FO progression, arguably an important desideratum for planning and reasoning about action and change. Note that the BAT in Example 15 is normal w.r.t. both $\alpha = cover(A)$ and $\beta = explode$ , yet, after progressing w.r.t. $\alpha$ , the theory shown in Eqs. (4a)-(4c) is no longer normal w.r.t. $\beta$ . This shows that FO progression for normal actions and acyclic actions can in general not be iterated.

Lastly, it is worth mentioning that there are BATs where all actions are normal (hence acyclic) and progression is iteratively FO but the BAT is neither PANACK nor disjunctive normal. Consider a variant of the BAT from Example 15:

$$
\begin{array}{l} \text { broken } (x, d o (a, s)) \equiv a = \text { explode } \land \neg \text { shielded } (x, s) \lor \\ b r o k e n (x, s) \\ \text {   shielded } (x, d o (a, s)) \equiv \neg (a = \text {   unshieldbroken   } \\ \wedge \text {   broken } (x)) \wedge \text {   shielded } (x, s) \\ \end{array}
$$

Namely, explode is as before, but unshieldbroken will remove shields for all broken objects. Let $D_{S_{0}} = \{\}$ . Clearly, both explode and unshieldbroken are normal: for explode, broken $\in$ NLE(explode) and shielded $\in$ LE(explode), while for unshieldbroken it is the reverse. Both fluents are in NLE(D) and they mutually appear on the RHS of the SSAs, so the BAT is not disjunctive normal. One can check that the BAT is indeed iteratively FO progressable.

# 6 Related Work

Lin and Reiter (1997) provided a general account of progression. They also showed that context-free and relatively complete action theories are iteratively FO progressable. Two ways of extensions exist: some works extend the result on relatively complete action theories by increasing the expressiveness of the initial KB. E.g., (Vassos and Patrizi 2013) proposed relatively complete action theories with bounded unknowns which allow less complete information in the KB, and (De Giacomo et al. 2016) extend this further to bounded situation calculus action theories. Both classes are shown to admit iterable FO progression. Meanwhile, some efforts increase the expressiveness of SSAs (compared to context-free ones). E.g. local-effect theories (Vassos, Lakemeyer, and Levesque 2008; Liu and Lakemeyer 2009) allow fluents to appear on the RHS of an SSA, albeit in a limited fashion. Moreover, normal and acyclic actions extend this further by allowing complex dependencies among fluents, but, as demonstrated, FO progression can only be guaranteed for single actions and might not be iterable. Arenas et al. (2018) show that progression is iteratively FO progressable for so-called universal basic action theory with constants. This class is incomparable to the above (and ours) and even admits infinite theories (determining whether a finite progression exists is in general undecidable).

Readers interested in an overview on FO progression – without focus on iterability – are referred to (Vassos and Patrizi 2013) and (Liu and Claßen 2024). Progression has many applications, e.g., (Lakemeyer and Levesque 2009; Liu and Feng 2023) considered the interplay between progression and the notion of only-knowing after actions. (Belle and Levesque 2014; Liu and Belle 2024) studied probabilistic progression in the situation calculus. Other works that involve progression include (Fang, Liu, and Van Ditmarsch 2019) for multi-agent modal logic, (Schwering, Lakemeyer, and Pagnucco 2015; Claßen and Delgrande 2022) for belief revision, and (Claßen 2013; Liu et al. 2023; Liu 2023) for planning and verification in GOLOG. We believe our work suggests new possibilities for these applications.

# 7 Conclusion

We studied the FO definability of progression in the situation calculus with a focus on iterability. We generalized the result by Liu and Lakemeyer, obtaining disjunctive normal action theories as a class that is iteratively FO progressable. We also proposed a new type of action theory, called PANACK, that strictly subsumes the disjunctive normal ones, where fluents can be mutually dependent in a complex manner, and again showed it to be iteratively FO progressable. For future work, besides identifying other (larger) classes that are FO progressable, we plan to study the applicability of our results in the context of planning, verification, and synthesis.

# Acknowledgments

Daxin was funded by a Royal Society University Research Fellowship.

# References

Ackermann, W. 1935. Untersuchungen über das Eliminationsproblem der mathematischen Logik. Mathematische Annalen, 110(1): 390–413.

Arenas, M.; Baier, J. A.; Navarro, J. S.; and Sardina, S. 2018. On the progression of situation calculus universal theories with constants. In KR.

Belle, V.; and Levesque, H. 2014. How to progress beliefs in continuous domains. In KR.

Claßen, J. 2013. Planning and verification in the agent language Golog. Ph.D. thesis, RWTH Aachen University.

Claßen, J.; and Delgrande, J. P. 2022. Projection of Belief in the Presence of Nondeterministic Actions and Fallible Sensing. In KR, 400–404.

Claßen, J.; and Zarrieß, B. 2017. Decidable Verification of Decision-Theoretic Golog. In FroCoS, volume 10483 of LNCS, 227–243. Springer.

De Giacomo, G.; Lespérance, Y.; Patrizi, F.; and Vassos, S. 2016. Progression and verification of situation calculus agents with bounded beliefs. Studia Logica, 104: 705–739.

Fang, L.; Liu, Y.; and Van Ditmarsch, H. 2019. Forgetting in multi-agent modal logics. Artificial Intelligence, 266: 51–80.

Lakemeyer, G.; and Levesque, H. J. 2009. A semantical account of progression in the presence of defaults. Conceptual Modeling: Foundations and Applications: Essays in Honor of John Mylopoulos, 82–98.

Lin, F.; and Reiter, R. 1994. Forget it. In Working Notes of AAAI Fall Symposium on Relevance, 154–159.

Lin, F.; and Reiter, R. 1997. How to progress a database. Artificial Intelligence, 92(1-2): 131–167.

Liu, D. 2023. Projection in a probabilistic epistemic logic and its application to belief-based program verification. Ph.D. thesis, RWTH Aachen University, Germany.

Liu, D.; and Belle, V. 2024. Progression with Probabilities in the Situation Calculus: Representation and Succinctness. In AAMAS, 1210–1218.

Liu, D.; and Claßen, J. 2024. First-Order Progression beyond Local-Effect and Normal Actions. In IJCAI. IJCAI Organization.

Liu, D.; and Feng, Q. 2023. On the progression of belief. Artificial Intelligence, 322: 103947.

Liu, D.; Huang, Q.; Belle, V.; and Lakemeyer, G. 2023. Verifying Belief-Based Programs via Symbolic Dynamic Programming. In ECAI, 1497–1504. IOS Press.

Liu, Y.; and Lakemeyer, G. 2009. On first-order definability and computability of progression for local-effect actions and beyond. In IJCAI, 860–866.

McCarthy, J.; and Hayes, P. 1969. Some philosophical problems from the standpoint of artificial intelligence. In Meltzer, B.; and Michie, D., eds., Machine Intelligence 4, 463–502. New York: American Elsevier.

Reiter, R. 1991. The Frame Problem in the Situation Calculus: A simple Solution (sometimes) and a Completeness Result for Goal Regression. Artificial Intelligence and Mathematical Theory of Computation: Papers in Honor of John McCarthy, 359–380.

Reiter, R. 2001. Knowledge in action: logical foundations for specifying and implementing dynamical systems. MIT press.

Schwering, C.; Lakemeyer, G.; and Pagnucco, M. 2015. Belief revision and progression of knowledge bases in the epistemic situation calculus. In IJCAI.

Vassos, S.; Lakemeyer, G.; and Levesque, H. J. 2008. First-Order Strong Progression for Local-Effect Basic Action Theories. In KR, 662–672.

Vassos, S.; and Levesque, H. J. 2013. How to progress a database III. Artificial Intelligence, 195: 203–221.

Vassos, S.; and Patrizi, F. 2013. A Classification of First-Order Progressable Action Theories in Situation Calculus. In IJCAI, 1132–1138.

Zarrieß, B.; and Claßen, J. 2016. Decidable verification of Golog programs over non-local effect actions. In AAAI.