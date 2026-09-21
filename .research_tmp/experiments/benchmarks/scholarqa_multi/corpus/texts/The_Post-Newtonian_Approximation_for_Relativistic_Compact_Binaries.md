# The Post-Newtonian Approximation for Relativistic Compact Binaries

Toshifumi Futamase

Astronomical Institute

Tohoku University

Sendai 980-8578

Japan

email: tof@astr.tohoku.ac.jp

http://www.astr.tohoku.ac.jp/\~tof/

Yousuke Itoh

Astronomical Institute

Tohoku University

Sendai 980-8578

Japan

email: yousuke@astr.tohoku.ac.jp

Living Reviews in Relativity

ISSN 1433-8351

Accepted on 19 January 2007

Published on 12 March 2007

# Abstract

We discuss various aspects of the post-Newtonian approximation in general relativity. After presenting the foundation based on the Newtonian limit, we show a method to derive post-Newtonian equations of motion for relativistic compact binaries based on a surface integral approach and the strong field point particle limit. As an application we derive third post-Newtonian equations of motion for relativistic compact binaries which respect the Lorentz invariance in the post-Newtonian perturbative sense, admit a conserved energy, and are free from any ambiguity.

# Imprint / Terms of Use

Living Reviews in Relativity are published by the Max Planck Institute for Gravitational Physics (Albert Einstein Institute), Am Mühlenberg 1, 14476 Potsdam, Germany. ISSN 1433-8351

This review is licensed under a Creative Commons Attribution-Non-Commercial-NoDerivs 2.0 Germany License: http://creativecommons.org/licenses/by-nc-nd/2.0/de/

Because a Living Reviews article can evolve over time, we recommend to cite the article as follows:

Toshifumi Futamase and Yousuke Itoh, "The Post-Newtonian Approximation for Relativistic Compact Binaries", Living Rev. Relativity, 10, (2007), 2. [Online Article]: cited [<date>], http://www.livingreviews.org/lrr-2007-2

The date given as <date> then uniquely identifies the version of the article you are referring to.

# Article Revisions

Living Reviews supports two different ways to keep its articles up-to-date:

Fast-track revision A fast-track revision provides the author with the opportunity to add short notices of current research results, trends and developments, or important publications to the article. A fast-track revision is refereed by the responsible subject editor. If an article has undergone a fast-track revision, a summary of changes will be listed here.

Major update A major update will include substantial changes and additions and is subject to full external refereeing. It is published with a new publication number.

For detailed documentation of an article's evolution, please refer always to the history document of the article's online version at http://www.livingreviews.org/lrr-2007-2.

# Contents

# 1 Introduction 5

1.1 Gravitational wave detection and post-Newtonian approximation ..... 5   
1.2 Post-Newtonian equations of motion 7   
1.3 Plan of this paper 9

# 2 Foundation of the Post-Newtonian Approximation 10

2.1 Newtonian limit along a regular asymptotic Newtonian sequence 10   
2.2 Post-Newtonian hierarchy 12   
2.3 Explicit calculation in harmonic coordinates 13

# 3 Post-Newtonian Equations of Motion for Compact Binaries 16

3.1 Strong field point particle limit 16   
3.2 Surface integral approach and body zone 17   
3.3 Scalings on the initial hypersurface 18   
3.4 Newtonian equations of motion for extended bodies 20

# 4 Formulation 22

4.1 Field equations 22   
4.2 Near zone contribution 24

4.2.1 Body zone contribution 24   
4.2.2 N/B contribution 27

4.3 Lorentz contraction and multipole moments 27

4.4 General form of the equations of motion 28   
4.5 On the arbitrary constant $R_A$ 29

4.5.1 $R_{A}$ dependence of the field 29   
4.5.2 $R_{A}$ dependence of the equations of motion 29

4.6 Newtonian equations of motion 30   
4.7 First post-Newtonian equations of motion 31   
4.8 Body zone boundary dependent terms 34

# 5 Third Post-Newtonian Gravitational Field 36

5.1 Super-potential method 36   
5.2 Super-potential-in-series method 36   
5.3 Direct-integration method 37

# 6 Third Post-Newtonian Mass-Energy Relation 39

6.1 Meaning of $P_{A\Theta}^{\tau}$ 40

# 7 Third Post-Newtonian Momentum-Velocity Relation 42

# 8 Third Post-Newtonian Equations of Motion 44

8.1 Third post-Newtonian equations of motion with logarithmic terms 44   
8.2 Arbitrary constant $\epsilon R_A$ 45   
8.3 Consistency relation 46   
8.4 Third post-Newtonian equations of motion 46   
8.5 Comparison 50   
8.6 Summary 52   
8.7 Going further 53

# 9 Acknowledgements 55

# A Far Zone Contribution 56

# B Effects of Extendedness of Stars 60

B.1 Spin-orbit coupling force 61   
B.2 Spin-spin coupling force 62   
B.3 Quadrupole-orbit coupling force 62   
B.4 Spin geodesic precession 63   
B.5 Remarks 64

# C A Generalized Equivalence Principle Including The Emission of Gravitational Wave 65

C.1 Introduction 65   
C.2 Equations of motion 66

# References 69

# 1 Introduction

# 1.1 Gravitational wave detection and post-Newtonian approximation

The motion and associated emission of gravitational waves (GW) of self-gravitating systems have been a main research interest in general relativity. The problem is complicated conceptually as well as mathematically because of the nonlinearity of the Einstein equations. There is no hope in any foreseeable future to have exact solutions describing motions of arbitrarily shaped, massive bodies, so we have to adopt some sort of approximation schemes for solving the Einstein equations to study such problems. In the past years many types of approximation schemes have been developed depending on the nature of the system under consideration. Here we shall focus on a particular scheme called the post-Newtonian (PN) approximation. There are many systems in astrophysics where Newtonian gravity is dominant, but general relativistic gravity plays also an important role in their evolution. For such systems it would be nice to have an approximation scheme which gives a Newtonian description in the lowest order and general relativistic effects as higher order perturbations. The post-Newtonian approximation is perfectly suited for this purpose. Historically Einstein computed first the post-Newtonian effects, e.g. the precession of the perihelion [72]. Studies of the post-Newtonian approximation were made by Lorentz and Droste [117], Einstein, Infeld, and Hoffmann [73], Fock [77], Plebansky and Bazanski [131], and Chandrasekhar and associates [40, 41, 42, 43].

Now it is widely known that the post-Newtonian approximation is important in analyzing a number of relativistic problems, such as the equations of motion of binary pulsars $[34, 61, 74, 89]$ , solar-system tests of general relativity $[159, 160, 161, 164]$ , and gravitational radiation reaction $[39, 42]$ . Any approximation scheme necessitates one or several small parameters characterizing the nature of the system under consideration. A typical parameter which most of the schemes adopt is the magnitude of the metric deviation from a certain background metric. In particular if the background is Minkowski spacetime and there is no other parameter, the scheme is sometimes called the post-Minkowskian approximation in the sense that the constructed spacetime reduces to Minkowski spacetime in the limit that the parameter tends to zero. This limit is called the weak field limit. In the case of the post-Newtonian approximation the background spacetime is also Minkowski spacetime, but there is another small parameter, that is, the typical velocity of the system divided by the speed of light. We introduce a non-dimensional parameter $\epsilon$ to express the “slowness” of the system. These two parameters (the deviation from the flat metric and the velocity) have to have a certain relation in the following sense. As the gravitational field gets weaker, all velocities and forces characteristic of the material systems become smaller, in order to permit the weakening of gravity to remain an important effect in the system’s dynamics. For example in the case of a binary system, the typical velocity would be the orbital velocity $v/c \sim \epsilon$ and the deviation from the flat metric would be the Newtonian potential, say $\Phi$ . Then these are related by $\Phi/c^{2} \sim v^{2}/c^{2} \sim \epsilon^{2}$ which guarantees that the system is bounded by its own gravity.

In the post-Newtonian approximation, the equations of general relativity take the form of Newton's equations in an appropriate limit as $\epsilon \rightarrow 0$ . Such a limit is called the Newtonian limit and it will be the basis of constructing the post-Newtonian approximation. However, the limit is not in any sense trivial since it may be thought of as two limits tied together as just described. It is also worth noting that the Newtonian limit cannot be uniform everywhere for all time. For example any compact binary system, no matter how weak the gravity between its components and slow its orbital motion is, will eventually spiral together due to backreaction from the emission of gravitational waves. As the result the effects of its Newtonian gravity will be swamped by those of its gravitational waves. This will mean that higher order effects of the post-Newtonian approximation eventually dominate the lowest order Newtonian dynamics and thus if the post-Newtonian approximation is not carefully constructed, this effect can lead to many

formal problems, such as divergent integrals $[71]$ . It has been shown that such divergences may be avoided by carefully defining the Newtonian limit $[79]$ . Moreover, the use of such a limit provides us a strong indication that the post-Newtonian hierarchy is an asymptotic approximation to general relativity $[82]$ . Therefore we shall first discuss in this paper the Newtonian limit and how to construct the post-Newtonian hierarchy before attacking practical problems in later sections.

Before going into the details, we mention the reason for the growth of interest in the post-Newtonian approximation in recent years. Certainly the discovery of the binary neutron star system PSR 1913+16 was a strong reason to have renewed interest in the post-Newtonian approximation, since it is the first system found in which general relativistic gravity plays a fundamental role in its evolution [89]. Particularly the indirect discovery of gravitational waves by the observation of the period shortening led to many fruitful studies of the equations of motion with gravitational radiation emission in 1980s (see [47, 48, 49] for a review). The effect of radiation reaction appears in the form of a potential force at the order of $\epsilon^5$ higher than the Newtonian force in the equations of motion. Ehlers and colleagues [71] critically discussed the foundation of the so-called quadrupole formula for the radiation reaction (see also the introduction of [129]). There have been various attempts to show the validity of the quadrupole formula [5, 47, 79, 90, 104, 105, 106, 107, 135, 138, 139, 155, 157, 156]. Namely, Damour [46] proved the formula for compact binaries with the help of the “dominant Schwarzschild condition” [47]. Blanchet and Damour [21] proved it for general fluid systems.

At that time, however, no serious attempts with direct detection of gravitational waves in mind had been made for the study of higher order effects in the equations of motion. The situation changed gradually in the late 1980s because of the increasing expectation of a direct detection of gravitational waves by kilometer-size interferometric gravitational wave detectors, such as LIGO [1, 116], VIRGO [35, 153], GEO [62, 83], and TAMA [114, 149]. Coalescing binary neutron stars are the most promising candidates of sources of gravitational waves for such detectors. The reasons are that (i) we expect, say, the (initial) LIGO to detect the signal of coalescence of binary neutron stars about once per year to once per hundreds of years [38, 103, 102], and (ii) the waveform from coalescing binaries can be predicted with high accuracy compared to other sources [1, 151, 161]. Information carried by gravitational waves tells us not only various physical parameters of neutron stars [45], but also the cosmological parameters [75, 119, 141, 142, 158] if and only if we can make a detailed comparison between the observed signal with theoretical predictions during the epoch of the so-called inspiraling phase where the orbital separation is much larger than the radius of the component stars [44]. This is the place where the post-Newtonian approximation may be applied to make theoretical templates for gravitational waves. The problem is that in order to make any meaningful comparison between theory and observation we need to know the detailed waveforms generated by the motion up to, say, 4 PN order which is of order $\epsilon^8$ higher than the Newtonian order [2, 3, 10, 147]. This request from gravitational wave astronomy forces us to construct higher order post-Newtonian equations of motion and waveform templates.

Replying to this request, there have been various works studying the equations of motion for a compact binary system and developing higher order post-Newtonian gravitational waveform templates for such a system. The most systematic among those works that have succeeded in achieving higher order iteration are the ones by Blanchet, Damour, and Iyer who have developed a scheme to calculate the waveform at a higher order, where the post-Minkowskian approximation is used to construct the external field and the post-Newtonian approximation is used to construct the field near the material source. They and their collaborators have obtained the waveform up to 3.5 PN order which is of order $\epsilon^7$ higher than the lowest quadrupole wave [23, 24, 29, 31, 32] by using the equations of motion up to that order [22, 91, 93, 111, 123, 130]. The 3.5 PN waveform includes tail terms which manifest nonlinearity of general relativity. Blanchet and Schäfer [33] have investigated a spectral (Fourier) decomposition of the tail and computed the contribution of the tail to the gravitational wave luminosity emitted by a binary system having a general eccentric

orbit. Asada and Futamase [12] showed that the dominant part of the tail term originates from the phase shift of the wave due to the Coulomb part of the gravitational field.

In this paper, we mainly discuss the foundation of the Newtonian limit and the post-Newtonian equations of motion for relativistic compact binaries in an inspiralling phase. In the next Section 1.2 we briefly give an historical introduction on the latter topic.

# 1.2 Post-Newtonian equations of motion

Many authors have derived equations of motion up to the 1 PN order [66, 73, 76, 94, 117, 152], up to 2.5 PN order [30, 47, 50, 51, 52, 46, 87, 95, 113, 130], and up to 3 PN order [22, 91, 93]. The 3.5 PN correction to the equations of motion has been derived by [111, 123, 130]. The post-Newtonian equations of motion are now available in harmonic coordinates up to 3.5 PN order inclusively. See also [17, 85, 86, 96] for the 3.5 PN and the 4.5 PN correction based on the energy balance. Besides the equations of motion, attempts have been made to derive a Hamiltonian for a binary system. The second order post-Newtonian computation of the Hamiltonian was tackled by [125, 126, 127] and completed in [56, 135, 136, 137]. Damour, Jaranowski, and Schäfer have completed the 3 PN order Hamiltonian in [54]. So far, the post-Newtonian Arnowitt–Deser–Misner (ADM) Hamiltonian is available in ADM transverse-traceless coordinates up to 3.5 PN order inclusively [97].

The equations of motion for a two point mass binary in harmonic coordinates up to 2.5 PN order, at which the radiation reaction effect first appears, were derived by Damour and Deruelle $[52, 46]$ based on the post-Minkowskian approach $[14]$ . These works used Dirac delta distributions to express the point masses mathematically, therefore they inevitably resorted to a purely mathematical regularization procedure to deal with divergences arising from the nonlinearity of general relativity. Damour $[47]$ addressed the applicability of the use of a Dirac delta distribution to self-gravitating objects. By investigating the tidal effect exerted by the companion star on the main star, he gave a plausible argument known as the “dominant Schwarzschild condition” which supports that up to 2.5 PN order, the field around the main star is recovered by the energy momentum tensor expressed in terms of a Dirac delta distribution.

Direct validations of the 2.5 PN equations of motion by Damour and Deruelle, where a Dirac delta distribution was not used, have been obtained in several works $[87, 95, 113, 130]$ . Grishchuk and Kopeikin $[87]$ and Kopeikin $[113]$ worked on extended but intrinsically spherical bodies with weak internal gravity using the post-Newtonian approximation both inside and outside the stars. They volume-integrated the equations of the conservation of the stress-energy tensor of the matter for an ideal fluid with two compact supports and obtained their equations of motion. The volume integral approach was adopted also by Pati and Will $[130]$ . The present authors and Asada on the other hand derived the 2.5 PN equations of motion $[95]$ for point particles with arbitrarily strong internal gravity using a regular point particle limit called the strong field point particle limit $[81]$ . These authors also used the local conservation law but adopted a surface integral approach $[73]$ , and they did not specify an explicit form of the stress-energy tensor but assumed that it satisfies some scaling on the initial hypersurface.

Blanchet, Faye, and Ponsot [30] also derived the 2.5 PN equations of motion using Dirac delta distributions for which Hadamard's partie finie regularization was employed to handle the divergences due to their use of Dirac delta distributions. In this approach, they have assumed that the two point masses follow regularized geodesic equations. (More precisely, they have assumed that the dynamics of two point masses are described by a regularized action, from which a regularized geodesic equation was shown to be derived.) They also derived the gravitational field up to 2.5 PN order in an explicit form which may help constructing initial data for numerical simulations of compact binaries.

All the works quoted above agree with each other. Namely, our work [95] shows the applicability of the Damour and Deruelle 2.5 PN equations of motion to a relativistic compact binary

consisting of regular stars with strong internal gravity. We mention here that stars consisting of relativistic compact binaries, for which we are searching as gravitational wave sources, have a strong internal gravitational field, and that it is a nontrivial question whether a star follows the same orbit regardless of the strength of its internal gravity.

Currently we have the equations of motion for relativistic compact binaries through the 3.5 PN approximation of general relativity in hand. Actually the 3.5 PN correction to the equations of motion is relatively easily derived [111, 123, 130]. At 3 PN order, an issue on undetermined coefficient associated with the regularization procedures was found which we now briefly discuss.

A 3 PN iteration result was first reported by Jaranowski and Schäfer [98, 99]. There a 3 PN ADM Hamiltonian in the ADM transverse traceless (ADMTT) gauge for two point masses expressed as two Dirac delta distributions was derived based on the ADM canonical approach [125, 135]. However, it was found in [98, 99] that the regularization they had used caused one coefficient $\omega_{\mathrm{kinetic}}$ to be undetermined in their framework. Moreover, they later found another undetermined coefficient in their Hamiltonian, called $\omega_{\mathrm{static}}$ [100]. Origins of these two coefficients were attributed to some unsatisfactory features of regularization they had used, such as violation of the Leibniz rule. The former coefficient, which appears as a numerical multiplier of the term that depends on the momenta of the point particles, was then fixed as $\omega_{\mathrm{kinetic}} = 41/24$ by a posteriori imposing Poincaré invariance on their 3 PN Hamiltonian [53]. As for the latter coefficient, Damour et al. [54] succeeded in fixing it as $\omega_{\mathrm{static}} = 0$ , adopting dimensional regularization $^{1}$ . Moreover, with this method they found the same value of $\omega_{\mathrm{kinetic}}$ as in [53], which ensures Lorentz invariance of their Hamiltonian.

On the other hand, Blanchet and Faye have tackled the derivation of the 3 PN equations of motion for two point masses expressed as two Dirac delta distributions in harmonic coordinates $[25, 27]$ based on their previous work $[30]$ . The divergences due to their use of Dirac delta distributions were systematically regularized with the help of Lorentz invariant generalized Hadamard partie finie regularization. They have extended the notion of the Hadamard partie finie regularization to regularize divergent integrals and a singular function which does not permit a power-like expansion near its singularities $[26]$ . Furthermore, the regularization is carefully constructed in $[28]$ so that it respects Lorentz invariance. Their equations of motion respect the Lorentz invariance in the post-Newtonian perturbative sense and admits a conserved energy of orbital motion modulo the 2.5 PN radiation reaction effect. They found, however, that there exists one and only one undetermined coefficient (which they call $\lambda$ ).

Interestingly, the two groups independently constructed a transformation between the two gauges and found that these two results coincide with each other when there exists a relation [55, 64]

$$
\omega_ {\text { static }} = - \frac {1 1}{3} \lambda - \frac {1 9 8 7}{8 4 0}. \tag {1}
$$

Therefore, it is possible to fix the $\lambda$ parameter from the result of [54] as

$$
\lambda = - \frac {1 9 8 7}{3 0 8 0}. \tag {2}
$$

However, the applicability of mathematical regularization to the current problem is not a trivial issue, but an assumption to be verified, or at least supported by convincing arguments. There is no argument such as the “dominant Schwarzschild condition” [47] at 3 PN order.

The present authors have derived the 3 PN equations of motion for relativistic compact binaries [91, 92, 93] in harmonic coordinates based on their previous work [95]. Namely, they did not use Dirac delta distributions. As a result, they did not find any undetermined coefficient at all in

the equations of motion and found the same value of the $\lambda$ parameter as Equation (2). Thus, the issue of the undetermined coefficient problem has been solved.

Physically equivalent 3 PN equations of motion in harmonic coordinates were also completed by Blanchet, Damour, and Esposito-Farèse [22] based on their previous works [25, 27, 30]. There, they have used the dimensional regularization to overcome the problem with the (generalized) Hadamard partie finie. The physical equivalence between the two results, Blanchet–Damour–Esposito-Farèse's and our equations of motion, suggests that, at least up to 3 PN order, a particle (with a strong internal gravity) follows a regularized (in some sense) geodesic equation in a dynamical spacetime, a part of whose gravitational field the particle itself generates.

Will and his collaborators $[130, 129, 163]$ have been tackling the 3 PN iteration of the equations of motion where they take into account the density profile of the stars explicitly. Their result may (or may not) give a direct check of the effacement principle $[47]$ up to 3 PN order which states that the motion of the objects depends only on their masses and not on their internal structures up to the order where the tidal effect comes into play.

# 1.3 Plan of this paper

This article is organized as follows: In Section 2 we introduce the Newtonian limit in general relativity and present how to construct the post-Newtonian hierarchy. There we mention how to avoid divergent integrals which appear at a higher order in the previous treatments. We shall then give a formulation of the dynamics of a Newtonian perfect fluid as the lowest order post-Newtonian approximation in harmonic coordinates.

We focus our attention on the derivation of the 3 PN equations of motion based on the work by the authors of this article from Section 3 to Section 8.

In Section 3, we discuss the basics of our derivation of the post-Newtonian equations of motion for relativistic compact binaries. There we discuss how to incorporate strong internal gravity in the post-Newtonian approximation.

In Section 4, we formulate our method in more detail. As an application, we derive the Newtonian and the 1 PN equations of motion in our formalism.

In Section 5, we briefly explain how to derive the 3 PN gravitational field when possible, and how to derive equations of motion when the gravitational field is unavailable in an explicit closed form.

In Sections 6 and 7, we derive the mass-energy relation and the momentum-velocity relation up to 3 PN order, and finally in Section 8, we derive the 3 PN equations of motion. The resulting equations of motion respect Lorentz invariance, admit a conserved energy (modulo the 2.5 PN radiation reaction effect), and are free from any ambiguity. Section 8 ends with a summary and some remarks on possible future study of the 4 PN order iteration.

In Appendix A, we give a brief sketch of the direct integration of the relaxed Einstein equations (DIRE) method by Will and Wiseman [129, 162, 165] which we have used in this article.

Although in the main body of this article we focus our attention on spherical massive bodies (or point particles), we shall apply our formalism to extended bodies in Appendix B.

In Appendix C, using the strong field point particle limit and the surface integral approach, we discuss that a particle with strong internal gravity moves on a geodesic of some smooth metric part which is produced by the particle itself. This work is done by the authors in the collaboration with Takashi Fukumoto.

Throughout this article, we will use units where G = c = 1 unless otherwise mentioned.

# 2 Foundation of the Post-Newtonian Approximation

Since the Newtonian limit is the basis of the post-Newtonian approximation, we shall first formulate the Newtonian limit. We follow the formulation by Futamase and Schutz $[82]$ . We will not mention other formulations of the post-Newtonian approximation $[63, 70, 134]$ .

# 2.1 Newtonian limit along a regular asymptotic Newtonian sequence

This formulation is based on the observation that any asymptotic approximation of any theory needs a sequence of solutions of the basic equations of the theory $[146, 154]$ . Namely, if we write the equations in abstract form as

$$
E (g) = 0 \tag {3}
$$

for an unknown function g, one would like to have a one-parameter (or possibly multi-parameter) family of solutions,

$$
E (g (\lambda)) = 0, \tag {4}
$$

where $\lambda$ is some parameter. Asymptotic approximation then says that a function $f(\lambda)$ approximates $g(\lambda)$ to order $\lambda^{p}$ if $|f(\lambda)-g(\lambda)|/\lambda^{p}\to0$ as $\lambda\to0$ . We choose the sequence of solutions with appropriate properties in such a way that the properties reflect the character of the system under consideration.

We shall formulate the post-Newtonian approximation according to the general idea just described. As stated in the introduction, we would like to have an approximation which applies to the systems whose motions are described almost by Newtonian theory. Thus we need a sequence of solutions of the Einstein equations parameterized by $\epsilon$ (the typical velocity of the system divided by the speed of light) which has Newtonian character as $\epsilon \rightarrow 0$ .

The Newtonian character is most conveniently described by the following scaling law. The Newtonian equations involve six variables, namely the density $\rho$ , the pressure P, the gravitational potential $\Phi$ , and the velocity $v^{i}, i = 1, 2, 3$ :

$$
\nabla^ {2} \Phi - 4 \pi \rho = 0, \tag {5}
$$

$$
\partial_ {t} \rho + \nabla_ {i} (\rho v ^ {i}) = 0, \tag {6}
$$

$$
\rho \partial_ {t} v ^ {i} + \rho v ^ {j} \nabla_ {j} v ^ {i} + \nabla^ {i} P + \rho \nabla^ {i} \Phi = 0, \tag {7}
$$

supplemented by an equation of state. For simplicity we have considered a perfect fluid.

It can be seen that the variables $\{\rho(x^{i},t),P(x^{i},t),\Phi(x^{i},t),v^{i}(x^{j},t)\}$ obeying the above equations satisfy the following scaling law:

$$
\rho (x ^ {i}, t) \rightarrow \epsilon^ {2} \rho (x ^ {i}, \epsilon t),
$$

$$
P \left(x ^ {i}, t\right)\rightarrow \epsilon^ {4} P \left(x ^ {i}, \epsilon t\right), \tag {8}
$$

$$
v ^ {i} (x ^ {k}, t) \rightarrow \epsilon v ^ {i} (x ^ {k}, \epsilon t),
$$

$$
\Phi (x ^ {i}, t) \rightarrow \epsilon^ {2} \Phi (x ^ {i}, \epsilon t).
$$

One can easily understand the meaning of this scaling by noticing that $\epsilon$ is the magnitude of the typical velocity (divided by the speed of light). Then the magnitude of the gravitational potential will be of order $\epsilon^{2}$ because of the balance between gravity and the centrifugal force. The scaling of the time variable expresses the fact that the weaker gravity is $(\epsilon \rightarrow 0)$ the longer the time scale is.

Thus we wish to have a sequence of solutions of the Einstein equations which has the above scaling as $\epsilon\to0$ . We shall also take the point of view that the sequence of solutions is determined by the appropriate sequence of initial data. This has a practical advantage because there will be no solutions of the Einstein equations which satisfy the above scaling (8) even as $\epsilon\to0$ . This is because the Einstein equations are nonlinear in the field variables, so it will not be possible to

enforce the scaling everywhere in spacetime. We shall therefore impose it only on the initial data for the solution of the sequence.

Here we first give a general discussion on the formulation of the post-Newtonian approximation independent of any initial value formalism and then present the concrete treatment in harmonic coordinates. The condition is used because of its popularity and some advantages in the generalization to the systems with strong internal gravity.

As the initial data for the matter we take the same data set in the Newtonian case, namely the density $\rho$ , the pressure P, and the coordinate velocity $v^{i}$ . In most of the application, we usually assume a simple equation of state which relates the pressure to the density. The initial data for the gravitational field are $g_{\mu\nu}, \partial g_{\mu\nu}/\partial t$ . Since general relativity is an overdetermined system, there will be constraint equations among the initial data for the field. We shall write the free data for the field as $(Q_{ij}, P_{ij})$ whose explicit forms depend on the coordinate condition one assumes. In any coordinates we shall assume these data for the field vanish since we are interested in the evolution of an isolated system by its own gravitational interaction. It is expected that this choice corresponds to the absence of radiation far away from the source. Thus we choose the following initial data which is indicated by the Newtonian scaling:

$$
\rho (t = 0, x ^ {i}, \epsilon) = \epsilon^ {2} a (x ^ {i}),
$$

$$
P (t = 0, x ^ {i}, \epsilon) = \epsilon^ {4} b (x ^ {i}),
$$

$$
v ^ {i} (t = 0, x ^ {k}, \epsilon) = \epsilon c ^ {i} (x ^ {k}), \tag {9}
$$

$$
Q _ {i j} (t = 0, x ^ {i}, \epsilon) = 0,
$$

$$
P _ {i j} (t = 0, x ^ {i}, \epsilon) = 0,
$$

where the functions $a$ , $b$ , and $c^i$ are $C^\infty$ functions that have compact support contained entirely within a sphere of a finite radius.

Corresponding to the above data, we have a one-parameter set of spacetime parameterized by $\epsilon$ . It may be helpful to visualize the set as a fiber bundle, with base space R being the real line (coordinate $\epsilon$ ) and fiber $R^{4}$ being the spacetime (coordinates $t, x^{i}$ ). The fiber $\epsilon = 0$ is Minkowski spacetime since it is defined by zero data. In the following we shall assume that the solutions are sufficiently smooth functions of $\epsilon$ for small $\epsilon \neq 0$ . We wish to take the limit $\epsilon \to 0$ along the sequence. The limit is, however, not unique and is defined by giving a smooth nowhere vanishing vector field on the fiber bundle which is nowhere tangent to each fiber [84, 146]. The integral curves of the vector field give a correspondence between points in different fibers, namely events in different spacetimes with different values of $\epsilon$ . Remembering the Newtonian scaling of the time variable in the limit, we introduce the Newtonian dynamical time,

$$
\tau = \epsilon t, \tag {10}
$$

and define the integral curve as the curve on which $\tau$ and $x^{i}$ stay constant $^{2}$ . In fact if we take the limit $\epsilon\to0$ along this curve, the orbital period of the binary system with $\epsilon=0.01$ is 10 times that of the system with $\epsilon=0.1$ as expected from the Newtonian scaling. This is what we define as the Newtonian limit. Notice that this map never reaches the fiber $\epsilon=0$ (Minkowski spacetime). There is no pure vacuum Newtonian limit as expected.

In the following we assume the existence of such a sequence of solutions constructed by the initial data satisfying the above scaling with respect to $\epsilon$ . We shall call such a sequence a regular asymptotically Newtonian sequence. We have to make further mathematical assumptions about the sequence to make explicit calculations. We will not go into details partly because in order to prove the assumptions we need a deep understanding of the existence and uniqueness properties of the Cauchy problem of the Einstein equations with perfect fluids of compact support which are not available at present.

# 2.2 Post-Newtonian hierarchy

We shall now define the Newtonian, post-Newtonian, and higher approximations of various quantities as the appropriate higher tangents of the corresponding quantities to the above integral curve at $\epsilon = 0$ . For example the hierarchy of approximations for the spacetime metrics can be expressed as follows:

$$
\begin{array}{l} g _ {\mu \nu} (\epsilon , \tau , x ^ {i}) = g _ {\mu \nu} (0, \tau , x ^ {i}) + \epsilon (\mathcal {L} _ {V} g _ {\mu \nu}) (0, \tau , x ^ {i}) \\ + \frac {1}{2} \epsilon^ {2} (\mathcal {L} _ {V} ^ {2} g _ {\mu \nu}) (0, \tau , x ^ {i}) + \dots + \frac {\epsilon^ {n}}{n !} (\mathcal {L} _ {V} ^ {n} g _ {\mu \nu}) (0, \tau , x ^ {i}) + R _ {n + 1}, \tag {11} \\ \end{array}
$$

where $L_{V}$ is the Lie derivative with respect to the tangent vector of the curves defined above, and the remainder term $R_{n+1}^{\mu\nu}$ is

$$
R _ {n + 1} ^ {\mu \nu} = \frac {\epsilon^ {n + 1}}{(n + 1) !} \int_ {0} ^ {1} d \ell (1 - \ell) ^ {n + 1} (\mathcal {L} _ {V} ^ {n + 1} g _ {\mu \nu}) (\ell \epsilon , \tau , x ^ {i}). \tag {12}
$$

Taylor's theorem guarantees that the series is an asymptotic expansion about $\epsilon = 0$ under certain assumptions mentioned above. It may be useful to point out that the above definition of the approximation scheme may be formulated purely geometrically in terms of a jet bundle.

The above definition of the post-Newtonian hierarchy gives us an asymptotic series in which each term in the series is manifestly finite. This is based on the $\epsilon$ dependence of the domain of dependence of the field point $(\tau, x^{k})$ . The region is finite with finite values of $\epsilon$ , and the diameter of the region increases like $\epsilon^{-1}$ as $\epsilon \to 0$ . Without this linkage of the region with the expansion parameter $\epsilon$ , the post-Newtonian approximation leads to divergences in the higher orders. This is closely related to the retarded expansion. Namely, it is assumed that the slow motion assumption enables one to Taylor expand the retarded integrals in retarded time such as

$$
\int d r f (\tau - \epsilon r, \dots) = \int d r f (\tau , \dots) - \epsilon \int d r r f (\tau , \dots), _ {\tau} + \dots , \tag {13}
$$

and assign the second term to a higher order because of its explicit $\epsilon$ in front. This is incorrect because $r \to \epsilon^{-1}$ as $\epsilon \to 0$ and thus $\epsilon r$ is not uniformly small in the Newtonian limit. Only if the integrand falls off sufficiently fast, the retardation can be ignored. This happens in the lower order PN terms. But at some higher order there appear many terms which do not fall off sufficiently fast because of the nonlinearity of the Einstein equations. This is the reason that the formal PN approximation produces the divergent integrals. It turns out that such a divergence appears at 4 PN order, indicating a breakdown of the PN approximation in harmonic coordinates $[20]^{3}$ . This sort of divergence may be eliminated if we remember that the upper bound of the integral does depend on $\epsilon$ as $\epsilon^{-1}$ . Thus we would get something like $\epsilon^{n}\ln\epsilon$ instead of $\epsilon^{n}\ln\infty$ in the usual approach. This shows that the asymptotic Newtonian sequence is not differentiable in $\epsilon$ at $\epsilon = 0$ , but there is no divergence in the expansion and it has still an asymptotic approximation in $\epsilon$ that involves logarithms.

Other than the initial value formulation method $[79, 82]$ mentioned above, various methods have been proposed to solve this problem of the divergent integrals. It is known that a higher order post-Newtonian metric does not respect the asymptotically flat condition. This does not mean that the post-Newtonian approximation is useless at such a high order. The problem is related to the fact that a simple post-Newtonian iteration is meaningful only in the near zone – about one wavelength distance away from the material source – and is not useful outside of the near zone, called far zone, where the wave effect (retardation effect) is manifest. So roughly

speaking, if a far zone metric satisfying proper boundary conditions at infinity is solved so that we have a boundary condition to the field equations for a post-Newtonian metric in the buffer zone, we can find a post-Newtonian metric which is meaningful in the sense that it respects the correct behaviour at the near zone boundary.

Blanchet and Damour have developed a systematic approach of a matched asymptotic expansion. They solved the far zone metric using a multipolar post-Minkowskinan expansion. The far zone metric satisfies a stationarity condition and is parametrized by radiative multipole moments. On the other hand, they solve a post-Newtonian near zone metric up to a homogeneous solution. They then establish an association between those radiative multipole moments and the source multipole moments that characterize a post-Newtonian near zone metric to fix the homogeneous solution, and find a post-Newtonian metric which satisfies the correct behaviour in the buffer zone.

Will and Wiseman have developed the DIRE method $[129, 162, 165]$ where they split the integral region in the retarded integral into two – one being the near zone and the other being the far zone. The near zone metric is solved by a post-Newtonian expansion. The retarded integral over the far zone is directly evaluated with the assumption of sufficient stationarity of the system in the infinite past.

In fact, both the Blanchet–Damour method and the Will–Wiseman method are proved to give a physically equivalent result $[19]$ . In this paper, for our computation of the 3 PN equations of motion, we will use the Will and Wiseman method to solve the problem of the breakdown of the post-Newtonian approximation in the near zone.

# 2.3 Explicit calculation in harmonic coordinates

Here we shall use the above formalism to make an explicit calculation in harmonic coordinates. The reduced Einstein equations in the harmonic condition are written as

$$
\tilde {g} ^ {\alpha \beta} \tilde {g} _ {, \alpha \beta} ^ {\mu \nu} = 1 6 \pi \Theta^ {\mu \nu} - \tilde {g} _ {, \beta} ^ {\mu \alpha} \tilde {g} _ {, \alpha} ^ {\nu \beta}, \tag {14}
$$

$$
\partial_ {\mu} \left[ \tilde {g} ^ {\mu \nu} \partial_ {\nu} x ^ {\alpha} \right] = 0, \tag {15}
$$

where

$$
\tilde {g} ^ {\mu \nu} = (- g) ^ {1 / 2} g ^ {\mu \nu}, \tag {16}
$$

$$
\Theta^ {\alpha \beta} = (- g) (T ^ {\alpha \beta} + t _ {\mathrm{LL}} ^ {\alpha \beta}), \tag {17}
$$

where $t_{LL}^{\mu\nu}$ is the Landau–Lifshitz pseudotensor [115]. In this section we shall choose an isentropic perfect fluid for $T^{\alpha\beta}$ which is enough for most applications,

$$
T ^ {\alpha \beta} = (\rho + \rho \Pi + P) u ^ {\alpha} u ^ {\beta} + P g ^ {\alpha \beta}, \tag {18}
$$

where $\rho$ is the rest mass density, $\Pi$ the internal energy, P the pressure, and $u^{\mu}$ the four-velocity of the fluid with normalization

$$
g _ {\alpha \beta} u ^ {\alpha} u ^ {\beta} = - 1. \tag {19}
$$

The conservation of energy and momentum is expressed as

$$
\nabla_ {\beta} T ^ {\alpha \beta} = 0. \tag {20}
$$

Defining the gravitational field variable as

$$
h ^ {\mu \nu} = \eta^ {\mu \nu} - (- g) ^ {1 / 2} g ^ {\mu \nu}, \tag {21}
$$

where $\eta^{\mu\nu}$ is the Minkowski metric, the reduced Einstein equations (14) and the gauge condition (15) take the following form:

$$
\left(\eta^ {\alpha \beta} - h ^ {\alpha \beta}\right) h _ {, \alpha \beta} ^ {\mu \nu} = - 1 6 \pi \Theta^ {\mu \nu} + h _ {, \beta} ^ {\mu \alpha} h _ {, \alpha} ^ {\nu \beta}, \tag {22}
$$

$$
h _ {, \nu} ^ {\mu \nu} = 0. \tag {23}
$$

Thus the characteristics are determined by the operator $(\eta^{\alpha\beta}-h^{\alpha\beta})\partial_{\alpha}\partial_{\beta}$ , and thus the light cone deviates from that in the flat spacetime. We may use this form of the reduced Einstein equations in the calculation of the waveform far away from the source because the deviation plays a fundamental role there [12]. However, in the study of the gravitational field near the source it is not necessary to consider the deviation of the light cone from the flat one and thus it is convenient to use the following form of the reduced Einstein equations [5]:

$$
\eta^ {\mu \nu} h _ {, \mu \nu} ^ {\alpha \beta} = - 1 6 \pi \Lambda^ {\alpha \beta}, \tag {24}
$$

where

$$
\Lambda^ {\alpha \beta} = \Theta^ {\alpha \beta} + \chi_ {, \mu \nu} ^ {\alpha \beta \mu \nu}, \tag {25}
$$

$$
\chi^ {\alpha \beta \mu \nu} = (1 6 \pi) ^ {- 1} (h ^ {\alpha \nu} h ^ {\beta \mu} - h ^ {\alpha \beta} h ^ {\mu \nu}). \tag {26}
$$

Equations (23) and (24) together imply the conservation law

$$
\Lambda_ {, \beta} ^ {\alpha \beta} = 0. \tag {27}
$$

We shall take as our variables the set $\{\rho, P, v^i, h^{\alpha\beta}\}$ , with the definition

$$
v ^ {i} = u ^ {i} / u ^ {0}. \tag {28}
$$

The time component of four-velocity $u^{0}$ is determined from Equation (19). To make a well-defined system of equations we must add the conservation law for the number density n, which is some function of the density $\rho$ and pressure P:

$$
\nabla_ {\alpha} (n u ^ {\alpha}) = 0. \tag {29}
$$

Equations (27) and (29) imply that the flow is adiabatic. The role of the equation of state is played by the arbitrary function $n(\rho, p)$ .

Initial data for the above set of equations are $h^{\alpha\beta}, h^{\alpha\beta,0}, \rho, P$ , and $v^{i}$ , but not all these data are independent because of the existence of the constraint equations. Equations (23) and (24) imply the four constraint equations among the initial data for the field,

$$
\Delta h ^ {\alpha 0} + 1 6 \pi \Lambda^ {\alpha 0} - \delta^ {i j} h _ {i, j} ^ {\alpha , 0} = 0, \tag {30}
$$

where $\Delta$ is the Laplacian in the flat space. We shall choose $h^{ij}$ and $h^{ij,0}$ as free data and solve Equation (30) for $h^{\alpha0}(\alpha=0,\ldots,3)$ and Equation (23) for $h^{\alpha0,0}$ . Of course these constraints cannot be solved explicitly, since $\Lambda^{\alpha0}$ contains $h^{\alpha0}$ , but they can be solved iteratively as explained below. As discussed above, we shall assume that the free data $h^{ij}$ and $h^{ij,0}$ for the field vanish. One can show that such initial data satisfy the O'Murchadha and York criterion for the absence of radiation far away from the source [124].

In the actual calculation, it is convenient to use an expression with explicit dependence of $\epsilon$ . The harmonic condition allows us to have such an expression in terms of the retarded integral,

$$
h ^ {\mu \nu} (\epsilon , \tau , x ^ {i}) = 4 \int_ {C (\epsilon , \tau , x ^ {i})} d ^ {3} y   \Lambda^ {\mu \nu} (\tau - \epsilon r, y ^ {i}, \epsilon) / r + h _ {\mathrm{H}} ^ {\mu \nu} (\epsilon , \tau , x ^ {i}), \tag {31}
$$

where $r = |y^{i} - x^{i}|$ and $C(\epsilon, \tau, x^{i})$ is the past flat light cone of the event $(\tau, x^{i})$ in the spacetime given by $\epsilon$ , truncated where it intersects with the initial hypersurface $\tau = 0$ . $h_{H}^{\mu\nu}$ is the unique solution of the homogeneous wave equation in the flat spacetime,

$$
\Box h _ {\mathrm{H}} ^ {\mu \nu} = \eta^ {\alpha \beta} h _ {H, \alpha \beta} ^ {\mu \nu} = 0. \tag {32}
$$

$h_{H}^{\mu\nu}$ evolves from a given initial data on the $\tau = 0$ initial hypersurface which are subject to the constraint equations (30). The explicit form of $h_{H}^{\mu\nu}$ is available via the Poisson formula (see e.g. [144]),

$$
h _ {\mathrm{H}} ^ {\mu \nu} (\tau , x ^ {i}) = \frac {\tau}{4 \pi} \oint_ {\partial C (\tau , x ^ {i})} h _ {, \tau} ^ {\mu \nu} (\tau = 0, y ^ {i}) d \Omega_ {y} + \frac {1}{4 \pi} \frac {\partial}{\partial \tau} \left[ \tau \oint_ {\partial C (\tau , x ^ {i})} h ^ {\mu \nu} (\tau = 0, y ^ {i}) d \Omega_ {y} \right]. (3 3)
$$

We shall henceforth ignore the homogeneous solutions because they play no important role. Because of the $\epsilon$ dependence of the integral region, the domain of integral is finite as long as $\epsilon \neq 0$ and their diameter increases like $\epsilon^{-1}$ as $\epsilon \to 0$ .

Given the formal expression (31) in terms of initial data (9), we can take the Lie derivative and evaluate these derivatives at $\epsilon = 0$ . The Lie derivative is nothing but a partial derivative with respect to $\epsilon$ in the coordinate system for the fiber bundle given by $(\epsilon, \tau, x^i)$ . Accordingly one should convert all the time indices to $\tau$ indices. For example, $T^{\tau \tau} = \epsilon^2 T^{tt}$ which is of order $\epsilon^4$ , since $T^{tt} \sim \rho$ is of order $\epsilon^2$ . Similarly the other components of stress-energy tensor $T^{\tau i} = \epsilon T^{ti}$ and $T^{ij}$ are of order $\epsilon^4$ as well. Thus we expect that the first nonvanishing derivative in Equation (31) will be the forth derivative. In fact we find

$$
{ } _ { 4 } h ^ { \tau \tau } ( \tau , x ^ { i } ) = 4 \int _ { R ^ { 3 } } \frac { { } _ { 2 } \rho ( \tau , y ^ { i } ) } { r } d ^ { 3 } y , \tag {34}
$$

$$
_ 4 h ^ {\tau i} (\tau , x ^ {i}) = 4 \int_ {R ^ {3}} \frac {2 \rho (\tau , y ^ {k}) _ {1} v ^ {i} (\tau , x ^ {k})}{r} d ^ {3} y, \tag {35}
$$

$$
{ } _ { 4 } h ^ { i j } ( \tau , x ^ { k } ) = 4 \int _ { R ^ { 3 } } \frac { { } _ { 2 } \rho ( \tau , y ^ { k } ) _ { 1 } v ^ { i } ( \tau , y ^ { k } ) _ { 1 } v ^ { j } ( \tau , y ^ { k } ) + { } _ { 4 } t _ { \mathrm{LL} } ^ { i j } ( \tau , y ^ { k } ) } { r } d ^ { 3 } y , \tag {37}
$$

where we have adopted the notation

$$
{ } _ { n } f ( \tau , x ^ { i } ) = \frac { 1 } { n ! } \operatorname* { l i m } _ { \epsilon \to 0 } \frac { \partial ^ { n } } { \partial \epsilon ^ { n } } f ( \epsilon , \tau , x ^ { i } ) , \tag {38}
$$

and

$$
{ } _ { 4 } t _ { \mathrm{LL} } ^ { i j } = \frac { 1 } { 6 4 \pi } \left( { } _ { 4 } h ^ { \tau \tau , i } { } _ { 4 } h ^ { \tau \tau , j } - \frac { 1 } { 2 } \delta ^ { i j } { } _ { 4 } h ^ { \tau \tau , k } { } _ { 4 } h ^ { \tau \tau } { } _ { , k } \right) . ( 3 9 )
$$

In the above calculation we have taken the point of view that $h^{\mu \nu}$ is a tensor field, defined by giving its components in the assumed harmonic coordinates as the difference between the tensor density $\sqrt{-g} g^{\mu \nu}$ and $\eta^{\mu \nu}$ .

The conservation law (27) also has its first nonvanishing derivatives at this order, which are

$$
{ } _ { 2 } \rho _ { , \tau } + ( { } _ { 2 } \rho _ { 1 } v ^ { i } ) _ { , i } = 0 , \tag {40}
$$

$$
\left(_ {2} \rho_ {1} v ^ {i}\right) _ {, \tau} + \left(_ {2} \rho_ {1} v ^ {i} _ {1} v ^ {j}\right) _ {, j} + _ {4} P ^ {i} - \frac {1}{4} _ {2} \rho_ {4} h ^ {\tau \tau , i} = 0. \tag {41}
$$

Equations (34), (40), and (41) constitute Newtonian theory of gravity. Thus the lowest nonvanishing derivative with respect to $\epsilon$ is indeed Newtonian theory, and the 1 PN and 2 PN equations emerge from the sixth and eighth derivatives, respectively, in the conservation law (27). At the next derivative, the quadrupole radiation reaction term emerges.

# 3 Post-Newtonian Equations of Motion for Compact Binaries

In this section we review briefly the strong field point particle limit, the surface integral approach for the evaluation of the gravitational force, and scalings of matter and field variables on an initial hypersurface. Also we revisit some elementary but useful equations, the Newtonian velocity momentum relation and the Newtonian equations of motion for extended bodies. These are the basic ideas of our formalism. On the base of our formalism, see also $[11, 91, 92, 93, 94, 95, 140]$ . On the post-Newtonian approximation and related issues from different viewpoints, see $[47, 48, 49, 16, 18, 19]$ and references therein.

# 3.1 Strong field point particle limit

In a relativistic compact binary system in an inspiralling phase, each star is well approximated by a point particle with a few low order multipole moments. This is because in the inspiralling phase, the stellar size divided by the orbital separation is much smaller than unity.

If we wish to apply the post-Newtonian approximation to the inspirating phase of binary neutron stars, the strong internal gravity must be taken into account. The usual post-Newtonian approximation explicitly assumes the weakness of the gravitational field everywhere including inside the material source. It is argued by applying the strong equivalence principle that the external gravitational field which governs the orbital motion of the binary system is independent of the internal structure of the components up to tidal interaction. Thus it is expected that the results obtained under the assumption of weak gravity also apply for the case of a neutron star binary. Experimental evidence for the strong equivalence principle is obtained only for systems with weak gravity $[159, 160, 161, 164]$ , but at present no experiment is available in a case with strong internal gravity.

In the theoretical aspect, the theory of extended objects in general relativity $[69]$ is still in a preliminary stage for an application to realistic systems. The matched asymptotic expansion technique has been used to treat a system with strong gravity in certain situations $[47, 65, 66, 104, 105]$ . Another way to handle strong internal gravity is by the use of a Dirac delta distribution type source with a fixed mass $[73]$ . However, this makes the Einstein equations mathematically meaningless because of their nonlinearity. Physically, there is no such source in general relativity because of the existence of black holes. Before a body shrinks to a point, it forms a black hole whose size is fixed by its mass. For this reason, it has been claimed that no point particle exists in general relativity.

This conclusion is not correct, however. We can shrink the body keeping the compactness $(M/R)$ , i.e. the strength of the internal field fixed. Namely we should scale the mass M just like the radius R. This can be fitted nicely into the concept of the regular asymptotic Newtonian sequence defined in Section 2 because there the mass also scales along the sequence of solutions. In fact, if we take the masses of two stars as M, and the separation between two stars as L, then $\Phi \sim M/L$ . Thus the mass M scales as $\epsilon^{2}$ if we fix the separation $^{4}$ . In the above we have assumed that the density scales as $\epsilon^{2}$ to guarantee this scaling for the mass while keeping the size of the body fixed. Now we shrink the size as $\epsilon^{2}$ to keep the compactness of each component. Then the density should scale as $\epsilon^{-4}$ . We shall call such a scheme the strong field point particle limit since the limit keeps the strength of internal gravity $^{5}$ . The above consideration suggests the following initial data to define a regular asymptotic Newtonian sequence which describes a nearly Newtonian

system with strong internal gravity [81]. The initial data are two uniformly rotating fluids with compact spatial support whose stress-energy tensor and size scales as $\epsilon^{-4}$ and $\epsilon^{2}$ , respectively. We also assume that each of these fluid configurations would be a stationary equilibrium solution of the Einstein equations if the other were absent. This is necessary for the suppression of irrelevant internal motions of each star. Any remaining motions are tidal effects caused by the other body, which will be of order $\epsilon^{6}$ smaller than the internal self-force. These data allow us to use the Newtonian time $\tau = \epsilon t$ as a natural time coordinate everywhere including the interior region of the stars.

As for multipole moments, the scaling $R = \mathcal{O}(\epsilon^{2})$ enables us to incorporate the multipole expansion of the stars into the post-Newtonian approximation. In inspiralling compact binaries the tidally and rotationally induced quadrupole moments are too small to affect the binary orbital motion. The time to coalesce is too short for the binary to be tidally locked and corotate [15, 44, 110]. The phase shift due to the quadrupole-orbit couplings may be negligible in the LIGO bandwidth [44]. However, to detect gravitational waves from inspiralling binaries, we have to have highly accurate prior knowledge about the binary orbital motion, say, 4 PN equations of motion or so. The effect of quadrupole moments on the orbital motion can be of about the same order as that of spin-spin interactions [132], while the spin-spin interactions appear at about 2.5 PN order for slowly rotating stars. Also, at the late inspiralling phase, an effect of extendedness of the stars on the motion will be important. Thus it is important to take the multipole moments of the stars into account in a way that is suitable for compact stars when we derive the equations of motion for an inspiralling compact binary.

# 3.2 Surface integral approach and body zone

One way to evaluate the gravitational force acting on a star is the volume integral approach. In the Newtonian case, the gravitational force $F_{1}^{i}$ on star 1 becomes

$$
F _ {1} ^ {i} = \int_ {B _ {1}} d ^ {3} x \rho \frac {\partial \Phi}{\partial x ^ {i}}. \tag {42}
$$

The integral region $B_{1}$ covers star 1 but does not cover the star 2 (the companion star). To evaluate the above integral we must know the internal structure, which is generally a difficult task even in the Newtonian dynamics, needless to say in general relativity. By means of the surface integral approach we can put off dealing with the internal structure problem until tidal effects affect the orbital motion. In the Newtonian case, using the Poisson equation and Gauss's law, we can rewrite the above volume integral into a surface integral,

$$
F _ {1} ^ {i} = - \oint_ {\partial B _ {1}} d S _ {j} t ^ {i j}, \tag {43}
$$

$$
t ^ {i j} \equiv \frac {1}{4 \pi} \left(\frac {\partial \Phi}{\partial x ^ {i}} \frac {\partial \Phi}{\partial x ^ {j}} - \frac {\delta^ {i j}}{2} \frac {\partial \Phi}{\partial x ^ {k}} \frac {\partial \Phi}{\partial x ^ {k}}\right). \tag {44}
$$

Note that the sphere $\partial B_{1}$ has no intersection, neither with star 1 nor star 2. Thus, we can evaluate the gravitational force acting on star 1 without knowledge about the internal structure of star 1.

The surface integral approach was used by Einstein, Infeld, and Hoffmann in general relativity $[73]$ . They used the vacuum Einstein equations only, and their method can be applied to any object including a black hole. We will take the surface integral approach in this article. But in our formalism, we shall treat only regular objects like a neutron star.

Now let us introduce the body zone to provide the surfaces of the surface integral approach. The scalings of $R$ and $m$ motivate us to define the body zone of star $A$ ( $A = 1,2$ ) as $B_A \equiv \{x^i ||\vec{x} - \vec{z}_A(\tau)| < \epsilon R_A\}$ and the body zone coordinates of the star $A$ as $\alpha_A^i \equiv \epsilon^{-2}(x^i - z_A^i(\tau))$ .

![](images/8e8449f66e2e0d9e03e4d53966744a129ae9899e8ab6347ab5ef81d02d750068.jpg)

<details>
<summary>text_image</summary>

B_A
X
</details>

![](images/5b137b9d7fcc771495b5acbbd02585b8264cc7cf165543a6f114ed37e3ee227b.jpg)

<details>
<summary>text_image</summary>

B_A
α
</details>

Figure 1: Body zone coordinates and near zone coordinates. In the near zone coordinates $(\tau, x^{i})$ , both the body zone and the star shrink as $\epsilon$ (thin dotted arrow) and $\epsilon^{2}$ (thick dotted arrow) respectively. Both the thin and thick arrows point inside. In the body zone coordinates $(\tau, \alpha_{A}^{i})$ , the star does not shrink while the body zone boundary goes to infinity as $\epsilon^{-1}$ (thin dotted line pointing outside).   
![](images/ce52defcfb0c5e55e91b43d454971bf7a7ffdc8b1728735bc4ef2658bbd40a86.jpg)

<details>
<summary>text_image</summary>

εR₂
B₂
Z₂
εR₁
B₁
Z₁
δB₁
</details>

Figure 2: Gravitational energy momentum flux through the body zone boundary. The meshed two circles represent stars 1 and 2. Each star is surrounded by the body zone represented here by a striped area. The arrows around star 1 represent the gravitational energy momentum flux flowing through the body zone boundary.

$z_{A}^{i}(\tau)$ is a representative point of the star A, e.g. the center of the mass of the star A. $R_{A}$ , called the body zone radius, is an arbitrary length scale (much smaller than the orbital separation and not identical to the radius of the star) and constant (i.e. $dR_{A}/d\tau = 0$ ). With the body zone coordinates, the star does not shrink when $\epsilon \to 0$ , while the boundary of the body zone goes to infinity (see Figure 1).

Then it is appropriate to define the star's characteristic quantities such as the mass, the spin, and so on with the body zone coordinates. On the other hand the body zone serves us with a surface $\partial B_A$ , through which gravitational energy momentum flux flows, and in turn it amounts to the gravitational force acting on star $A$ (see Figure 2). Since the body zone boundary $\partial B_A$ is far away from the surface of star $A$ , we can evaluate the gravitational energy momentum flux over $\partial B_A$ with the post-Newtonian gravitational field. In fact we shall express our equations of motion in terms of integrals over $\partial B_A$ and be able to evaluate them explicitly.

# 3.3 Scalings on the initial hypersurface

Following [79, 82, 138], we use the initial value formulation to solve the Einstein equations. As initial data for the matter variables and the gravitational field, we take a set of nearly stationary

solutions of the exact Einstein equations representing two widely separated fluid balls, each of which rotates uniformly. We assume that these solutions are parametrized by $\epsilon$ and that the matter and field variables have the following scalings.

The density scales as $\epsilon^{-4}$ (in the $(t,x^{i})$ coordinates), implied by the scalings of m and R. The scaling of the density suggests that the natural dynamical time (free fall time) $\eta$ inside the star may be $\eta=\epsilon^{-2}t$ . Then if we cannot assume almost stationary condition on the stars, it is difficult to use the post-Newtonian approximation [80, 81]. In practice, however, our formalism is still applicable to pulsating stars if the effect of pulsation is not important in the orbital motion.

The velocity of stellar rotation is assumed to be $\mathcal{O}(\epsilon)$ . In other words, we assume that the star rotates slowly and is pressure supported $^{6}$ . By this assumption, the spin-orbit coupling force appears at 2 PN order rather than the usual 1.5 PN order. The slowly spinning motion assumption is not crucial: In fact, it is straightforward to incorporate a rapidly spinning compact body into our formalism.

From these initial data we have the following scalings of the star $A$ 's stress-energy tensor components $T_A^{\mu \nu}$ in the body zone coordinates: $T_A^{\tau \tau} = \mathcal{O}(\epsilon^{-2})$ , $T_A^{\tau i} = \mathcal{O}(\epsilon^{-4})$ , $T_A^{ij} = \mathcal{O}(\epsilon^{-8})$ . Here the underlined indices mean that for any tensor $A^i$ , $A^i = \epsilon^{-2}A^i$ . In [94], we have transformed $T_N^{\mu \nu}$ , the components of the stress-energy tensor of the matter in the near zone coordinates, to $T_A^{\mu \nu}$ using the transformation from the near zone coordinates to the (generalized) Fermi normal coordinates at 1 PN order [13]. It is difficult, however, to construct the (generalized) Fermi normal coordinates at an high post-Newtonian order. Therefore we shall not use it. We simply assume that for $T_N^{\mu \nu}$ (or rather $\Lambda_N^{\mu \nu}$ , the source term of the relaxed Einstein equations; see Equation (63)),

$$
T _ {N} ^ {\tau \tau} = \mathcal {O} (\epsilon^ {- 2}), \tag {45}
$$

$$
T _ {N} ^ {\tau \underline {{i}}} = \mathcal {O} (\epsilon^ {- 4}), \tag {46}
$$

$$
T _ {N} ^ {\underline {{i j}}} = \mathcal {O} (\epsilon^ {- 8}), \tag {47}
$$

as their leading scalings.

As for the field variables on the initial hypersurface, we simply assume that the field is of 2.5 PN order except for the field determined by the constraint equations. Note that the radiation reaction effect to the stars first appears at the 2.5 PN order. Futamase showed that even if one takes the field of order 1 PN, initial value of the field does not affect the subsequent motion of the system up to 2.5 PN order [80]. Thus, we expect that the initial value of the field does not affect the orbital motion of the system up to 3 PN order, though a detailed calculation has not been done yet.

It is worth noticing that the initial value formulation has some advantages. First, by using the initial value formulation one can avoid the famous runaway solution problem in a radiation reaction problem. Second, one can construct an initial condition on some spacelike hypersurface rather than at past null infinity. Putting an initial condition for the field in the past null infinity requires a prior knowledge about the spacetime, which is obtained through the time evolution of the field from the initial condition. The initial value formulation can give in a sense a realistic initial condition. In our universe there may be no past null infinity because of the big bang.

An interesting initial condition is the statistical initial condition $[138]$ . Here the binary system is in the background gravitational radiation bath for which we know only its statistical properties. For example, the phase of the radiation is assumed to be random and irrelevant to the motion of the binary. The origins of the radiation are cosmological, or related to the evolution of the system before the initial hypersurface. Then we can evaluate the expected time evolution of the binary system by letting the system evolve from a set of possible initial conditions and taking a statistical ensemble average over the initial conditions.

# 3.4 Newtonian equations of motion for extended bodies

Before ending this section, we present some equations for Newtonian extended bodies (stars). These equations will give a useful guideline when we develop our formalism.

The basic equations are the equation of continuity, the Euler equation, and the Poisson equation, respectively:

$$
\frac {\partial \rho}{\partial \tau} + \frac {\partial \rho v ^ {i}}{\partial x ^ {i}} = 0, \tag {48}
$$

$$
\rho \frac {D v ^ {i}}{D \tau} = \rho \frac {\partial \Phi}{\partial x _ {i}}, \tag {49}
$$

$$
\Delta \Phi = 4 \pi \rho . \tag {50}
$$

We define the mass, the dipole moment, the quadrupole moment, and the momentum of the star $A$ as

$$
m _ {A} \equiv \int_ {B _ {A}} d ^ {3} x \rho , \tag {51}
$$

$$
D _ {A} ^ {i} \equiv \int_ {B _ {A}} d ^ {3} x \rho (x ^ {i} - z _ {A} ^ {i}), \tag {52}
$$

$$
I _ {A} ^ {i j} \equiv \int_ {B _ {A}} d ^ {3} x \rho (x ^ {i} - z _ {A} ^ {i}) (x ^ {j} - z _ {A} ^ {j}), \tag {53}
$$

$$
P _ {A} ^ {i} \equiv \int_ {B _ {A}} d ^ {3} x \rho v ^ {i}. \tag {54}
$$

Here $z_{A}^{i}$ is a representative point of the star A. The time derivative of the mass vanishes. Setting the time derivative of the dipole moment to zero gives the velocity momentum relation and a definition of the center of mass,

$$
\frac {d D ^ {i}}{d \tau} = P _ {A} ^ {i} - m _ {A} v _ {A} ^ {i} = 0, \tag {55}
$$

where $v_{A}^{i} = dz_{A}^{i} / d\tau$ . Using the velocity momentum relation, we calculate the time derivative of the momentum,

$$
m _ {1} \frac {d ^ {2} z _ {1} ^ {i}}{d \tau^ {2}} = F _ {1} ^ {i}, \tag {56}
$$

where $F_{1}^{i}$ is defined by Equation (43). The Newtonian potential can be expressed by the mass and multipole moment as

$$
\Phi = \sum_ {A = 1, 2} \left[ \frac {m _ {A}}{| \vec {x} - \vec {z} _ {A} |} + \frac {1}{2} I _ {A} ^ {i j} \frac {\partial^ {2}}{\partial x ^ {i} \partial x ^ {j}} \left(\frac {1}{| \vec {x} - \vec {z} _ {A} |}\right) \right]. \tag {57}
$$

Substituting $\Phi$ into Equation (56), we obtain the equations of motion,

$$
m _ {1} \frac {d ^ {2} z _ {1} ^ {i}}{d \tau^ {2}} = \frac {\partial}{\partial z _ {1} ^ {i}} \frac {m _ {1} m _ {2}}{| \vec {z} _ {1} - \vec {z} _ {2} |} + \frac {1}{2} (I _ {1} ^ {i j} + I _ {2} ^ {i j}) \frac {\partial^ {3}}{\partial z _ {1} ^ {i} \partial z _ {2} ^ {j} \partial z _ {2} ^ {k}} \left(\frac {1}{| \vec {z} _ {1} - \vec {z} _ {2} |}\right). \tag {58}
$$

Here we ignored the mass multipole moments of the stars that are of higher order than the quadrupole moments.

Actually, it is straightforward to formally include all the Newtonian mass multipole moments in the surface integral approach,

$$
m _ {1} \frac {d ^ {2} z _ {1} ^ {i}}{d \tau^ {2}} = \sum_ {p = 0} \sum_ {q = 0} \frac {(- 1) ^ {p + 1} (2 p + 2 q + 1) ! !}{p ! q !} \frac {I _ {1} ^ {\langle M _ {p} \rangle} I _ {2} ^ {\langle N _ {q} \rangle}}{r _ {1 2} ^ {p + q + 2}} N ^ {\langle i M _ {p} N _ {q} \rangle}, \tag {59}
$$

where $\vec{r}_{12} = \vec{z}_{1} - \vec{z}_{2}$ , $N^{i} = r_{12}^{i}/r_{12}$ , $M_{p} = m_{1}m_{2}\ldots m_{p}$ is a corrective index, $\langle\ldots\rangle$ denotes the symmetric-tracefree operation on the indices between the brackets, and $I_{A}^{M_{p}}$ are the Newtonian mass multipole moments of order 2p.

# 4 Formulation

Following the basic idea explained in the previous Section 3, we develop now our formalism for the derivation of the post-Newtonian equations of motion suitable for relativistic compact binaries. See also [94, 95].

# 4.1 Field equations

As discussed in the previous Section 3, we express our equations of motion in terms of surface integrals over the body zone boundary where it is assumed that the metric slightly deviates from the flat metric $\eta^{\mu \nu} = \mathrm{diag}\left(-\epsilon^{2},1,1,1\right)$ (in the near zone coordinates $(\tau ,x^{i}))$ . Thus we define a deviation field $h^{\mu \nu}$ as

$$
h ^ {\mu \nu} \equiv \eta^ {\mu \nu} - \sqrt {- g} g ^ {\mu \nu}, \tag {60}
$$

where $g$ is the determinant of the metric. Our $h^{\mu \nu}$ differs from the corresponding field in [30] in a sign. Indices are raised or lowered by the flat (auxiliary) metric $\eta^{\mu \nu}$ unless otherwise stated.

Now we impose the harmonic coordinate condition on the metric,

$$
h _ {, \nu} ^ {\mu \nu} = 0, \tag {61}
$$

where the comma denotes the partial derivative. In the harmonic gauge, we can recast the Einstein equations into the relaxed form,

$$
\Box h ^ {\mu \nu} = - 1 6 \pi \Lambda^ {\mu \nu}, \tag {62}
$$

where $\square = \eta^{\mu \nu}\partial_{\mu}\partial_{\nu}$ is the flat d'Alembertian and

$$
\Lambda^ {\mu \nu} \equiv \Theta^ {\mu \nu} + \chi_ {, \alpha \beta} ^ {\mu \nu \alpha \beta}, \tag {63}
$$

$$
\Theta^ {\mu \nu} \equiv (- g) (T ^ {\mu \nu} + t _ {\mathrm{LL}} ^ {\mu \nu}), \tag {64}
$$

$$
\chi^ {\mu \nu \alpha \beta} \equiv \frac {1}{1 6 \pi} (h ^ {\alpha \nu} h ^ {\beta \mu} - h ^ {\alpha \beta} h ^ {\mu \nu}). \tag {65}
$$

Here, $T^{\mu\nu}$ and $t_{LL}^{\mu\nu}$ denote the stress-energy tensor of the stars and the Landau–Lifshitz pseudotensor [115]. The explicit form of $t_{LL}^{\mu\nu}$ in the harmonic gauge is

$$
\begin{array}{l} (- 1 6 \pi g) t _ {\mathrm{LL}} ^ {\mu \nu} = g _ {\alpha \beta} g ^ {\gamma \delta} h _ {, \gamma} ^ {\mu \alpha} h _ {, \delta} ^ {\nu \beta} + \frac {1}{2} g ^ {\mu \nu} g _ {\alpha \beta} h _ {, \delta} ^ {\alpha \gamma} h _ {, \gamma} ^ {\beta \delta} - 2 g _ {\alpha \beta} g ^ {\gamma (\mu} h _ {, \delta} ^ {\nu) \alpha} h _ {, \gamma} ^ {\delta \beta} \\ + \frac {1}{2} \left(g ^ {\mu \alpha} g ^ {\nu \beta} - \frac {1}{2} g ^ {\mu \nu} g ^ {\alpha \beta}\right) \left(g _ {\gamma \delta} g _ {\epsilon \zeta} - \frac {1}{2} g _ {\gamma \epsilon} g _ {\delta \zeta}\right) h _ {, \alpha} ^ {\gamma \epsilon} h _ {, \beta} ^ {\delta \zeta}. \tag {66} \\ \end{array}
$$

$\chi^{\mu \nu \alpha \beta}$ originates from our use of the flat d'Alembertian instead of the curved space d'Alembertian. In consistency with the harmonic condition, the conservation law is expressed as

$$
\Lambda_ {, \nu} ^ {\mu \nu} = 0. \tag {67}
$$

Note that the divergence of $\chi^{\mu \nu \alpha \beta}$ itself vanishes identically due to the symmetry of its indices.

Now we rewrite the relaxed Einstein equations into an integral form,

$$
h ^ {\mu \nu} (\tau , x ^ {i}) = 4 \int_ {C (\tau , x ^ {k})} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |} + h _ {\mathrm{H}} ^ {\mu \nu} (\tau , x ^ {i}), \tag {68}
$$

where $C(\tau, x^{k})$ means the past light cone emanating from the event $(\tau, x^{k})$ . $C(\tau, x^{k})$ is truncated at the $\tau = 0$ hypersurface. $h_{H}^{\mu\nu}$ is a homogeneous solution of the homogeneous wave equation in the

flat spacetime, $\Box h_{\mathrm{H}}^{\mu \nu} = 0$ . $h_{\mathrm{H}}^{\mu \nu}$ evolves from a given initial data on the $\tau = 0$ initial hypersurface. The explicit form of $h_{\mathrm{H}}^{\mu \nu}$ is available via the Poisson formula (see e.g. [144])

$$
h _ {\mathrm{H}} ^ {\mu \nu} (\tau , x ^ {i}) = \frac {\tau}{4 \pi} \oint_ {\partial C (\tau , x ^ {i})} h _ {, \tau} ^ {\mu \nu} (\tau = 0, y ^ {i}) d \Omega_ {y} + \frac {1}{4 \pi} \frac {\partial}{\partial \tau} \left[ \tau \oint_ {\partial C (\tau , x ^ {i})} h ^ {\mu \nu} (\tau = 0, y ^ {i}) d \Omega_ {y} \right]. (6 9)
$$

We solve the Einstein equations as follows. First we split the integral region into two zones: the near zone and the far zone.

The near zone is the region containing the gravitational wave source where the wave character of the gravitational radiation is not manifest. In other words, in the near zone the retardation effect on the field is negligible. The near zone covers the whole source system. The size of the near zone is about a little larger than one wave length of the gravitational wave emitted by the source. In this paper we take the near zone as a sphere centered at some fixed point and enclosing the binary system. The radius of this sphere is set to be $R/\epsilon$ , where R is arbitrary but larger than the size of the binary and the wave length of the gravitational radiation. The scaling of the near zone radius is derived from the $\epsilon$ dependence of the wavelength of the gravitational radiation emitted due to the orbital motion of the binary. Note that roughly speaking the frequency of such a wave is about twice the Keplerian frequency of the binary. The center of the near zone sphere would be determined, if necessary, for example, to be the center of mass of the near zone. The outside of the near zone is the far zone where the retardation effect of the field is crucial.

For the near zone field point $P_{N}$ , we write the field as

$$
h ^ {\mu \nu} (\mathrm{P} _ {\mathrm{N}} \in N) = h _ {P _ {N} (N)} ^ {\mu \nu} + h _ {P _ {N} (F)} ^ {\mu \nu} + h _ {\mathrm{H}} ^ {\mu \nu}, \tag {70}
$$

$$
h _ {P _ {N} (N)} ^ {\mu \nu} \equiv 4 \int_ {N = \{y: | y | \leq \mathcal {R} / \epsilon \}} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |}, \tag {71}
$$

$$
h _ {P _ {N} (F)} ^ {\mu \nu} \equiv 4 \int_ {F = \{y: | y | > \mathcal {R} / \epsilon \}} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |}. \tag {72}
$$

$h_{P_N(N)}^{\mu \nu}$ is the near zone integral contribution to the near zone field, and $h_{P_N(F)}^{\mu \nu}$ is the far zone integral contribution to the near zone field.

The far zone contribution can be evaluated with the DIRE method developed in $[129]$ . An explicit calculation shows that apparently there are the far zone contributions to the near zone field at 3 PN order. However, these 3 PN contributions are merely a gauge. Pati and Will showed that the far zone contribution does not affect the equations of motion up to 3 PN order inclusively and that the far zone contribution first appears at 4 PN order. This result is consistent with the earlier result of Blanchet and Damour $[20]$ who used the multipolar-post-Minkowskian formalism. We follow the DIRE method and check that the far zone contribution does not affect the equations of motion up to 3 PN order in Appendix A. Henceforth we shall focus our attention on the near zone contribution $h_{P_{N}(N)}^{\mu\nu}$ and do not write down the far zone contribution in the following calculation of the field.

As for the homogeneous solution, we shall ignore it for simplicity. If we take random initial data for the field $[138]$ supposed to be of 1 PN order $[79]$ , they are irrelevant to the dynamics of the binary system up to the radiation reaction order $[79]$ . As we have assumed in the previous Section 3.3 that the magnitude of the free data of the gravitational field on the initial hypersurface is 2.5 PN order, we expect that the homogeneous solution does not affect the equations of motion up to 3 PN order. We leave a full implementation of the initial value formulation on the field as future work.

It is worth noticing that when we let $\tau$ become sufficiently large, then the condition $h_{\mathrm{H}}^{\mu \nu}(\tau ,x^{i}) = 0$ corresponds to a no-incoming radiation condition at (Minkowskian) past null infinity (see e.g. [77]).

Equation (69) can be written down as Kirchhoff's formula,

$$
h _ {\mathrm{H}} ^ {\mu \nu} (\tau , x ^ {i}) = \oint_ {\partial C (\tau , x ^ {i})} \frac {d \Omega_ {y}}{4 \pi} \left[ \frac {\partial}{\partial \rho} (\rho h ^ {\mu \nu} (\tau^ {\prime}, y ^ {i})) + \frac {\partial}{\partial \tau^ {\prime}} (\rho h ^ {\mu \nu} (\tau^ {\prime}, y ^ {i})) \right] \bigg | _ {\tau^ {\prime} = 0, \rho = | \vec {x} - \vec {y} | = \tau}. (7 3)
$$

Then the no-incoming radiation condition at (Minkowskian) past null-infinity,

$$
\lim _ {\substack {\tau = r,\\r \rightarrow \infty}} \left[ \frac {\partial}{\partial r} (r h ^ {\mu \nu}) + \frac {\partial}{\partial \tau} (r h ^ {\mu \nu}) \right] = 0, \tag{74}
$$

is a sufficient condition to $h_{\mathrm{H}}^{\mu \nu}(\tau ,x^{i}) = 0$ when $\tau$ goes to infinity.

Now we shall devote ourselves to the evaluation of the near zone contribution to the near zone field,

$$
h ^ {\mu \nu} (\tau , x ^ {i}) = 4 \int_ {N (\tau , x ^ {k})} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |}. \tag {75}
$$

Henceforth we shall omit the subscript $P_N(N)$ of the field $h^{\mu \nu}$ for notational simplicity.

# 4.2 Near zone contribution

We shall evaluate the near zone contribution as follows. First, we make a retarded expansion of Equation (75) and change the integral region to a $\tau = constant$ spatial hypersurface,

$$
h ^ {\mu \nu} = 4 \sum_ {n = 0} \frac {(- \epsilon) ^ {n}}{n !} \left(\frac {\partial}{\partial \tau}\right) ^ {n} \int_ {N} d ^ {3} y | \vec {x} - \vec {y} | ^ {n - 1} \Lambda_ {N} ^ {\mu \nu} (\tau , y ^ {k}; \epsilon). \tag {76}
$$

Note that the above integral depends on the arbitrary length R in general. The cancellation between the R dependent terms in the far zone contribution and those in the near zone contribution was shown by [129] through all the post-Newtonian order. In the following, we shall omit the terms which have negative powers of $\mathcal{R}\left(\mathcal{R}^{-k}; k > 0\right)$ . In other words, we simply let $R \to 0$ whenever it gives a convergent result. On the other hand, we shall retain terms having positive powers of $\mathcal{R}\left(\mathcal{R}^{k}; k > 0\right)$ , and logarithmic terms ( $\ln R$ ) to confirm that the final result, in the end of calculation, is independent of R (and for logarithmic terms to keep the arguments of logarithm non-dimensional).

Second we split the integral into two parts: a contribution from the body zone $B_A$ , and from elsewhere, $N / B$ . Schematically we evaluate the following two types of integrals (we omit indices of the field),

$$
h = h _ {B} + h _ {N / B},
$$

$$
h _ {B} = \epsilon^ {6} \sum_ {A = 1, 2} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \frac {f (\tau , \vec {z} _ {A} + \epsilon^ {2} \vec {\alpha} _ {A})}{| \vec {r} _ {A} - \epsilon^ {2} \vec {\alpha} _ {A} | ^ {1 - n}}, \tag {77}
$$

$$
h _ {N / B} = \int_ {N / B} d ^ {3} y \frac {f (\tau , \vec {y})}{| \vec {x} - \vec {y} | ^ {1 - n}},
$$

where $\vec{r}_A \equiv \vec{x} - \vec{z}_A$ . We shall deal with these two contributions successively.

# 4.2.1 Body zone contribution

As for the body zone contribution, we make a multipole expansion using the scaling of the integrand, i.e. $\Lambda^{\mu\nu}$ in the body zone. For example, the n=0 part in Equation (76), $h_{B\ n=0}^{\mu\nu}$ , gives

$$
h _ {B n = 0} ^ {\tau \tau} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \left(\frac {P _ {A} ^ {\tau}}{r _ {A}} + \epsilon^ {2} \frac {D _ {A} ^ {k} r _ {A} ^ {k}}{r _ {A} ^ {3}} + \epsilon^ {4} \frac {3 I _ {A} ^ {\langle k l \rangle} r _ {A} ^ {k} r _ {A} ^ {l}}{2 r _ {A} ^ {5}} + \epsilon^ {6} \frac {5 I _ {A} ^ {\langle k l m \rangle} r _ {A} ^ {k} r _ {A} ^ {l} r _ {A} ^ {m}}{2 r _ {A} ^ {7}}\right) + \mathcal {O} (\epsilon^ {1 2}), \tag {78}
$$

$$
h _ {B n = 0} ^ {\tau i} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \left(\frac {P _ {A} ^ {i}}{r _ {A}} + \epsilon^ {2} \frac {J _ {A} ^ {k i} r _ {A} ^ {k}}{r _ {A} ^ {3}} + \epsilon^ {4} \frac {3 J _ {A} ^ {k l i} r _ {A} ^ {k} r _ {A} ^ {l}}{2 r _ {A} ^ {5}}\right) + \mathcal {O} (\epsilon^ {1 0}), \tag {79}
$$

$$
h _ {B n = 0} ^ {i j} = 4 \epsilon^ {2} \sum_ {A = 1, 2} \left(\frac {Z _ {A} ^ {i j}}{r _ {A}} + \epsilon^ {2} \frac {Z _ {A} ^ {k i j} r _ {A} ^ {k}}{r _ {A} ^ {3}} + \epsilon^ {4} \frac {3 Z _ {A} ^ {\langle k l \rangle i j} r _ {A} ^ {k} r _ {A} ^ {l}}{2 r _ {A} ^ {5}} + \epsilon^ {6} \frac {5 Z _ {A} ^ {\langle k l m \rangle i j} r _ {A} ^ {k} r _ {A} ^ {l} r _ {A} ^ {m}}{2 r _ {A} ^ {7}}\right) + \mathcal {O} (\epsilon^ {1 0}). (8 0)
$$

Here the operator $\langle \ldots \rangle$ denotes a symmetric and tracefree (STF) operation on the indices in the brackets. See [129, 150, 165] for some useful formulas of STF tensors. Also we define $r_A \equiv |\vec{r}_A|$ . To derive the 3 PN equations of motion, $h^{\tau \tau}$ up to $\mathcal{O}(\epsilon^{10})$ and $h^{\tau i}$ as well as $h^{ij}$ up to $\mathcal{O}(\epsilon^8)$ are required.

In the above equations we define the multipole moments as

$$
I _ {A} ^ {K _ {l}} \equiv \epsilon^ {2} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \Lambda_ {N} ^ {\tau \tau} \alpha_ {A} ^ {\underline {{{K}}} _ {l}}, \tag {81}
$$

$$
J _ {A} ^ {K _ {l} i} \equiv \epsilon^ {4} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \Lambda_ {N} ^ {\tau_ {i}} \alpha_ {A} ^ {\underline {{K}} _ {l}}, \tag {82}
$$

$$
Z _ {A} ^ {K _ {l} i j} \equiv \epsilon^ {8} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \Lambda_ {\overline {{N}}} ^ {\underline {{i j}}} \alpha_ {\overline {{A}}} ^ {\underline {{K}} _ {l}}, \tag {83}
$$

where we introduced a collective multi-index $I_{l} \equiv i_{1}i_{2} \ldots i_{l}$ and $\alpha_{A}^{I_{l}} \equiv \alpha_{A}^{i_{1}}\alpha_{A}^{i_{2}} \ldots \alpha_{A}^{i_{l}}$ . Then $P_{A}^{\tau} \equiv I_{A}^{K_{0}}$ , $D_{A}^{k_{1}} \equiv I_{A}^{K_{1}}$ , and $P_{A}^{k_{1}} \equiv J_{A}^{K_{1}}$ . We simply call $P_{A}^{\mu}$ the four-momentum of the star A, $P_{A}^{i}$ the momentum, and $P_{A}^{\tau}$ the energy $^{7}$ . Also we call $D_{A}^{k}$ the dipole moment of the star A, and $I_{A}^{kl}$ the quadrupole moment of the star A.

Then we transform these moments into more convenient forms. By the conservation law (67), we have

$$
\Lambda_ {N} ^ {\tau i} = \left(\Lambda_ {N} ^ {\tau \tau} y _ {A} ^ {i}\right) _ {, \tau} + \left(\Lambda_ {N} ^ {\tau j} y _ {A} ^ {i}\right) _ {, j} + v _ {A} ^ {i} \Lambda_ {N} ^ {\tau \tau}, \tag {84}
$$

$$
\Lambda_ {N} ^ {i j} = (\Lambda_ {N} ^ {\tau (i} y _ {A} ^ {j)}), _ {\tau} + (\Lambda_ {N} ^ {k (i} y _ {A} ^ {j)}), _ {k} + v _ {A} ^ {(i} \Lambda_ {N} ^ {j) \tau}, \tag {85}
$$

where $v_{A}^{i} \equiv \dot{z}_{A}^{i}$ , an overdot denotes a time derivative with respect to $\tau$ , and $\vec{y}_{A} \equiv \vec{y} - \vec{z}_{A}$ . Using these equations and noticing that the body zone remains unchanged (in the near zone coordinates), i.e. $\dot{R}_{A} = 0$ , we have

$$
P _ {A} ^ {i} = P _ {A} ^ {\tau} v _ {A} ^ {i} + Q _ {A} ^ {i} + \epsilon^ {2} \frac {d D _ {A} ^ {i}}{d \tau}, \tag {86}
$$

$$
J _ {A} ^ {i j} = \frac {1}{2} \left(M _ {A} ^ {i j} + \epsilon^ {2} \frac {d I _ {A} ^ {i j}}{d \tau}\right) + v _ {A} ^ {(i} D _ {A} ^ {j)} + \frac {1}{2} \epsilon^ {- 2} Q _ {A} ^ {i j}, \tag {87}
$$

$$
Z _ {A} ^ {i j} = \epsilon^ {2} P _ {A} ^ {\tau} v _ {A} ^ {i} v _ {A} ^ {j} + \frac {1}{2} \epsilon^ {6} \frac {d ^ {2} I _ {A} ^ {i j}}{d \tau^ {2}} + 2 \epsilon^ {4} v _ {A} ^ {(i} \frac {d D _ {A} ^ {j)}}{d \tau} + \epsilon^ {4} \frac {d v _ {A} ^ {(i}}{d \tau} D _ {A} ^ {j)} + \epsilon^ {2} Q _ {A} ^ {(i} v _ {A} ^ {j)} + \epsilon^ {2} R _ {A} ^ {(i j)} + \frac {1}{2} \epsilon^ {2} \frac {d Q _ {A} ^ {i j}}{d \tau}, \tag {88}
$$

$$
Z _ {A} ^ {k i j} = \frac {3}{2} A _ {A} ^ {k i j} - A _ {A} ^ {(i j) k}, \tag {89}
$$

where

$$
M _ {A} ^ {i j} \equiv 2 \epsilon^ {4} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \alpha_ {A} ^ {[ i} \Lambda_ {N} ^ {\underline {{j}} ] \tau}, \tag {90}
$$

$$
Q _ {A} ^ {K _ {l} i} \equiv \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \left(\Lambda_ {N} ^ {\tau k} - v _ {A} ^ {k} \Lambda_ {N} ^ {\tau \tau}\right) y _ {A} ^ {K _ {l}} y _ {A} ^ {i}, \tag {91}
$$

$$
R _ {A} ^ {K _ {l} i j} \equiv \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \left(\Lambda_ {N} ^ {k j} - v _ {A} ^ {k} \Lambda_ {N} ^ {\tau j}\right) y _ {A} ^ {K _ {l}} y _ {A} ^ {i}, (9 2)
$$

and

$$
A _ {A} ^ {k i j} \equiv \epsilon^ {2} J _ {A} ^ {k (i} v _ {A} ^ {j)} + \epsilon^ {2} v _ {A} ^ {k} J _ {A} ^ {(i j)} + R _ {A} ^ {k (i j)} + \epsilon^ {4} \frac {d J _ {A} ^ {k (i j)}}{d \tau}. \tag {93}
$$

The operators [ ] and ( ) attached to the indices denote anti-symmetrization and symmetrization. $M_{A}^{ij}$ is the spin of the star A and Equation (86) is the momentum-velocity relation. Thus our momentum-velocity relation is a direct analog of the Newtonian momentum-velocity relation (see Section 3.4). In general, we have

$$
J _ {A} ^ {K _ {l} i} = J _ {A} ^ {(K _ {l} i)} + \frac {2 l}{l + 1} J _ {A} ^ {(K _ {l - 1} [ k _ {l}) i ]}, \tag {94}
$$

$$
Z _ {A} ^ {K _ {l} i j} = \frac {1}{2} \left[ Z _ {A} ^ {(K _ {l} i) j} + \frac {2 l}{l + 1} Z _ {A} ^ {(K _ {l - 1} [ k _ {l}) i ] j} + Z _ {A} ^ {(K _ {l} j) i} + \frac {2 l}{l + 1} Z _ {A} ^ {(K _ {l - 1} [ k _ {l}) j ] i} \right], \tag {95}
$$

and

$$
J _ {A} ^ {(K _ {l} i)} = \frac {1}{l + 1} \epsilon^ {2} \frac {d I _ {A} ^ {K _ {l} i}}{d \tau} + v _ {A} ^ {(i} I _ {A} ^ {K _ {l})} + \frac {1}{l + 1} \epsilon^ {- 2 l} Q _ {A} ^ {K _ {l} i}, \tag {96}
$$

$$
Z _ {A} ^ {(K _ {l} i) j} + Z _ {A} ^ {(K _ {l} j) i} = \epsilon^ {2} v _ {A} ^ {(i} J _ {A} ^ {K _ {l}) j} + \epsilon^ {2} v _ {A} ^ {(j} J _ {A} ^ {K _ {l}) i} + \frac {2}{l + 1} \epsilon^ {4} \frac {d J _ {A} ^ {K _ {l} (i j)}}{d \tau} + \epsilon^ {- 2 l + 2} R _ {A} ^ {K _ {l} (i j)}, \tag {97}
$$

where $l$ is the number of indices in the multi-index $K_{l}$ .

Now, from the above equations, especially Equation (88), we find that the body zone contributions, $h_{B n=0}^{\mu\nu}$ , are of order $\mathcal{O}(\epsilon^{4})$ . Note that if we can not or do not assume the (nearly) stationarity of the initial data for the stars, then, instead of Equation (88) we have

$$
Z _ {A} ^ {i j} = \epsilon^ {2} P _ {A} ^ {\tau} v _ {A} ^ {i} v _ {A} ^ {j} + \frac {1}{2} \frac {d ^ {2} I _ {A} ^ {i j}}{d \eta^ {2}} + \dots , \tag {98}
$$

where we used the dynamical time $\eta$ (see Section 3.3). In this case the lowest order metric differs from the Newtonian form. From our (almost) stationarity assumption the remaining motion inside a star, apart from the spinning motion, is caused only by the tidal effect by the companion star and from Equation (88); it appears at 3 PN order [81].

To obtain the lowest order $h_{B_{n=0}}^{\mu\nu}$ , we have to evaluate the surface integrals $Q_{A}^{K_{l}i}, R_{A}^{K_{l}ij}$ . Generally, in $h_{B_{n=0}}^{\mu\nu}$ the moments $J_{A}^{K_{l}i}$ and $Z_{A}^{K_{l}ij}$ appear formally at the order $\epsilon^{2l+4}$ and $\epsilon^{2l+2}$ . Thus $Q_{A}^{K_{l}i}$ and $R_{A}^{K_{l}ij}$ appear as

$$
h _ {B n = 0} ^ {\tau i} \sim \dots + \epsilon^ {4} \frac {r _ {A} ^ {K _ {l}} Q _ {A} ^ {\langle K _ {l} \rangle i}}{r _ {A} ^ {2 l + 1}} + \dots , \tag {99}
$$

$$
h _ {B n = 0} ^ {i j} \sim \dots + \epsilon^ {4} \frac {r _ {A} ^ {K _ {l}} R _ {A} ^ {\langle K _ {l} \rangle (i j)}}{r _ {A} ^ {2 l + 1}} + \dots , \tag {100}
$$

where we omitted irrelevant terms and numerical coefficients. Thus one may expect that $Q_A^{K_l i}$ and $R_A^{K_l ij}$ appear at the order $\epsilon^4$ for any $l$ and we have to calculate an infinite number of moments. In fact, this is not the case and it was shown in [91] that only $l = 0,1$ terms of $R_A^{K_l ij}$ contribute to the 3 PN equations of motion. The important thing here is that $\epsilon^4 Q_A^{K_l i}$ and $\epsilon^4 R_A^{K_l ij}$ are at most $\mathcal{O}(\epsilon^4)$ in $h_{B\,n=0}^{\mu\nu}$ .

Finally, since the order of $h_{Bn}^{\mu \nu}(n\geq 1)$ is higher than that of $h_{Bn = 0}^{\mu \nu}$ , we conclude that $h_B^{\mu \nu} = \mathcal{O}(\epsilon^4)$ .

# 4.2.2 $N / B$ contribution

As far as the N/B contribution is concerned, since the integrand $\Lambda_{N}^{\mu\nu} = -gt_{LL}^{\mu\nu} + \chi^{\mu\nu\alpha\beta},_{\alpha\beta}$ is at least quadratic in the small deviation field $h^{\mu\nu}$ , we make the post-Newtonian expansion in the integrand. Then, basically, with the help of (super-)potentials $g(\vec{x})$ which satisfy $\Delta g(\vec{x}) = f(\vec{x})$ , $\Delta$ denoting the Laplacian, we have for each integral (see e.g. the n = 0 term in Equation (77))

$$
\int_ {N / B} d ^ {3} y \frac {f (\vec {y})}{| \vec {x} - \vec {y} |} = - 4 \pi g (\vec {x}) + \oint_ {\partial (N / B)} d S _ {k} \left[ \frac {1}{| \vec {x} - \vec {y} |} \frac {\partial g (\vec {y})}{\partial y ^ {k}} - g (\vec {y}) \frac {\partial}{\partial y ^ {k}} \left(\frac {1}{| \vec {x} - \vec {y} |}\right) \right]. \tag {101}
$$

Equation (101) can be derived without using Dirac delta distributions (see Appendix B of [91]). For the $n \geq 1$ terms in Equation (77), we use potentials many times to convert all the volume integrals into surface integrals and “ $-4\pi g(\vec{x})$ ” terms $^{8}$ .

In fact finding the super-potentials is one of the most formidable tasks especially when we proceed to high post-Newtonian orders. Fortunately, up to 2.5 PN order, all the required super-potentials are available. At 3 PN order, there appear many integrands for which we could not find the required super-potentials. To obtain the 3 PN equations of motion, we devise an alternative method similar to the method employed by Blanchet and Faye [28]. The details of the method will be explained later.

Now the lowest order integrands can be evaluated with the body zone contribution $h_B^{\mu \nu}$ , and since $h_B^{\mu \nu}$ is $\mathcal{O}(\epsilon^4)$ , we find

$$
\Lambda_ {N} ^ {\tau \tau} = \mathcal {O} (\epsilon^ {6}), \tag {102}
$$

$$
\Lambda_ {N} ^ {\tau i} = \mathcal {O} (\epsilon^ {6}), \tag {103}
$$

$$
\Lambda_ {N} ^ {i j} = (- g) t _ {\mathrm{LL}} ^ {i j} + \mathcal {O} (\epsilon^ {8}) = \epsilon^ {4} \frac {1}{6 4 \pi} \left(\delta_ {k} ^ {i} \delta_ {l} ^ {j} - \frac {1}{2} \delta^ {i j} \delta_ {k l}\right) _ {4} h _ {B} ^ {\tau \tau , k} _ {4} h _ {B} ^ {\tau \tau , l} + \mathcal {O} (\epsilon^ {5}), \tag {104}
$$

where we expanded $h_{B}^{\mu\nu}$ in an $\epsilon$ series,

$$
h _ {B} ^ {\mu \nu} = \sum_ {n = 0} \epsilon^ {4 + n} _ {n} h _ {B} ^ {\mu \nu}. \tag {105}
$$

Similarly, in the following we expand $h^{\mu\nu}$ in an $\epsilon$ series. From these equations we find that the deviation field in N/B, $h^{\mu\nu}$ , is $\mathcal{O}(\epsilon^{4})$ . (It should be noted that in the body zone $h^{\mu\nu}$ is assumed to be of order unity and within our method we can not calculate $h^{\mu\nu}$ there explicitly. To obtain $h^{\mu\nu}$ in the body zone, we have to know the internal structure of the star.)

# 4.3 Lorentz contraction and multipole moments

In Equations (81, 82, 83), we have defined multipole moments of a star. The definition of those multipole moments are operational ones and are not necessarily equal to “intrinsic” multipole moments of the star. This is clear, for example, if we remember that a moving ball has spurious multipole moments due to Lorentz contraction.

We have adopted the generalized Fermi normal coordinates (GFC) [13] as the star's reference coordinates to address this problem. A question specific to our formalism is whether the differences between the multipole moments defined in Equations (81, 82, 83) and the multipole moments in GFC give purely monopole terms. If so, we of course have to take into account such terms in the field to compute the equations of motion for two intrinsically spherical star binaries. This problem

is addressed in the Appendix C of $[91]$ and the differences are mainly attributed to the shape of the body zone. The body zone $B_{A}$ which is spherical in the near zone coordinates (NZC) is not spherical in the GFC mainly because of a kinematic effect (Lorentz contraction). To derive the 3 PN equations of motion, it turns out that it is sufficient to compute the difference in the STF quadrupole moment up to 1 PN order. The result is

$$
\begin{array}{l} \delta I _ {A} ^ {\langle i j \rangle} \equiv I _ {A, \mathrm{NZC}} ^ {\langle i j \rangle} - I _ {A, \mathrm{GFC}} ^ {\langle i j \rangle} \\ = \epsilon^ {- 6} \frac {1}{2} v _ {A} ^ {k} v _ {A} ^ {l} \oint_ {\partial B _ {A}} d S _ {k} y _ {A} ^ {l} y _ {A} ^ {\langle i} y _ {A} ^ {j \rangle} \Lambda^ {\tau \tau} + \mathcal {O} (\epsilon^ {3}) \\ = - \epsilon^ {2} \frac {4 m _ {A} ^ {3}}{5} v _ {A} ^ {\langle i} v _ {A} ^ {j \rangle} + \mathcal {O} (\epsilon^ {3}), \tag {106} \\ \end{array}
$$

where $I_{A,\mathrm{NZC}}^{ij} \equiv I_A^{ij}$ . $I_{A,\mathrm{GFC}}^{ij}$ are the quadrupole moments defined in the generalized Fermi normal coordinates. Note that the difference $\delta I_A^{\langle ij\rangle}$ is expressed in a surface integral form.

As is obvious from Equation (106), this difference appears even if the companion star does not exist. We note that we could derive the 3 PN metric for an isolated star A moving at a constant velocity using our method explained in this section by simply letting the mass of the companion star be zero. Actually, the $\delta I_{A}^{\langle ij\rangle}$ above is a necessary term which makes the so-obtained 3 PN metric equal to the Schwarzschild metric boosted at the constant velocity $\vec{v}_{A}$ in harmonic coordinates.

The reason why the influence of a body's Lorentz contraction appears starting from 3 PN order is as follows. The body's Lorentz contraction appears as an apparent deformation of the body, namely, apparent quadrupole moment as the leading order in the frame where the body is moving. The body's (real) quadrupole affects the field from 2 PN order through (see Equation (78))

$$
h _ {\mathrm{quad}} ^ {\tau \tau} = \epsilon^ {8} \sum_ {A} \frac {3 I _ {A} ^ {\langle i j \rangle} r _ {A} ^ {i} r _ {A} ^ {j}}{2 r _ {A} ^ {5}}. \tag {107}
$$

The radius of a compact body A is of order of its mass $m_{A}$ , and the Lorentz contraction deforms the body so that the radius of the body changes by the amount $\epsilon^{2}m_{A}v_{A}^{2} + \mathcal{O}(\epsilon^{4})$ . So the apparent quadrupole moments due to Lorentz contraction is of order of

$$
I _ {A \mathrm{apparent}} ^ {\langle i j \rangle} \sim \epsilon^ {2} m _ {A} ^ {3} v _ {A} ^ {2}. (1 0 8)
$$

This field is the 3 PN field and thus the Lorentz contraction of a body affects the equations of motion starting from 3 PN order.

# 4.4 General form of the equations of motion

From the definition of the four-momentum,

$$
P _ {A} ^ {\mu} (\tau) = \epsilon^ {2} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \Lambda_ {N} ^ {\tau \mu}, \tag {109}
$$

and the conservation law (67), we have the evolution equations for the four-momentum:

$$
\frac {d P _ {A} ^ {\mu}}{d \tau} = - \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {k \mu} + \epsilon^ {- 4} v _ {A} ^ {k} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {\tau \mu}. \tag {110}
$$

Here we used the fact that the size and the shape of the body zone are defined to be fixed (in the near zone coordinates), while the center of the body zone moves at the velocity of the star's representative point.

Substituting the momentum-velocity relation (86) into the spatial components of Equation (110), we obtain the general form of equations of motion for star A,

$$
\begin{array}{l} P _ {A} ^ {\tau} \frac {d v _ {A} ^ {i}}{d \tau} = - \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {k i} + \epsilon^ {- 4} v _ {A} ^ {k} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {\tau i} \\ + \epsilon^ {- 4} v _ {A} ^ {i} \left(\oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {k \tau} - v _ {A} ^ {k} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {\tau \tau}\right) \\ - \frac {d Q _ {A} ^ {i}}{d \tau} - \epsilon^ {2} \frac {d ^ {2} D _ {A} ^ {i}}{d \tau^ {2}}. \tag {111} \\ \end{array}
$$

All the right hand side terms in Equation (111) except the dipole moment are expressed as surface integrals. We can specify the value of $D_A^i$ freely to determine the representative point $z_A^i(\tau)$ of star $A$ . Up to 2.5 PN order we take $D_A^i = 0$ and simply call $z_A^i$ the center of mass of star $A$ . Note that in order to obtain the spin-orbit coupling force in the same form as in previous works [46, 108, 152], we have to make another choice for $z_A^i$ (see [94] and Appendix B.1). At 3 PN order, yet another choice of the value of the dipole moment $D_A^i$ shall be examined (see Sections 7 and 8.2).

In Equation (111), $P_A^\tau$ rather than the mass of star $A$ appears. Hence we have to derive a relation between the mass and $P_A^\tau$ . We shall derive that relation by solving the temporal component of the evolution equations (110) functionally.

Then, since all the equations are expressed with surface integrals except $D_{A}^{i}$ to be specified, we can derive the equations of motion for a strongly self-gravitating star using the post-Newtonian approximation.

# 4.5 On the arbitrary constant $R_{A}$

Since we have introduced the body zone by hand, the arbitrary body zone size $R_{A}$ seems to appear in the metric, the multipole moments of the stars, and the equations of motion. More specifically $R_{A}$ appears in them because of (i) the splitting of the deviation field into two parts (i.e. B and N/B contributions), the definition of the moments, and (ii) the surface integrals that we evaluate to derive the equations of motion.

# 4.5.1 $R_{A}$ dependence of the field

B and N/B contributions to the field depend on the body zone boundary $\epsilon R_{A}$ . But $h^{\mu\nu}$ itself does not depend on $\epsilon R_{A}$ . Thus it is natural to expect that there are renormalized multipole moments which are independent of $R_{A}$ since we use nonsingular matter sources. This renormalization would absorb the $\epsilon R_{A}$ dependence occurring in the computation of the N/B field (see Section 4.8 for an example of such a renormalization). One possible practical obstacle for this expectation might be the $\ln(\epsilon R_{A})$ dependence of multipole moments. Although at 3 PN order there appear such logarithmic terms, it is found that we could remove them by rechoosing the value of the dipole moment $D_{A}^{i}$ of the star.

Though we use the same symbol for the moments henceforth as before for notational simplicity, it should be understood that they are the renormalized ones. For instance, we use the symbol “ $P_{A}^{\mu}$ ” for the renormalized $P_{A}^{\mu}$ .

# 4.5.2 $R_{A}$ dependence of the equations of motion

Since we compute integrals over the body zone boundary, in general the resulting equations of motion seem to depend on the size of the body zone boundary, $\epsilon R_{A}$ . Actually this is not the case.

In the derivation of Equation (111), if we did not use the conservation law (67) until the final step, we have

$$
\begin{array}{l} P _ {A} ^ {\tau} \frac {d v _ {A} ^ {i}}{d \tau} + \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {k i} - \epsilon^ {- 4} v _ {A} ^ {k} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {\tau i} - \epsilon^ {- 4} v _ {A} ^ {i} \left(\oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {k \tau} - v _ {A} ^ {k} \oint_ {\partial B _ {A}} d S _ {k} \Lambda_ {N} ^ {\tau \tau}\right) + \\ \frac {d Q _ {A} ^ {i}}{d \tau} + \epsilon^ {2} \frac {d ^ {2} D _ {A} ^ {i}}{d \tau^ {2}} \\ = \epsilon^ {- 4} \int_ {B _ {A}} d ^ {3} y \Lambda_ {N, \nu} ^ {i \nu} - \epsilon^ {- 4} v _ {A} ^ {i} \int_ {B _ {A}} d ^ {3} y \Lambda_ {N, \nu} ^ {\tau \nu} + \epsilon^ {- 4} \frac {d}{d \tau} \left(\int_ {B _ {A}} d ^ {3} y \Lambda_ {N, \nu} ^ {\tau \nu} y _ {A} ^ {i}\right). \tag {112} \\ \end{array}
$$

Now the conservation law is satisfied for whatever value we take for $R_A$ , then the right hand side of the above equation is zero for any $R_A$ . Hence the equations of motion (111) do not depend on $R_A$ (a similar argument can be found in [73]).

Along the same line, the momentum-velocity relation (86) does not depend on $R_{A}$ .

In Section 4.8 we shall explicitly show the irrelevance of the field and the equations of motion to $R_{A}$ by checking the cancellation among the $R_{A}$ dependent terms up to 0.5 PN order.

# 4.6 Newtonian equations of motion

We first derive the Newtonian equations of motion and the 1 PN correction to the acceleration. The derivation of the 1 PN equations of motion includes some essences of our formalism and shows how it works properly. Thus we shall give a detailed explanation about the derivation, though the calculation is elementary and the resulting equations of motion are well-known.

Now let us derive the Newtonian mass-energy relation first. From Equations (102, 103) and the time component of Equation (110), we have

$$
\frac {d P _ {A} ^ {\tau}}{d \tau} = \mathcal {O} (\epsilon^ {2}). \tag {113}
$$

Then we define the mass of star $A$ as the integrating constant of this equation,

$$
m _ {A} \equiv \lim _ {\epsilon \rightarrow 0} P _ {A} ^ {\tau}. \tag {114}
$$

here $m_{A}$ is the ADM mass that star A had if it were isolated. (We took $\epsilon$ zero limit in Equation (114) to ensure that the mass defined above does not include the effect of the companion star and the orbital motion of the star itself. By this limit we ensure that this mass is the integrating constant of Equation (113).) By definition $m_{A}$ is constant. Then we obtain the lowest order $h^{\tau\tau}$ :

$$
{ } _ { 4 } h ^ { \tau \tau } = { } _ { 4 } h _ { B } ^ { \tau \tau } = 4 \sum _ { A = 1 , 2 } \frac { m _ { A } } { r _ { A } } . \tag {115}
$$

Second, from Equation (91) with Equations (102) and (103) we obtain $Q_A^i = 0$ at the lowest order. Thus we have the Newtonian momentum-velocity relation $P_A^i = m_A v_A^i + \mathcal{O}(\epsilon^2)$ from Equation (86) (we set $D_A^i = 0$ ).

Substituting Equation (104) with the lowest order $h^{\tau\tau}$ into the general form of equations of motion (111) and using the Newtonian momentum-velocity relation, we have for star 1:

$$
\begin{array}{l} m _ {1} \frac {d v _ {1} ^ {i}}{d \tau} = - \oint_ {\partial B _ {1}} d S _ {k (4)} [ (- g) t _ {\mathrm{LL}} ^ {i k} ] \\ = - \frac {1}{4 \pi} \left(\delta_ {l} ^ {i} \delta_ {m} ^ {k} - \frac {1}{2} \delta^ {i k} \delta_ {l m}\right) \sum_ {A = 1, 2} \sum_ {B = 1, 2} \oint_ {\partial B _ {1}} d S _ {k} \frac {m _ {A} m _ {B} y _ {A} ^ {l} y _ {B} ^ {m}}{y _ {A} ^ {3} y _ {B} ^ {3}}, \\ \end{array}
$$

![](images/970b62b3b1a7417eb665310e3228c049461881e9559be06fedbc55eb8bc140ea.jpg)

<details>
<summary>text_image</summary>

B₂
z₂
r₁₂
y₂ = r₁₂ + εR₁n₁
y₁ = εR₁n₁
z₁ B₁
</details>

Figure 3: The vectors used in the surface integral over the boundary of the body zone 1.

$$
\begin{array}{l} = - \frac {1}{4 \pi} \left(\delta_ {l} ^ {i} \delta_ {m} ^ {k} - \frac {1}{2} \delta^ {i k} \delta_ {l m}\right) \oint d \Omega_ {\mathbf {n _ {1}}} n _ {1} ^ {k} \bigg (\frac {m _ {1} ^ {2} n _ {1} ^ {l} n _ {1} ^ {m}}{(\epsilon R _ {1}) ^ {2}} + \frac {2 m _ {1} m _ {2} n _ {A} ^ {(l)} (r _ {1 2} ^ {m} + \epsilon R _ {1} n _ {1} ^ {m})}{| \vec {r} _ {1 2} + \epsilon R _ {1} \vec {n} _ {1} | ^ {3}} \\ \left. + \frac {(\epsilon R _ {1}) ^ {2} m _ {2} ^ {2} (r _ {1 2} ^ {l} + \epsilon R _ {1} n _ {1} ^ {l}) (r _ {1 2} ^ {m} + \epsilon R _ {1} n _ {1} ^ {m})}{| \vec {r} _ {1 2} + \epsilon R _ {1} \vec {n} _ {1} | ^ {3}}\right), \tag {116} \\ \end{array}
$$

where $y_{A}^{i}$ is the integral variable and $\vec{y}_{A} = \vec{y} - \vec{z}_{A}$ . We defined $\vec{n}_{1}$ as the spatial unit vector $^{9}$ emanating from $\vec{z}_{1}$ , $\vec{r}_{12} \equiv \vec{z}_{1} - \vec{z}_{2}$ , and $n_{12}^{i} \equiv r_{12}^{i}/r_{12}$ . We used $\vec{r}_{1} = \epsilon R_{1}\vec{n}_{1}$ and $\vec{r}_{2} = \vec{r}_{12} + \epsilon R_{1}\vec{n}_{1}$ . For details see Figure 3).

In the above equation, by virtue of the angular integral the first term (which is singular when the $\epsilon$ zero limit is taken) vanishes. The third term vanishes by letting $\epsilon$ go to zero. Only the second term survives and gives the Newtonian equations of motion as expected. This completes the Newtonian order calculations.

# 4.7 First post-Newtonian equations of motion

Next, at 1 PN order, we need ${}_6h^{\tau \tau}$ and ${}_4h^{\mu \nu}$ . The $n = 1$ term in the retardation expansion series of $h^{\tau \tau}$ , Equation (76), gives no contribution at the 1 PN order by the constancy of the mass $m_A$ , i.e. ${}_5h^{\tau \tau} = 0$ .

Now we obtain $_{4}h^{\tau i}$ from the Newtonian momentum-velocity relation:

$$
_ 4 h ^ {\tau i} = 4 \sum_ {A = 1, 2} \frac {m _ {A} v _ {A} ^ {i}}{r _ {A}}. \tag {117}
$$

We evaluate the surface integrals in the evolution equation for $P_A^\tau$ at 1 PN order. The result for star 1 is

$$
\frac {d P _ {1} ^ {\tau}}{d \tau} = \epsilon^ {2} m _ {1} \frac {d}{d \tau} \left(\frac {1}{2} v _ {1} ^ {2} + \frac {3 m _ {2}}{r _ {1 2}}\right) + \mathcal {O} (\epsilon^ {3}), \tag {118}
$$

where we used the Newtonian equations of motion. From this equation we have the mass-energy relation at 1 PN order,

$$
P _ {1} ^ {\tau} = m _ {1} \left[ 1 + \epsilon^ {2} \left(\frac {1}{2} v _ {1} ^ {2} + \frac {3 m _ {2}}{r _ {1 2}}\right) \right] + \mathcal {O} (\epsilon^ {3}). \tag {119}
$$

Then we have to calculate the 1 PN order $Q_A^i$ . The result is $Q_A^i = \epsilon^2 m_A^2 v_A^i / (6\epsilon R_A)$ . As $Q_A^i$ depends on $R_A$ , we ignore it (see Section 4.5). As a result we obtain the momentum-velocity relation at 1 PN order, $P_A^i = P_A^\tau v_A^i + \mathcal{O}(\epsilon^3)$ from Equation (86).

Now as for $_{4}h^{ij}$ , we first calculate the surface integrals $Q_{A}^{ij}$ , $R_{A}^{ij}$ , and $R_{A}^{kij}$ from Equations (91) and (92). We then find that they depend on $R_{A}$ , hence we ignore them and obtain

$$
h _ {B} ^ {i j} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {m _ {A} v _ {A} ^ {i} v _ {A} ^ {j}}{r _ {A}} + \mathcal {O} (\epsilon^ {5}). \tag {120}
$$

To derive ${}_6h^{\tau \tau}$ and ${}_4h^{ij}$ , we have to evaluate non-compact support integrals for ${}_6h_{N / B}^{\tau \tau}$ and ${}_4h_{N / B}^{ij}$ , and the $n = 2$ term in Equation (76) for $h^{\tau \tau}$ :

$$
h ^ {\tau \tau} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {P _ {A} ^ {\tau}}{r _ {A}} + 4 \epsilon^ {6} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} _ {6} [ (- g) t _ {\mathrm{LL}} ^ {\tau \tau} ] + 2 \epsilon^ {6} \frac {\partial}{\partial \tau^ {2}} \sum_ {A = 1, 2} P _ {A} ^ {\tau} r _ {A}, \tag {121}
$$

$$
h ^ {i j} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {m _ {A} v _ {A} ^ {i} v _ {A} ^ {j}}{r _ {A}} + 4 \epsilon^ {4} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} 4 [ (- g) t _ {\mathrm{LL}} ^ {i j} ], \tag {122}
$$

with

$$
{ } _ { 6 } [ ( - 1 6 \pi g ) t _ { \mathrm{LL} } ^ { \tau \tau } ] = - \frac { 7 } { 8 } { } _ { 4 } h ^ { \tau \tau , k } { } _ { 4 } h ^ { \tau \tau , k } , \tag {123}
$$

$$
{ } _ { 4 } [ ( - 1 6 \pi g ) t _ { \mathrm{LL} } ^ { i j } ] = \frac { 1 } { 4 } \left( \delta _ { k } ^ { i } \delta _ { k } ^ { i } - \frac { 1 } { 2 } \delta ^ { i j } \delta _ { k l } \right) { } _ { 4 } h ^ { \tau \tau , k } { } _ { 4 } h ^ { \tau \tau , l } . ( 1 2 4 )
$$

The evaluation of the twice retardation expansion term (the last term in Equation (121)) is straightforward. For the non-compact support integrals, it is sufficient to consider the following integral:

$$
\int_ {N / B} \frac {d ^ {3} y}{1 6 \pi | \vec {x} - \vec {y} |} _ {4} h ^ {\tau \tau , i} _ {4} h ^ {\tau \tau , j} = \frac {1}{\pi} \sum_ {A = 1, 2} P _ {A} ^ {\tau 2} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {r _ {A} ^ {i} r _ {A} ^ {j}}{r _ {A} ^ {6}} + \frac {2}{\pi} P _ {1} ^ {\tau} P _ {2} ^ {\tau} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {r _ {1} ^ {(i} r _ {2} ^ {j)}}{r _ {1} ^ {3} r _ {2} ^ {3}}. \tag {125}
$$

To evaluate the above Poisson integrals, we use Equation (101), thus we need to find superpotentials for the integrands. For this purpose, it is convenient to transform the tensorial integrands into scalar integrands with differentiation operators,

$$
\frac {r _ {A} ^ {i} r _ {A} ^ {j}}{r _ {A} ^ {6}} = \frac {1}{8} \left(\delta_ {k} ^ {i} \delta_ {l} ^ {j} + \delta^ {i j} \delta_ {k l}\right) \frac {\partial^ {2}}{\partial y ^ {k} y ^ {l}} \left(\frac {1}{r _ {A} ^ {2}}\right), \tag {126}
$$

$$
\frac {r _ {1} ^ {i} r _ {2} ^ {j}}{r _ {1} ^ {3} r _ {2} ^ {3}} = \frac {\partial}{\partial z _ {1} ^ {i}} \frac {\partial}{\partial z _ {2} ^ {i}} \left(\frac {1}{r _ {1} r _ {2}}\right). \tag {127}
$$

Then it is relatively easy to find the super-potentials for these scalars. The results are $\Delta\ln r_{1}=1/r_{1}^{2}$ and $\Delta\ln S=1/(r_{1}r_{2})$ where $S=r_{1}+r_{2}+r_{12}$ [77]. Equation (101) for each integrand then becomes

$$
\int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {r _ {A} ^ {i} r _ {A} ^ {j}}{r _ {A} ^ {6}} = - 4 \pi \left(\frac {\delta^ {i j}}{4 r _ {A} ^ {2}} - \frac {r _ {A} ^ {i} r _ {A} ^ {j}}{4 r _ {A} ^ {4}}\right) + \frac {4 \pi \delta^ {i j}}{3 \epsilon R _ {1} r _ {1}} + \mathcal {O} (\epsilon R _ {A}), \tag {128}
$$

$$
\begin{array}{l} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {r _ {1} ^ {i} r _ {2} ^ {j}}{r _ {1} ^ {3} r _ {2} ^ {3}} = - 4 \pi \left(- \frac {\delta^ {i j}}{r _ {1 2} S} - \frac {r _ {1} ^ {i} r _ {1 2} ^ {j}}{r _ {1} r _ {1 2} S ^ {2}} + \frac {r _ {1 2} ^ {i} r _ {1 2} ^ {j}}{r _ {1 2} ^ {2} S ^ {2}} + \frac {r _ {1 2} ^ {i} r _ {1 2} ^ {j}}{r _ {1 2} ^ {3} S} - \frac {r _ {1} ^ {i} r _ {1 2} ^ {j}}{r _ {1} r _ {2} S ^ {2}} + \frac {r _ {1 2} ^ {i} r _ {2} ^ {j}}{r _ {1 2} r _ {2} S ^ {2}}\right) \\ + \mathcal {O} (\epsilon R _ {A}). \tag {129} \\ \end{array}
$$

The second last term of Equation (128) and the terms abbreviated as $\mathcal{O}(\epsilon R_{A})$ in the above two equations arise from the surface integrals in Equation (101). Since they depend on $\epsilon R_{A}$ , we ignore

it (see Section 4.8). Substituting Equations (128) and (129) back into Equation (125), we can compute the non-compact support integrals.

Using the above results and the Newtonian equations of motion for the twice retardation expansion term, we finally obtain

$$
\begin{array}{l} h ^ {\tau \tau} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {P _ {A} ^ {\tau}}{r _ {A}} \\ + \epsilon^ {6} \left[ - 2 \sum_ {A = 1, 2} \frac {m _ {A}}{r _ {A}} \{(\vec {n} _ {A} \cdot \vec {v} _ {A}) ^ {2} - v _ {A} ^ {2} \} + 2 \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \vec {n} _ {1 2} \cdot (\vec {n} _ {1} - \vec {n} _ {2}) \right. \\ \left. + 7 \sum_ {A = 1, 2} \frac {m _ {A} ^ {2}}{r _ {A} ^ {2}} + 1 4 \frac {m _ {1} m _ {2}}{r _ {1} r _ {2}} - 1 4 \frac {m _ {1} m _ {2}}{r _ {1 2}} \sum_ {A = 1, 2} \frac {1}{r _ {A}} \right] + \mathcal {O} (\epsilon^ {7}), \tag {130} \\ \end{array}
$$

$$
\begin{array}{l} h ^ {i j} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {m _ {A} v _ {A} ^ {i} v _ {A} ^ {j}}{r _ {A}} \\ + \epsilon^ {4} \left[ \sum_ {A = 1, 2} \frac {m _ {A} ^ {2}}{r _ {A} ^ {2}} n _ {A} ^ {i} n _ {A} ^ {j} - \frac {8 m _ {1} m _ {2}}{r _ {1 2} S} n _ {1 2} ^ {i} n _ {1 2} ^ {j} \right. \\ \left. - 8 \left(\delta_ {k} ^ {i} \delta_ {l} ^ {j} - \frac {1}{2} \delta^ {i j} \delta_ {k l}\right) \frac {m _ {1} m _ {2}}{S ^ {2}} (\vec {n} _ {1 2} - \vec {n} _ {1}) ^ {(k} (\vec {n} _ {1 2} + \vec {n} _ {2}) ^ {l)} \right] + \mathcal {O} (\epsilon^ {5}), \tag {131} \\ \end{array}
$$

where $\vec{n}_A\equiv \vec{r}_A / r_A$

Evaluating the surface integrals in Equation (111) as in the Newtonian case, we obtain the 1 PN equations of motion,

$$
\begin{array}{l} m _ {1} \frac {d v _ {1} ^ {i}}{d \tau} = - \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \\ + \epsilon^ {2} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \Big [ n _ {1 2} ^ {i} \left(- v _ {1} ^ {2} - 2 v _ {2} ^ {2} + \frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {5 m _ {1}}{r _ {1 2}} + \frac {4 m _ {2}}{r _ {1 2}}\right) \\ \left. + V ^ {i} \left(4 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) - 3 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right)\right) \right], \tag {132} \\ \end{array}
$$

where we defined the relative velocity as $\vec{V} \equiv \vec{v}_1 - \vec{v}_2$ and we used the Newtonian equations of motion as well as Equation (119).

Finally let us give a summary of our procedure (see Figure 4). With the $n$ PN order equations of motion and $h^{\mu \nu}$ in hand, we first derive the $n + 1$ PN evolution equation for $P_A^\tau$ . Then we solve it functionally and obtain the mass-energy relation at $n + 1$ PN order. Next we calculate $Q_A^i$ at $n + 1$ PN order and derive the momentum-velocity relation at $n + 1$ PN order. Then we calculate $Q_A^{K_l i}$ and $R_A^{K_l ij}$ . With the $n + 1$ PN mass-energy relation, the $n + 1$ PN momentum-velocity relation, $Q_A^{K_l i}$ , and $R_A^{K_l ij}$ , we next derive the $n + 1$ PN deviation field $h^{\mu \nu}$ . Finally we evaluate the surface integrals which appear in the right hand side of Equation (111) and obtain the $n + 1$ PN equations of motion. In the above calculations we use the $n$ PN order equations of motion to reduce the order of the equations of motion whenever an acceleration appears in the right hand of the resulting equations of motion. For instance, when we meet $\epsilon^2 dv_1^i /d\tau$ in the right hand side of the equations motion and we have to evaluate this up to $\epsilon^2$ , then using the Newtonian equations of motion, we replace it by $-\epsilon^2 m_2r_{12}^i /r_{12}^3$ . Basically we shall derive the 3 PN equations of motion with the procedure as described above.

# 4.8 Body zone boundary dependent terms

As explained in Section 4.5 we discard the body zone boundary dependent terms in the field, since we expect that they cancel out between the body zone contribution and the N/B contribution. Before moving on to the higher order calculations, however, it is instructive to see that such a cancellation really occurs in the field and consequently the equations of motion up to 0.5 PN order.

First, we show the independence of the body zone radius $R_{A}$ in the 0.5 PN field. Returning back to the derivation of the 1 PN $h^{\tau\tau}$ with care for the $R_{A}$ dependence (see Equation (128)), we get

$$
\begin{array}{l} h ^ {\tau \tau} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {P _ {A} ^ {\tau}}{r _ {A}} + 4 \epsilon^ {6} \int_ {C / B} d ^ {3} y \frac {{} _ {6} \Lambda_ {N} ^ {\tau \tau} (\tau , y ^ {k})}{| \vec {x} - \vec {y} |} + 2 \epsilon^ {6} \frac {\partial^ {2}}{\partial \tau^ {2}} \sum_ {A = 1, 2} P _ {A} ^ {\tau} r _ {A} + \mathcal {O} (\epsilon^ {7}) \\ = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {1}{r _ {A}} \left(P _ {A} ^ {\tau} - \epsilon^ {2} \frac {7 m _ {A} ^ {2}}{2 \epsilon R _ {A}}\right) + \mathcal {O} (\epsilon^ {6}). \tag {133} \\ \end{array}
$$

Now we split $P_{A}^{\tau}$ as $P_{A}^{\tau} = \bar{P}_{A}^{\tau} + \tilde{P}_{A}^{\tau}$ where $\bar{P}_{A}^{\tau}$ does not depend on $R_{A}$ while $\tilde{P}_{A}^{\tau}$ does. In order to evaluate the $R_{A}$ dependent part of $P_{A}^{\tau}$ , $\tilde{P}_{A}^{\tau}$ , we use the fact that the integral of $\Lambda_{N}^{\tau\tau}$ over the near zone does not depend on the size of the body zone. (Notice that $P_{A}^{\tau}$ is defined as the volume integral of $\Lambda_{N}^{\tau\tau}$ over $B_{A}$ .) In fact, from the relevant expression of the pseudotensor, we obtain at the lowest order

$$
\begin{array}{l} \epsilon^ {2} \int_ {N / B} d ^ {3} \alpha \epsilon_ {6} ^ {6} \Lambda_ {N} ^ {\tau \tau} = \epsilon^ {2} \int_ {N / B} d ^ {3} y _ {6} \Lambda_ {N} ^ {\tau \tau} \\ = - \epsilon^ {2} \frac {7}{2} \sum_ {A = 1, 2} \frac {m _ {A} ^ {2}}{\epsilon R _ {A}} \tag {134} \\ + (\text { terms   independent   of } R _ {A}, \text { or   terms   having   positive   power(s)   of } R _ {A}). \\ \end{array}
$$

Hence we find $\tilde{P}_{A}^{\tau}=\epsilon^{2}7m_{A}^{2}/(2\epsilon R_{A})+\mathcal{O}(\epsilon^{2})$ . The above equation shows us that the 0.5 PN field is independent of $R_{A}$ and fully expressed by the $R_{A}$ independent energy $\bar{P}_{A}^{\tau}$ , or mass up to 0.5 PN order.

In the similar manner we split $P_{A}^{i}$ as $P_{A}^{i} = \bar{P}_{A}^{i} + \tilde{P}_{A}^{i}$ and obtain $\tilde{P}_{A}^{i} = \epsilon^{2}11m_{A}^{2}v_{A}^{i}/(3\epsilon R_{A}) + \mathcal{O}(\epsilon^{2})$ from the definition of $P_{A}^{i}$ . Thus from the fact that $Q_{A}^{i} = \epsilon^{2}m_{A}^{2}v_{A}^{i}/(6\epsilon R_{A}) + \mathcal{O}(\epsilon^{2})$ , we find that the 0.5 PN momentum-velocity relation does not depend on $R_{A}$ : $\bar{P}_{A}^{i} = \bar{P}_{A}^{\tau}v_{A}^{i} + \mathcal{O}(\epsilon^{2})$ . Finally evaluating the surface integrals in the general form of equations of motion using the “renormalized” (barred) moments, we find that the equations of motion are independent of $R_{A}$ as was expected.

As remarked in Section 4.5 from now on we use the same symbol for the renormalized moments as for the “bare” moments henceforth as before for notational simplicity.

![](images/69641e77d206b352768be754ceed01d04c79caa82267c8873f696b7de8b6e5db.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["n PN field: 2n+4h^ττ, 2n+2h^μi, 4h^ττ = 4 Σ_{A=1,2} m_A/r_A,<br>    n PN mass-energy relation: P_A^τ = m_A + m_A Σ_{k=1}^{2n} ε_k^k Γ_A,<br>    n PN momentum-velocity relation: P_A^i = P_A^τ v_A^i + Σ_{k=1}^{2n} ε_k^k Q_A^i,<br>    n PN equations of motion: m_A^i d v_A^i/dτ = -m_1 m_2 r_{12}^i / r_{12}^3 + Σ_{k=1}^{2n} ε_k^k F_A^i."] --> B["n + 1 PN mass-energy relation:<br>(dP_A^τ/dτ)_{n+1} = ε^{-4} ∫_{∂B_A} dS_k[-2n+4Λ_N^τk + v_A^k 2n+4Λ_N^ττ"],
    P_A^τ = m_A + m_A Σ_{k=1}^{2n+2} ε_k^k Γ_A.]
    B --> C["n + 1 PN Q_A^i integral: (Q_A^i)_{n+1} = ∫_{∂B_A} dS_k[2n+6Λ_N^τk - v_A^k 2n+6Λ_N^ττ"],
    Define z_A^i up to n+1 PN order by, for instance, setting D_A^i = 0.
    n + 1 PN momentum-velocity relation: P_A^i = P_A^τ v_A^i + Σ_{k=1}^{2n+2} ε_k^k Q_A^i.]
    C --> D["n + 1 PN Q_A^{K_l i} and R_A^{K_l i j} integrals,<br>    n + 1 PN N/B contribution:<br>2n+6h^ττ_{N/B} = Σ_{k=0}^{2n}(-ε)^k/k! ∫_{N/B} d^3 y|x - y|^k-1_{2n-k+6}(k) Λ^ττ(τ,y^i),<br>2n+4h^μi_{N/B} = Σ_{k=0}^{2n}(-ε)^k/k! ∫_{N/B} d^3 y|x - y|^k-1_{2n-k+4}(k) Λ^μi(τ,y^i)<br>    (n + 1 PN far zone contribution, if necessary)."]
    D --> E["n + 1 PN field: 2n+6h^ττ, 2n+4h^μi."]
    E --> F["Evolution equation for P_A^i: (dP_A^i/dτ)_{n+1} = - ∫_{∂B_A} dS_k 2n+6Λ^ki + v_A^k ∫_{∂B_A} dS_k 2n+6Λ^τi,<br>    n + 1 PN equations of motion: m_A^i dv_A^i/dτ = -m_1 m_2 r_{12}^i / r_{12}^3 + Σ_{k=1}^{2n+2} ε_k^k F_A^i."]
```
</details>

Figure 4: Flowchart of the post-Newtonian iteration.

# 5 Third Post-Newtonian Gravitational Field

In the post-Newtonian approximation, we need to solve Poisson equations to find the metric. Up to 2.5 PN order, explicit forms of the metric have been obtained in [30]. However, it seems impossible to derive the 3 PN accurate gravitational field in harmonic coordinates in a closed form completely. The problem is that it seems difficult (if at all possible) to find a particular solution of the Poisson equations for non-compact sources. The works so far overcome this problem by not solving the Poisson equations but keeping the Poisson integral unevaluated. Then to derive the equations of motion, we basically interchange the order of operations; first we evaluate surface integrals with the Poisson integrals as integrands (in the surface integral approach [91]) or compute derivatives of the Poisson integrals (when one adopts the geodesic equation [27]), and then we evaluate the remaining volume integrals. We first explain the usual method and a method to derive a field just around the star, and then explain the method mentioned above.

# 5.1 Super-potential method

Up to 2.5 PN order, we have solved all the Poisson equations necessary to derive the 2.5 PN gravitational field. At 3 PN order, we have found a part of the solutions of the Poisson equations, which we call (super-)potentials $^{10}$ . For example,

$$
\frac {r _ {1} ^ {i} r _ {1} ^ {j} r _ {1} ^ {k} r _ {2} ^ {l}}{r _ {1} ^ {5} r _ {2} ^ {3}} = \Delta \left[ - \frac {1}{3} \frac {\partial^ {4}}{\partial z _ {1} ^ {i} \partial z _ {1} ^ {j} \partial z _ {1} ^ {k} \partial z _ {2} ^ {l}} f ^ {(1, - 1)} + \frac {1}{3} (\delta^ {i j} \partial_ {z _ {1} ^ {k}} + \delta^ {i k} \partial_ {z _ {1} ^ {j}} + \delta^ {j k} \partial_ {z _ {1} ^ {i}}) \partial_ {z _ {2} ^ {l}} \ln S \right], (1 3 5)
$$

where $S = r_1 + r_2 + r_{12}$ , and $f^{(1, - 1)}$ which satisfies $\Delta f^{(1, - 1)} = r_1 / r_2$ is given in [99] as

$$
f ^ {(1, - 1)} = \frac {1}{1 8} (- r _ {1} ^ {2} - 3 r _ {1} r _ {1 2} - r _ {1 2} ^ {2} + 3 r _ {1} r _ {2} + 3 r _ {1 2} r _ {2} + r _ {2} ^ {2}) + \frac {1}{6} (- r _ {1} ^ {2} + r _ {1 2} ^ {2} + r _ {2} ^ {2}) \ln S. \tag {136}
$$

It is possible to add any homogeneous solution to super-potentials. In our formalism, the only place where we use super-potentials is Equation (101). In the case above, we could add, say, $1/r_{1}$ to $f^{(1,-1)}$ . (Note that to evaluate the surface integrals in the general form of equations of motion (111) we need super-potentials in the spatial region N/B which do not include any singularity due to the point particle limit.) It is easy to see that contribution from a possible additional homogeneous solution cancels out between the “ $-4\pi g(\vec{x})$ ” term and the surface integral in Equation (101).

Useful super-potentials for derivation of the 3 PN field are given in [30, 27, 91, 99].

# 5.2 Super-potential-in-series method

As all what we need to do is to evaluate the surface integrals in the general form of equations of motion (111), we need an expression for the gravitational field only around the star. In fact, we have developed such a method in [91] for the source term of the following form

$$
\partial_ {z _ {A} ^ {i}} \partial_ {z _ {A ^ {\prime}} ^ {j}} g (\vec {x}) \equiv \partial_ {z _ {A} ^ {i}} \partial_ {z _ {A ^ {\prime}} ^ {j}} \left(\frac {(\ln r _ {1}) ^ {p} (\ln r _ {2}) ^ {q}}{r _ {1} ^ {a} r _ {2} ^ {b}}\right), \tag {137}
$$

where $a$ and $b$ are integers and $p = 0,1$ , $q = 0,1$ . Note that $A, A' = 1,2$ . Then, we take spatial derivatives out of the Poisson integral,

$$
\int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \partial_ {z _ {A} ^ {i}} \partial_ {z _ {A ^ {\prime}} ^ {j}} g (\vec {y}) = \partial_ {z _ {A} ^ {i}} \partial_ {z _ {A ^ {\prime}} ^ {j}} \int_ {N / B} d ^ {3} y \frac {g (\vec {y})}{| \vec {x} - \vec {y} |} + \partial_ {z _ {A} ^ {i}} \oint_ {\partial B _ {A ^ {\prime}}} d S _ {j} \frac {g (\vec {y})}{| \vec {x} - \vec {y} |} + \oint_ {\partial B _ {A}} d S _ {i} \frac {\partial_ {z _ {A ^ {\prime}} ^ {j}} g (\vec {y})}{| \vec {x} - \vec {y} |}. \tag {138}
$$

Note that the integration region is N/B and therefore $g(\vec{x})$ is nonsingular in N/B. For this kind of source term, we have given a method in [91] to find a field $F_{[A,c]}^{(m,n)}$ in the neighborhood of star A in the following sense:

$$
\Delta F _ {[ A, c ]} ^ {(p, m, q, n)} - (\ln r _ {1}) ^ {p} r _ {1} ^ {m} (\ln r _ {2}) ^ {q} r _ {2} ^ {n} = \mathcal {O} \left(\frac {r _ {A} ^ {c + 1}}{r _ {1 2} ^ {c + 1}}\right) \quad \text { as } r _ {A} \rightarrow 0. \tag {139}
$$

We have checked at 3 PN order that the resulting field from this method is equal to the field obtained from the usual (super-potential) method whenever the super-potentials are available. Unfortunately, however, this method is not perfect and we need another method to derive the equations of motion which we explain now.

# 5.3 Direct-integration method

In the surface integral approach, we need to evaluate a surface integral of the Landau–Lifshitz pseudotensor which has a form $h^{\mu\nu,\alpha}h^{\lambda\sigma,\beta}$ . From an order counting, it would be clear that we need to evaluate a surface integral of the form “Newtonian potential” × “3 PN potential” to find the 3 PN equations of motion, where it seems impossible to find the “3 PN potential” in a closed form, as mentioned above. The surface integral mentioned here thus has the form

$$
\oint_ {\partial B _ {1}} d S _ {k} \frac {r _ {A} ^ {l}}{r _ {A} ^ {3}} \operatorname{disc} _ {\epsilon R _ {A}} \int_ {N / B} d ^ {3} y \frac {f (\vec {y})}{| \vec {x} - \vec {y} |} \tag {140}
$$

for star 1. Here the operator $\mathrm{disc}_{\epsilon R_A}$ means to discard all the $\epsilon R_A$ dependent terms other than logarithms of $\epsilon R_A$ [91]. The method to evaluate this type of integral is to exchange the order of integrals, first calculate the surface integral, and then calculate the volume integral. One caveat is that we can not simply exchange the order of integrals, and we put an operation $\mathrm{disc}_{\epsilon R_A}$ in front of the Poisson integral as in Equation (140) above.

We first exchange the order of integrals,

$$
\oint_ {\partial B _ {1}} d S _ {k} \frac {r _ {A} ^ {l}}{r _ {A} ^ {3}} \operatorname{disc} _ {\epsilon R _ {A}} \int_ {N / B} d ^ {3} y \frac {f (\vec {y} _ {1})}{| \vec {r} _ {1} - \vec {y} _ {1} |} = \lim _ {r _ {1} ^ {\prime} \rightarrow \epsilon R _ {1}} \operatorname{disc} _ {\epsilon R _ {A}} \int_ {N / B} d ^ {3} y _ {1} f (\vec {y} _ {1}) \oint_ {\partial B _ {1} ^ {\prime}} d S _ {k} \frac {1}{| \vec {y} _ {1} - r _ {1} ^ {\prime} \vec {n} _ {1} |} \partial_ {z _ {A} ^ {l}} \frac {1}{r _ {A}}, \tag {141}
$$

where we defined a sphere $B_1'$ whose center is $\vec{z}_1$ and that has a radius $r_1'$ which is a constant slightly larger than $\epsilon R_1$ for any (small) $\epsilon$ ( $\epsilon R_1 < r_1' \ll r_{12}$ ).

The reason we introduced $r_{1}^{\prime}$ is as follows. Suppose that we treat an integrand for which the super-potential is available. By calculating the Poisson integral, we have a piece of field corresponding to the integrand. The piece generally depends on $\epsilon R_{A}$ , however we reasonably discard such $R_{A}$ -dependent terms (other than logarithmic dependence) as explained in Section 4.5. Using the so-obtained $R_{A}$ -independent field, we evaluate the surface integrals in the general form of the 3 PN equations of motion by discarding the $\epsilon R_{A}$ dependence emerging from the surface integrals, and obtain the equations of motion. Thus the “discarding- $\epsilon R_{A}$ ” procedure must be employed each time when the field is derived and also when equations of motion are derived, not just once. Thus $r_{1}^{\prime}$ was introduced to distinguish the two kinds of $\epsilon R_{A}$ dependence and to discard the $\epsilon R_{A}$ dependence in the right order. We show here a simple example. Let us consider the following integral:

$$
\oint_ {\partial B _ {1}} d S _ {k} \frac {r _ {1} ^ {k}}{r _ {1} ^ {3}} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {1}{y _ {1} ^ {2}}. \tag {142}
$$

Using $\Delta\ln r_{1}=1/r_{1}^{2}$ , we can integrate the Poisson integral and obtain the “field”,

$$
\int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {1}{y _ {1} ^ {2}} = - 4 \pi \ln \left(\frac {r _ {1}}{\mathcal {R} / \epsilon}\right) + 4 \pi - 4 \pi \frac {\epsilon R _ {1}}{r _ {1}} + \mathcal {O} \left((\epsilon R _ {A}) ^ {2}\right). \tag {143}
$$

Since the “body zone contribution” must have an $\epsilon R_{1}$ dependence hidden in the “moments” as $4\pi\epsilon R_{1}/r_{1}[+\mathcal{O}((\epsilon R_{A})^{2})]$ (see Section 4.5), the last term should be discarded before we evaluate the surface integral in Equation (142) using this “field”. The surface integral gives the “equations of motion”,

$$
1 6 \pi^ {2} \left(\ln \left(\frac {\mathcal {R} / \epsilon}{\epsilon R _ {1}}\right) + 1\right). \tag {144}
$$

On the other hand, we can derive the “equations of motion” by first evaluating the surface integral over $\partial B_{1}^{\prime}$ ,

$$
\begin{array}{l} \oint_ {\partial B _ {1}} d S _ {k} \frac {r _ {1} ^ {k}}{r _ {1} ^ {3}} \int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {1}{y _ {1} ^ {2}} = \int_ {N / B} \frac {d ^ {3} y}{y _ {1} ^ {2}} \oint_ {\partial B _ {1} ^ {\prime}} d S _ {k} \frac {r _ {1} ^ {k}}{r _ {1} ^ {3}} \frac {1}{| \vec {r _ {1}} - \vec {y _ {1}} |} \\ = 1 6 \pi^ {2} \left[ \int_ {\epsilon R _ {1}} ^ {r _ {1} ^ {\prime}} \frac {d y}{r _ {1} ^ {\prime}} + \int_ {r _ {1} ^ {\prime}} ^ {\mathcal {R} / \epsilon} \frac {d y}{y _ {1}} \right] \\ = 1 6 \pi^ {2} \left(\ln \left(\frac {\mathcal {R} / \epsilon}{r _ {1} ^ {\prime}}\right) + 1 - \frac {\epsilon R _ {1}}{r _ {1} ^ {\prime}}\right). \tag {145} \\ \end{array}
$$

Thus, if we take $\partial B_{1}$ as the integral region instead of $\partial B_1'$ in the first equality in Equation (145), or if we take $\lim_{r_1' \to \epsilon R_1}$ without employing $\mathrm{disc}_{\epsilon R_A}$ beforehand, we will obtain an incorrect result,

$$
1 6 \pi^ {2} \ln \left(\frac {\mathcal {R} / \epsilon}{\epsilon R _ {1}}\right), \tag {146}
$$

which disagrees with Equation (144).

With this caution in mind, it is straightforward (though tedious) to evaluate the surface integrals and then the volume integrals, and thus evaluate all the necessary integrals for our derivation of the 3 PN equations of motion.

When we have derived the 3 PN equations of motion, we have used the super-potential method whenever possible, and used a combination of the above three methods when necessary. In fact, for a computational check, we have used the direct-integration method to evaluate the contributions to the equations of motion from all of the 3 PN N/B nonretarded field, $_{8}h_{N/B n=0}^{\tau i}$ and $_{10}h_{N/B n=0}^{\tau\tau} + _{8}h_{N/B n=0^{l}}^{l}$ . As expected, we obtain the same result from two computations; the result from the direct-integration method agrees with that from the combination of the three methods: the direct-integration method, the super-potential method, and the super-potential-in-series method.

# 6 Third Post-Newtonian Mass-Energy Relation

It was found that the direct-integration part does not play any role in the evaluation of the evolution equation of $P_{A}^{\tau}$ at 3 PN order. Thus we use the same method as in the evaluation of the 2.5 PN equations of motion. Evaluating the surface integrals in Equation (110), we obtain the evolution equation of $P_{A}^{\tau}$ as

$$
\begin{array}{l} \frac {d P _ {1 \Theta} ^ {\tau}}{d \tau} = - \epsilon^ {2} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left[ 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right] \\ + \epsilon^ {4} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left[ - \frac {9}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} + \frac {1}{2} v _ {1} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} \right. \\ - 2 v _ {1} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) (\vec {n} _ {1 2} \cdot \vec {V}) + 5 v _ {2} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - 4 v _ {2} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) \\ \left. + \frac {m _ {1}}{r _ {1 2}} (- 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {1})) + \frac {m _ {2}}{r _ {1 2}} (- 1 0 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + 1 1 (\vec {n} _ {1 2} \cdot \vec {v} _ {2})) \right] \\ + \epsilon^ {6} \frac {m _ {1} m _ {2}}{r _ {1 2}} \left[ - \left(\frac {3}{2} v _ {1} ^ {4} + 2 v _ {1} ^ {2} v _ {2} ^ {2} + 4 v _ {2} ^ {4}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \left(\frac {5}{8} v _ {1} ^ {4} + \frac {3}{2} v _ {1} ^ {2} v _ {2} ^ {2} + 7 v _ {2} ^ {4}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right. \\ + \left(2 v _ {1} ^ {2} + 4 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \left(2 v _ {1} ^ {2} + 8 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ + \left(3 v _ {1} ^ {2} + 1 2 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - \left(\frac {3}{4} v _ {1} ^ {2} + 1 2 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \\ + 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} - 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ - \frac {1 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} - \frac {4 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {5} \\ + \frac {m _ {1}}{r _ {1 2}} \left\{\left(- 4 2 v _ {1} ^ {2} - \frac {1 1 7}{4} v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + 6 0 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} \right. \\ + \left(\frac {1 3 7}{4} v _ {1} ^ {2} + \frac {3 7}{2} v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + \frac {2 9 7}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {v}. \vec {v} _ {2}) \\ - \frac {2 1 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v}. \vec {v} _ {2}) - 1 5 1 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \\ \left. + 1 0 9 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - 2 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \right\} \\ + \frac {m _ {2}}{r _ {1 2}} \left\{- \left(1 3 v _ {1} ^ {2} + 1 8 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \left(1 7 v _ {1} ^ {2} + 2 5 v _ {2} ^ {2}\right) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right. \\ + 2 6 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) - 2 8 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) + 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) ^ {2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) \\ + 1 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - 2 0 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \} \\ + \frac {m _ {1} ^ {2}}{r _ {1 2} ^ {2}} \left(\frac {3 3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - \frac {1 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) - \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left(\frac {3 5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \frac {1 7}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) \\ \left. + \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} \left(- 1 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \frac {2 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) \right] \\ + \mathcal {O} (\epsilon^ {7}). \tag {147} \\ \end{array}
$$

Remarkably, we can integrate Equation (147) functionally:

$$
P _ {1 \Theta} ^ {\tau} = m _ {1} \left(1 + \epsilon_ {2} ^ {2} \Gamma_ {1} + \epsilon_ {4} ^ {4} \Gamma_ {1} + \epsilon_ {6} ^ {6} \Gamma_ {1}\right) + \mathcal {O} (\epsilon^ {7}), \tag {148}
$$

with

$$
{ } _ { 2 } \Gamma _ { 1 } = \frac { 1 } { 2 } v _ { 1 } ^ { 2 } + \frac { 3 m _ { 2 } } { r _ { 1 2 } } , \tag {149}
$$

$$
{ } _ { 4 } \Gamma _ { 1 } = - \frac { 3 m _ { 2 } } { 2 r _ { 1 2 } } ( \vec { n } _ { 1 2 } \cdot \vec { v } _ { 2 } ) ^ { 2 } + \frac { 2 m _ { 2 } } { r _ { 1 2 } } v _ { 2 } ^ { 2 } + \frac { 7 m _ { 2 } } { 2 r _ { 1 2 } } v _ { 1 } ^ { 2 } - \frac { 4 m _ { 2 } } { r _ { 1 2 } } ( \vec { v } _ { 1 } \cdot \vec { v } _ { 2 } ) + \frac { 3 } { 8 } v _ { 1 } ^ { 4 } + \frac { 7 m _ { 2 } ^ { 2 } } { 2 r _ { 1 2 } ^ { 2 } } - \frac { 5 m _ { 1 } m _ { 2 } } { 2 r _ { 1 2 } ^ { 2 } } , ( 1 5 0 )
$$

and

$$
\begin{array}{l} { } _ { 6 } \Gamma _ { 1 } = \frac { m _ { 1 } ^ { 2 } m _ { 2 } } { 2 r _ { 1 2 } ^ { 3 } } + \frac { 2 1 m _ { 1 } m _ { 2 } ^ { 2 } } { 4 r _ { 1 2 } ^ { 3 } } + \frac { 5 m _ { 2 } ^ { 3 } } { 2 r _ { 1 2 } ^ { 3 } } + \frac { 5 } { 1 6 } v _ { 1 } ^ { 6 } \\ + \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} \left(\frac {4 5}{4} v _ {1} ^ {2} + \frac {1 9}{2} v _ {2} ^ {2} + \frac {1}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - 1 9 (\vec {v} _ {1} \cdot \vec {v} _ {2}) - (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2}\right) \\ \left. + \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left(\frac {4 3}{8} v _ {1} ^ {2} + \frac {5 3}{8} v _ {2} ^ {2} - \frac {6 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - \frac {5 3}{4} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {8 5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - \frac {6 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2}\right) \right. \\ + \frac {m _ {2}}{r _ {1 2}} \left(\frac {3 3}{8} v _ {1} ^ {4} + \frac {3}{2} v _ {1} ^ {2} v _ {2} ^ {2} + v _ {2} ^ {4} - 6 v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - 4 v _ {2} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {7}{4} v _ {1} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2}\right) \\ \left. - \frac {5}{2} v _ {2} ^ {2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} + 2 \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) ^ {2} + 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) + \frac {9}{8} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {4}\right). \tag {151} \\ \end{array}
$$

Equation (148) together with Equations (149, 150, 151) gives the 3 PN order mass-energy relation.

# 6.1 Meaning of $P_{A\Theta}^{\tau}$

In this section we explain the meaning of $P_{A\Theta}^{\tau}$ . First of all, we expand in an $\epsilon$ series the four-velocity of star A normalized as $g_{\mu\nu}u_{A}^{\mu}u_{A}^{\nu} = -\epsilon^{-2}$ , where $u_{A}^{i} = u_{A}^{\tau}v_{A}^{i}$ . The result is

$$
\begin{array}{l} u _ {A} ^ {\tau} = 1 + \epsilon^ {2} \left[ \frac {1}{2} v _ {A} ^ {2} + \frac {1}{4} _ {4} h ^ {\tau \tau} \right] \\ + \epsilon^ {4} \left[ \frac {1}{4} _ {6} h ^ {\tau \tau} + \frac {1}{4} _ {4} h ^ {k} _ {k} - \frac {3}{3 2} (_ {4} h ^ {\tau \tau}) ^ {2} + \frac {5}{8} _ {4} h ^ {\tau \tau} v _ {A} ^ {2} - _ {4} h ^ {\tau} _ {k} v _ {A} ^ {k} + \frac {3}{8} v _ {A} ^ {4} \right] \\ + \epsilon^ {5} \frac {1}{4} \left[ _ {7} h ^ {\tau \tau} + _ {5} h ^ {k} _ {k} \right] \\ + \epsilon^ {6} \left[ \frac {1}{4} _ {8} h ^ {\tau \tau} + \frac {1}{4} _ {6} h ^ {k} _ {k} + \frac {1}{1 6} _ {4} h ^ {\tau \tau} _ {4} h ^ {k} _ {k} - \frac {3}{1 6} _ {4} h ^ {\tau \tau} _ {6} h ^ {\tau \tau} + \frac {7}{1 2 8} (_ {4} h ^ {\tau \tau}) ^ {3} + \frac {1}{4} _ {4} h ^ {\tau k} _ {4} h ^ {\tau} _ {k} \right. \\ - _ {6} h ^ {\tau} _ {k} v _ {A} ^ {k} - \frac {1}{4} _ {4} h ^ {\tau \tau} _ {4} h ^ {\tau} _ {k} v _ {A} ^ {k} + \frac {1}{2} _ {4} h _ {k l} v _ {A} ^ {k} v _ {A} ^ {l} + \frac {1}{8} _ {4} h ^ {k} _ {k} v _ {A} ^ {2} + \frac {5}{6 4} (_ {4} h ^ {\tau \tau}) ^ {2} v _ {A} ^ {2} \\ \left. + \frac {5}{8} _ {6} h ^ {\tau \tau} v _ {A} ^ {2} - \frac {3}{2} _ {4} h ^ {\tau} _ {k} v _ {A} ^ {k} v _ {A} ^ {2} + \frac {2 7}{3 2} _ {4} h ^ {\tau \tau} v _ {A} ^ {4} + \frac {5}{1 6} v _ {A} ^ {6} \right] \\ + \mathcal {O} \left(\epsilon^ {7}\right). \tag {152} \\ \end{array}
$$

The field should be evaluated somehow at $\vec{z}_A$ . This is a formal series since the metric derived via the point particle description diverges at $\vec{z}_A$ .

Now let us regularize this equation with the Hadamard partie finie regularization (see [88, 143] and for example, [26, 30] in the literature of the post-Newtonian approximation). Consider a function $f(\vec{x})$ which can be expanded around $\vec{z}_1$ in the form

$$
f (\vec {r} _ {1}) = \sum_ {p = p _ {0}} \frac {1}{r _ {1} ^ {p}} \underset {p} {\hat {f}} (\vec {N} _ {1}). \tag {153}
$$

Then the Hadamard partie finie at $\vec{z}_1$ of the function $f(\vec{x})$ is defined by

$$
\oint \frac {d \Omega_ {\mathbf {n} _ {1}}}{4 \pi} \underset {0} {\hat {f}} (\vec {n} _ {1}). \tag {154}
$$

For example, by this procedure $h^{\tau \tau}$ becomes (see Equation (130))

$$
[ h ^ {\tau \tau} ] _ {1} ^ {H} = 4 \epsilon^ {4} \frac {P _ {2} ^ {\tau}}{r _ {1 2}} + \epsilon^ {6} \left[ - 2 \frac {m _ {2}}{r _ {1 2}} \{(\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - v _ {2} ^ {2} \} - 1 6 \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} + 7 \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} \right] + \mathcal {O} (\epsilon^ {7}) \tag {155}
$$

for star 1. In the above equation, $[f]_A^{\mathrm{H}}$ means that we regularize the quantity $f$ at star $A$ by the Hadamard partie finie. Evaluating Equation (152) and $[\sqrt{-g}]_A^{\mathrm{H}}$ by this procedure, then comparing the result to Equation (148) combined with Equations (149, 150, 151), we find at least up to 3 PN order:

$$
P _ {A \Theta} ^ {\tau} = m _ {A} [ \sqrt {- g} u _ {A} ^ {\tau} ] _ {A} ^ {\mathrm{H}}. \tag {156}
$$

It is important that even after regularizing all the divergent terms, there remains a nonlinear effect. Equation (156) is natural. And note that we have never assumed this relation in advance. This relation has been derived by solving the evolution equation for $P_{A\Theta}^{\tau}$ functionally. In this regard, it is worth mentioning that the naturality of Equation (156) supports the use of the Hadamard partie finie regularization (or any regularization procedures if we can reproduce Equation (156) with them) to derive the 3 PN mass-energy relation to deal with divergences when one uses Dirac delta distributions $^{11}$ .

Finally, we note that up to 3 PN order

$$
[ \sqrt {- g} u _ {A} ^ {\tau} ] _ {A} ^ {\mathrm{H}} = [ \sqrt {- g} ] _ {A} ^ {\mathrm{H}} [ u _ {A} ^ {\tau} ] _ {A} ^ {\mathrm{H}} + \mathcal {O} (\epsilon^ {7}) \tag {157}
$$

is satisfied if we use the Hadamard partie finie regularization explained above.

# 7 Third Post-Newtonian Momentum-Velocity Relation

We now derive the 3 PN momentum-velocity relation by calculating the $Q_{A}^{i}$ integral at 3 PN order. From the definition of the $Q_{A}^{i}$ integral, Equation (91),

$$
Q _ {A} ^ {i} = \epsilon^ {6} \oint_ {\partial B _ {A}} d S _ {k} \left(_ {1 0} \Lambda_ {N} ^ {\tau k} - v _ {A 1 0} ^ {k} \Lambda_ {N} ^ {\tau \tau}\right) y _ {A} ^ {i}, \tag {158}
$$

we find that the calculation required is almost the same as that in the equation for $dP_{A}^{\tau}/d\tau$ . Namely, it turns out that we do not need to use the direct-integration method to compute the $Q_{A}^{i}$ integral. Therefore it is straightforward to evaluate the surface integrals in the definition of $Q_{A}^{i}$ . Here we split $Q_{A}^{i}$ into $Q_{A\Theta}^{i}$ and $Q_{A\chi}^{i}$ as

$$
Q _ {A} ^ {i} = Q _ {A \Theta} ^ {i} + Q _ {A \chi} ^ {i}, \tag {159}
$$

with

$$
Q _ {A \Theta} ^ {i} = \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \left(\Theta_ {N} ^ {\tau k} - v _ {A} ^ {k} \Theta_ {N} ^ {\tau \tau}\right) y _ {A} ^ {i}, \tag {160}
$$

$$
Q _ {A \chi} ^ {i} = \epsilon^ {- 4} \oint_ {\partial B _ {A}} d S _ {k} \left(\chi_ {N} ^ {\tau k \alpha \beta}, _ {\alpha \beta} - v _ {A} ^ {k} \chi_ {N} ^ {\tau \tau \alpha \beta}, _ {\alpha \beta}\right) y _ {A} ^ {i}, \tag {161}
$$

and we show only $Q_{A\Theta}^{i}$ :

$$
\leq^ {6} Q _ {1 \Theta} ^ {i} = - \epsilon^ {6} \frac {m _ {1} ^ {3} m _ {2} n _ {1 2} ^ {\langle i j \rangle} v _ {1 2} ^ {j}}{2 r _ {1 2} ^ {3}} = - \epsilon^ {6} \frac {d}{d \tau} \left(\frac {m _ {1} ^ {3} m _ {2}}{6 r _ {1 2} ^ {3}} r _ {1 2} ^ {i}\right) = \epsilon^ {6} \frac {d}{d \tau} \left(\frac {1}{6} m _ {1} ^ {3} a _ {1} ^ {i}\right), \tag {162}
$$

where $\leq_{n}f$ is the quantity $f$ up to order $n$ inclusively. Here it should be understood that $a_A^i$ in the last expression is evaluated with the Newtonian acceleration.

$Q_{A}^{i}$ of $\mathcal{O}(\epsilon^6)$ appears at the 4 PN or higher order field. Thus up to 3 PN order, ${}_{6}Q_{A}^{i}$ affects the equations of motion only through the 3 PN momentum-velocity relation. For this reason, not $Q_{A\chi}^{i}$ but $Q_{A\Theta}^{i}$ is necessary to derive the 3 PN equations of motion. The explicit expression for $Q_{A\chi}^{i}$ is given in [91].

Now with $Q_{A\Theta}^{i}$ in hand, we obtain the momentum-velocity relation. It turns out that the $\chi$ part of the momentum velocity relation is a trivial identity $^{12}$ . Thus, defining the $\Theta$ parts of $P_{A}^{\mu}$ and $D_{A}^{i}$ in the same way as for $Q_{A}^{i}$ , we obtain

$$
P _ {1 \Theta} ^ {i} = P _ {1 \Theta} ^ {\tau} v _ {1} ^ {i} + \epsilon^ {6} \frac {d}{d \tau} \left(\frac {1}{6} m _ {1} ^ {3} a _ {1} ^ {i}\right) + \epsilon^ {2} \frac {d D _ {1 \Theta} ^ {i}}{d \tau}. \tag {163}
$$

As explained in the previous Sections 3.4 and 4.4, we define the representative point $z_{A}^{i}$ of star A by choosing the value of $D_{A}^{i}$ . In other words, one can freely choose $D_{A}^{i}$ in principle $^{13}$ . One may set $D_{A}^{i}$ equal to zero up to 2.5 PN order. Alternatively, one may find it “natural” to see a three-momentum proportional to a three-velocity and take another choice,

$$
D _ {A \Theta} ^ {i} = - \epsilon^ {4} \frac {1}{6} m _ {A} ^ {3} a _ {A} ^ {i} = \epsilon^ {4} \delta_ {A \Theta} ^ {i}. \tag {164}
$$

Henceforth, we shall define $z_A^i$ by this equation.

Finally, it is important to realize that the nonzero dipole moment $D_{A}^{i}$ of order $\epsilon^{4}$ affects the 3 PN field and the 3 PN equations of motion in essentially the same manner as the Newtonian dipole moment affects the Newtonian field and equations of motion. From Equations (78, 79, 80) we see that $\delta_{A\Theta}^{i}$ appears only at $_{10}h^{\tau\tau}$ as

$$
h ^ {\tau \tau} | _ {\delta_ {A \Theta}} = 4 \epsilon^ {1 0} \sum_ {A = 1, 2} \frac {\delta_ {A \Theta} ^ {k} r _ {A} ^ {k}}{r _ {A} ^ {3}} + \mathcal {O} (\epsilon^ {1 1}). \tag {165}
$$

Then the corresponding acceleration becomes

$$
\left. m _ {1} a _ {1} ^ {i} \right| _ {\delta_ {A \Theta}} = - \epsilon^ {6} \frac {3 m _ {1} \delta_ {2 \Theta} ^ {i}}{r _ {1 2} ^ {3}} n _ {1 2} ^ {\langle i k \rangle} + \epsilon^ {6} \frac {3 m _ {2} \delta_ {1 \Theta} ^ {i}}{r _ {1 2} ^ {3}} n _ {1 2} ^ {\langle i k \rangle} - \epsilon^ {6} \frac {d ^ {2} \delta_ {1 \Theta} ^ {i}}{d \tau^ {2}}. \tag {166}
$$

The last term compensates the $Q_A^i$ integral contribution appearing through the momentum-velocity relation (163).

Note that this change of the acceleration does not affect the existence of the conservation of the (Newtonian-sense) energy,

$$
\left. m _ {1} a _ {1} ^ {i} \right| _ {\delta_ {A \Theta}} v _ {1} ^ {i} + \left. m _ {2} a _ {2} ^ {i} \right| _ {\delta_ {A \Theta}} v _ {2} ^ {i} = \epsilon^ {6} \frac {d}{d \tau} \sum_ {A = 1, 2} \left[ \delta_ {A \Theta} ^ {k} \frac {d v _ {A} ^ {k}}{d \tau} - v _ {A} ^ {k} \frac {d}{d \tau} \delta_ {A \Theta} ^ {k} \right]. \tag {167}
$$

# 8 Third Post-Newtonian Equations of Motion

# 8.1 Third post-Newtonian equations of motion with logarithmic terms

To derive the 3 PN equations of motion, we evaluate the surface integrals in the general form of the equations of motion (111) using the field $8h^{\tau \tau}$ , the field $\leq_6 h^{\mu \nu}$ , the 3 PN body zone contributions, and the 3 PN $N / B$ contributions corresponding to the results from the super-potential method and the super-potential-in-series method. We then combine the result with the terms from the direct-integration method.

From 3 PN order, the effects of the $Q_A^{K_l i}$ and $R_A^{K_l ij}$ integrals appearing in the 3 PN field in $h_B^{\mu \nu}$ contribute to the 3 PN equations of motion. ${}_6Q_{A\Theta}^i$ given in Equation (162) affects the 3 PN equations of motion through the 3 PN momentum-velocity relation. Since we define the representative points of the stars via Equation (164), we add the corresponding acceleration given by Equation (166). Furthermore, our choice of the representative points of the stars makes $D_{A\chi}^i$ appear independently of $D_{A\Theta}^i$ in the field, and hence ${}_4D_{A\chi}^i$ affects the 3 PN equations of motion. In summary, the $\leq 4Q_A^{K_l i}, \leq 4R_A^{K_l ij}, {}_6Q_{A\Theta}^i, \delta_{A\Theta}^i$ , and ${}_4D_{A\chi}^i$ contributions to the 3 PN field can be written as

$$
{ } _ { 1 0 } h ^ { \tau \tau } + { } _ { 8 } h ^ { k } { } _ { k } = 4 \sum _ { A = 1 , 2 } \frac { r _ { A } ^ { k } } { r _ { A } ^ { 3 } } \left( \delta _ { A \Theta } ^ { k } + { } _ { 4 } D _ { A \chi } ^ { k } + { } _ { 4 } R _ { A } ^ { k l l } - \frac { 1 } { 2 } { } _ { 4 } R _ { A } ^ { l l k } \right) + \dots , \tag {168}
$$

where “...” are other contributions. On the other hand, ${}_{6}Q_{A\Theta}^{i}$ and $\delta_{A\Theta}^{i}$ affect the equations of motion through the momentum-velocity relation in Equation (111),

$$
m _ {1} \left(\frac {d v _ {1} ^ {i}}{d \tau}\right) _ {\leq 3 \mathrm{PN}} = - \epsilon^ {6} \frac {d _ {6} Q _ {1 \Theta} ^ {i}}{d \tau} - \epsilon^ {6} \frac {d ^ {2} \delta_ {1 \Theta} ^ {i}}{d \tau^ {2}} + \dots , \tag {169}
$$

but they cancel each other out, since we choose Equation (164). Then these contributions to a 3 PN acceleration can be summarized into

$$
\left(\underset {\leq 4} {\text { the   contribution   to }} m _ {1} a _ {1} ^ {i} \text { from } Q _ {A} ^ {K _ {l} i}, \underset {\leq 4} {\leq} R _ {A} ^ {K _ {l} i j}, _ {6} Q _ {A \Theta} ^ {i}, \delta_ {A \Theta , 4} ^ {i} D _ {A \chi} ^ {i}\right) = \epsilon^ {6} \frac {1 1 8}{9} \frac {m _ {1} ^ {3} m _ {2} ^ {2}}{r _ {1 2} ^ {6}} r _ {1 2} ^ {i} + \epsilon^ {6} \frac {1 1 8}{9} \frac {m _ {1} ^ {2} m _ {2} ^ {3}}{r _ {1 2} ^ {6}} r _ {1 2} ^ {i}. \tag {170}
$$

Collecting these contributions mentioned above, we obtain the 3 PN equations of motion. However, we found that logarithmic terms having the arbitrary constants $\epsilon R_{A}$ in their arguments survive,

$$
\begin{array}{l} m _ {1} \left(\frac {d v _ {1} ^ {i}}{d \tau}\right) ^ {\text { with   log }} = m _ {1} \left(\frac {d v _ {1} ^ {i}}{d \tau}\right) _ {\leq 2. 5 \mathrm{PN}} \\ + \epsilon^ {6} \frac {m _ {1} ^ {2} m _ {2}}{r _ {1 2} ^ {3}} \left[ \frac {4 4 m _ {1} ^ {2}}{3 r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {1}}\right) - \frac {4 4 m _ {2} ^ {2}}{3 r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {2}}\right) \right. \\ \left. + \frac {2 2 m _ {1}}{r _ {1 2}} \left(5 (\vec {n} _ {1 2} \cdot \vec {V}) ^ {2} n _ {1 2} ^ {i} - V ^ {2} n _ {1 2} ^ {i} - 2 (\vec {n} _ {1 2} \cdot \vec {V}) V ^ {i}\right) \ln \left(\frac {r _ {1 2}}{\epsilon R _ {1}}\right) \right] \\ + \dots + \mathcal {O} (\epsilon^ {7}), \tag {171} \\ \end{array}
$$

where the acceleration through 2.5 PN order, $(dv_{1}^{i}/d\tau)_{\leq2.5\mathrm{PN}}$ , is the Damour and Deruelle 2.5 PN acceleration. In our formalism, we have computed it in [95]. The “...” stands for the terms that do not include any logarithms.

Since this equation contains two arbitrary constants, the body zone radii $R_{A}$ , at first sight its predictive power on the orbital motion of the binary seems to be limited. In the next Section 8.2, we shall show that by a reasonable redefinition of the representative points of the stars, we can remove $R_{A}$ from our equations of motion. There, we show the explicit form of the 3 PN equations of motion we obtained.

# 8.2 Arbitrary constant $\epsilon R_{A}$

The reason logarithms appear in the 3 PN equations of motion (171) is in a sense easy to understand. Since the post-Newtonian approximation is a weak field expansion, at some level of iteration in the evaluation of field, we have inevitably logarithms in the field through volume integrals such as

$$
\int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \left(\frac {m}{y}\right) ^ {n}, \tag {172}
$$

where m and y are typical the mass and length scale in the orbital motion, respectively, and n is a positive integer. For instance consider $m_{1}^{3}(\vec{r}_{1} \cdot \vec{a}_{1})/r_{1}^{5}$ as an integrand. Then we find

$$
\int_ {N / B} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \frac {m _ {1} ^ {3} (\vec {y} _ {1} \cdot \vec {a} _ {1})}{y _ {1} ^ {5}} = \frac {m _ {1} ^ {3} (\vec {r} _ {1} \cdot \vec {a} _ {1})}{3 r _ {1} ^ {3}} \ln \left(\frac {r _ {1}}{\epsilon R _ {1}}\right) + \dots . \tag {173}
$$

Actually, we could remove the $\epsilon R_{A}$ dependence in our 3 PN equations of motion via an alternative choice of the center of mass. The following alternative choice of the representative point of star A removes the $\epsilon R_{A}$ dependence in Equation (171):

$$
D _ {A \Theta , \mathrm{new}} ^ {i} (\tau) = \epsilon^ {4} \delta_ {A \Theta} ^ {i} (\tau) - \epsilon^ {4} \frac {2 2}{3} m _ {A} ^ {3} a _ {A} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {A}}\right) \equiv \epsilon^ {4} \delta_ {A \Theta} ^ {i} (\tau) + \epsilon^ {4} \delta_ {A \ln} ^ {i} (\tau) \equiv \epsilon^ {4} \delta_ {A} ^ {i} (\tau). \tag {174}
$$

Note that this redefinition of the center of mass does not affect the existence of the energy conservation as was shown by Equation (167). We can examine the effect of this redefinition on the equations of motion using Equation (166) (using $\delta_{A\ln}^{i}$ instead of $\delta_{A\Theta}$ ). Then we have

$$
\begin{array}{l} m _ {1} a _ {1} ^ {i} | _ {\delta_ {A \ln}} = - \epsilon^ {6} \frac {3 m _ {1} \delta_ {2 \ln} ^ {i}}{r _ {1 2} ^ {3}} n _ {1 2} ^ {\langle i k \rangle} + \epsilon^ {6} \frac {3 m _ {2} \delta_ {1 \ln} ^ {i}}{r _ {1 2} ^ {3}} n _ {1 2} ^ {\langle i k \rangle} - \epsilon^ {6} \frac {d ^ {2} \delta_ {1 \ln} ^ {i}}{d \tau^ {2}} \\ = - \frac {4 4 m _ {1} ^ {3} m _ {2} ^ {2}}{3 r _ {1 2} ^ {5}} n _ {1 2} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {1}}\right) + \frac {4 4 m _ {1} ^ {2} m _ {2} ^ {3}}{3 r _ {1 2} ^ {5}} n _ {1 2} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {2}}\right) \\ - \frac {2 2 m _ {1} ^ {3} m _ {2}}{r _ {1 2} ^ {4}} \left(5 (\vec {n} _ {1 2} \cdot \vec {V}) ^ {2} n _ {1 2} ^ {i} - V ^ {2} n _ {1 2} ^ {i} - 2 (\vec {n} _ {1 2} \cdot \vec {V}) V ^ {i}\right) \ln \left(\frac {r _ {1 2}}{\epsilon R _ {1}}\right) \\ + \frac {2 2 m _ {1} ^ {3} m _ {2}}{3 r _ {1 2} ^ {4}} \left(\frac {m _ {1}}{r _ {1 2}} n _ {1 2} ^ {i} + \frac {m _ {2}}{r _ {1 2}} n _ {1 2} ^ {i} - V ^ {2} n _ {1 2} ^ {i} + 8 (\vec {n} _ {1 2} \cdot \vec {V}) ^ {2} n _ {1 2} ^ {i} - 2 (\vec {n} _ {1 2} \cdot \vec {V}) ^ {2} V ^ {i}\right). (1 7 5) \\ \end{array}
$$

Comparing the above equations with Equation (171), we easily conclude that the representative point $z_A^i$ of star $A$ defined by

$$
D _ {A \Theta , \text { new }} ^ {i} (\tau) = \epsilon^ {- 6} \int_ {B _ {A}} d ^ {3} y \left(y ^ {i} - z _ {A} ^ {i} (\tau)\right) \Theta_ {N} ^ {\tau \tau} (\tau , y ^ {k}) = \epsilon^ {4} \delta_ {A} ^ {i} (\tau) \tag {176}
$$

obeys the equations of motion free from any logarithmic term and hence free from any ambiguity up to 3 PN order inclusively. We note that in our formalism $z_A^i$ is defined by the value of $D_A^i$ , and in turn we have a freedom to assign to $D_A^i$ any value as we like (though it may be natural to set the value of $D_A^i$ such that $z_A^i$ resides inside star $A$ ). We also note that we define $z_A^i$ order by order.

We mention here that Blanchet and Faye $[27]$ have already noticed that in their 3 PN equations of motion a suitable coordinate transformation removes (parts of) the logarithmic dependence of arbitrary parameters corresponding (roughly) to our body zone radii $^{14}$ . It is well-known that choosing different values of dipole moments corresponds to a coordinate transformation.

# 8.3 Consistency relation

Our new choice of the dipole moments of the stars is in a sense natural. To see this, let us consider the harmonic condition:

$$
0 = h ^ {\tau \mu} _ {, \mu} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \left[ \frac {\dot {P} _ {A} ^ {\tau}}{r _ {A}} + \frac {r _ {A} ^ {i}}{r _ {A} ^ {3}} \left(P _ {A} ^ {\tau} v _ {A} ^ {i} + \epsilon^ {2} \dot {D} _ {A} ^ {i} - P _ {A} ^ {i}\right) \right] + \sum_ {A = 1, 2} \oint_ {\partial B _ {A}} \frac {d S _ {i}}{| \vec {x} - \vec {y} |} \left(\Lambda_ {N} ^ {\tau i} - v _ {A} ^ {i} \Lambda_ {N} ^ {\tau \tau}\right) \dots , (1 7 7)
$$

$$
0 = h ^ {i \mu} _ {, \mu} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \frac {\dot {P} _ {A} ^ {i}}{r _ {A}} + \sum_ {A = 1, 2} \oint_ {\partial B _ {A}} \frac {d S _ {j}}{| \vec {x} - \vec {y} |} \left(\Lambda_ {N} ^ {i j} - v _ {A} ^ {j} \Lambda_ {N} ^ {\tau i}\right) + \dots , \tag {178}
$$

where “...” are irrelevant terms. These equations are a manifestation of the fact that the harmonic condition is consistent with the evolution equation for $P_{A}^{\tau}$ , the momentum-velocity relation, and the equations of motion (and relations among higher multipole moments, hidden in “...”). Thus if the logarithmic dependence of $\epsilon R_{A}$ arises from the second term of Equation (178), $P_{A}^{i}$ must have the same logarithmic dependence (times minus sign) to ensure harmonicity. This and the momentum velocity relation in turn mean $P_{A}^{\tau}$ , $v_{A}^{i} = \dot{z}_{A}^{i}$ , or $D_{A}^{i}$ have corresponding logarithmic dependence. Since we already know that the $P_{A}^{\tau}$ have no logarithm up to 3 PN order, $z_{A}^{i}$ or $D_{A}^{i}$ should have logarithms. This is consistent with the fact that a choice of $D_{A}^{i}$ determines $z_{A}^{i}$ . $z_{A}^{i}$ depends on logarithms if the old choice is taken, while it does not if our new choice is taken.

There is yet another fact which supports our interpretation. Let us retain $D_{A}^{i} \neq 0$ for a while. We then find that the near zone dipole moment $D_{N}^{i}$ defined by a volume integral of $\Lambda_{N}^{\tau\tau} y^{i}$ becomes

$$
\epsilon^ {2} D _ {N} ^ {i} \equiv \epsilon^ {- 4} \int_ {N} d ^ {3} y \Lambda_ {N} ^ {\tau \tau} y ^ {i} = \sum_ {A = 1, 2} P _ {A} ^ {\tau} z _ {A} ^ {i} + \epsilon^ {2} \sum_ {A = 1, 2} D _ {A} ^ {i} + \epsilon^ {- 4} \int_ {N / B} d ^ {3} y \Lambda_ {N} ^ {\tau \tau} y ^ {i}. \tag {179}
$$

Then if we take the old choice of $D_{A}^{i}$ , the volume integral becomes

$$
\int_ {N / B} d ^ {3} y \Lambda_ {N} ^ {\tau \tau} y ^ {i} = \epsilon^ {4} \frac {2 2}{3} \sum_ {A = 1, 2} m _ {A} ^ {3} a _ {A} ^ {i} \ln \left(\frac {r _ {1 2}}{\epsilon R _ {A}}\right) + \dots , \tag {180}
$$

where terms denoted by “...” have no logarithmic dependence. Notice that the near zone dipole moment can be freely determined, say, $D_{N}^{i}=0$ , because we can define the origin of the near zone freely. By taking temporal derivatives twice of $D_{N}^{i}$ , we see that $D_{A,new}^{i}$ gives a natural definition of the center of the mass in terms of which the 3 PN equations of motion are independent of $\epsilon R_{A}$ .

# 8.4 Third post-Newtonian equations of motion

By adding $m_{1}a_{1}^{i}|_{\delta_{A\ln}}$ to Equation (171), we obtain our 3 PN equations of motion for two spherical compact stars whose representative points are defined by Equation (176),

$$
\begin{array}{l} m _ {1} \frac {d v _ {1} ^ {i}}{d \tau} = - \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \\ + \epsilon^ {2} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \left[ - v _ {1} ^ {2} - 2 v _ {2} ^ {2} + 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + \frac {5 m _ {1}}{r _ {1 2}} + \frac {4 m _ {2}}{r _ {1 2}} \right] \\ + \epsilon^ {2} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} V ^ {i} \left[ 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right] \\ + \epsilon^ {4} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \left[ - 2 v _ {2} ^ {4} + 4 v _ {2} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - 2 (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {3}{2} v _ {1} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + \frac {9}{2} v _ {2} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} \right] \\ - 6 (\vec {v} _ {1} \cdot \vec {v} _ {2}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - \frac {1 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} - \frac {5 7}{4} \frac {m _ {1} ^ {2}}{r _ {1 2} ^ {2}} - 9 \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} - \frac {6 9}{2} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \\ \end{array}
$$

$$
\begin{array}{l} + \frac {m _ {1}}{r _ {1 2}} \left(- \frac {1 5}{4} v _ {1} ^ {2} + \frac {5}{4} v _ {2} ^ {2} - \frac {5}{2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {3 9}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} \right. \\ \left. - 3 9 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) + \frac {1 7}{2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2}\right) \\ \left. + \frac {m _ {2}}{r _ {1 2}} \left(4 v _ {2} ^ {2} - 8 (\vec {v} _ {1} \cdot \vec {v} _ {2}) + 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2}\right) \right] \\ + \epsilon^ {4} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} V ^ {i} \left[ \frac {m _ {1}}{r _ {1 2}} \left(- \frac {6 3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \frac {5 5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) + \frac {m _ {2}}{r _ {1 2}} (- 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2})) \right. \\ + v _ {1} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + 4 v _ {2} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - 5 v _ {2} ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) (\vec {n} _ {1 2} \cdot \vec {V}) \\ \left. - 6 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} + \frac {9}{2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {3} \right] \\ + \epsilon^ {5} \frac {4 m _ {1} ^ {2} m _ {2}}{5 r _ {1 2} ^ {3}} \left[ n _ {1 2} ^ {i} (\vec {n} _ {1 2} \cdot \vec {V}) \left(- 6 \frac {m _ {1}}{r _ {1 2}} + \frac {5 2}{3} \frac {m _ {2}}{r _ {1 2}} + 3 V ^ {2}\right) + V ^ {i} \left(2 \frac {m _ {1}}{r _ {1 2}} - 8 \frac {m _ {2}}{r _ {1 2}} - V ^ {2}\right) \right] \\ + \epsilon^ {6} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} \left[ \frac {3 5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {6} - \frac {1 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} v _ {1} ^ {2} + \frac {1 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \right. \\ + 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} - \frac {1 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} v _ {2} ^ {2} + \frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} v _ {2} ^ {2} \\ - 1 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) v _ {2} ^ {2} - 2 \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) ^ {2} v _ {2} ^ {2} + \frac {1 5}{2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} v _ {2} ^ {4} \\ + 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) v _ {2} ^ {4} - 2 v _ {2} ^ {6} \\ + \frac {m _ {1}}{r _ {1 2}} \left(- \frac {1 7 1}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {4} + \frac {1 7 1}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right. \\ - \frac {7 2 3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + \frac {3 8 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \\ - \frac {4 5 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} + \frac {2 2 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {1} ^ {2} - \frac {2 0 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} \\ + \frac {1 9 1}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} - \frac {9 1}{8} v _ {1} ^ {4} - \frac {2 2 9}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ + 2 4 4 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) - \frac {2 2 5}{2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) \\ + \frac {9 1}{2} v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {1 7 7}{4} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {2 2 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {2} ^ {2} \\ - \frac {2 8 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {2} ^ {2} + \frac {2 5 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {2} ^ {2} - \frac {9 1}{4} v _ {1} ^ {2} v _ {2} ^ {2} \\ \left. + 4 3 (\vec {v} _ {1} \cdot \vec {v} _ {2}) v _ {2} ^ {2} - \frac {8 1}{8} v _ {2} ^ {4}\right) \\ + \frac {m _ {2}}{r _ {1 2}} \left(- 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + 1 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \right. \\ + 6 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} + 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ + 1 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + 4 (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} \\ - 4 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) v _ {2} ^ {2} - 1 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} v _ {2} ^ {2} - 8 \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) v _ {2} ^ {2} + 4 v _ {2} ^ {4}) \\ + \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} \bigg (- (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} + 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + \frac {4 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} \\ \left. + 1 8 \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) - 9 v _ {2} ^ {2}\right) \\ \end{array}
$$

$$
\begin{array}{l} + \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left(\frac {4 1 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - \frac {3 7 5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + \frac {1 1 1 3}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} \right. \\ \left. - \frac {6 1 5 \pi^ {2}}{6 4} (\vec {n} _ {1 2} \cdot \vec {V}) ^ {2} + 1 8 v _ {1} ^ {2} + \frac {1 2 3 \pi^ {2}}{6 4} V ^ {2} + 3 3 (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {3 3}{2} v _ {2} ^ {2}\right) \\ + \frac {m _ {1} ^ {2}}{r _ {1 2} ^ {2}} \left(- \frac {2 0 6 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} + 5 4 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - \frac {9 3 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} \right. \\ \left. + \frac {4 7 1}{8} v _ {1} ^ {2} - \frac {3 5 7}{4} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) + \frac {3 5 7}{8} v _ {2} ^ {2}\right) \\ \left. + \frac {1 6 m _ {2} ^ {3}}{r _ {1 2} ^ {3}} + \frac {m _ {1} ^ {2} m _ {2}}{r _ {1 2} ^ {3}} \left(\frac {5 4 7}{3} - \frac {4 1 \pi^ {2}}{1 6}\right) - \frac {1 3 m _ {1} ^ {3}}{1 2 r _ {1 2} ^ {3}} + \frac {m _ {1} m _ {2} ^ {2}}{r _ {1 2} ^ {3}} \left(\frac {5 4 5}{3} - \frac {4 1 \pi^ {2}}{1 6}\right) \right] \\ + \epsilon^ {6} \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} V ^ {i} \left[ \frac {1 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} - \frac {4 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {5} - \frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} v _ {1} ^ {2} \right. \\ + 6 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) - 6 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {3} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) \\ - 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} - 1 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {2} ^ {2} + 1 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} v _ {2} ^ {2} \\ + (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} v _ {2} ^ {2} - 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) v _ {2} ^ {2} + 8 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) v _ {2} ^ {2} \\ + 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) v _ {2} ^ {4} - 7 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {2} ^ {4} \\ + \frac {m _ {2}}{r _ {1 2}} \left(- 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + 8 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} \right. \\ + 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) + 4 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) \\ - 2 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) v _ {2} ^ {2} - 4 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) v _ {2} ^ {2}) \\ + \frac {m _ {1}}{r _ {1 2}} \left(- \frac {2 4 3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} + \frac {5 6 5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) \right. \\ - \frac {2 6 9}{4} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {2} - \frac {9 5}{1 2} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) ^ {3} + \frac {2 0 7}{8} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) v _ {1} ^ {2} \\ - \frac {1 3 7}{8} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) v _ {1} ^ {2} - 3 6 \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) + \frac {2 7}{4} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right) \\ \left. + \frac {8 1}{8} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {1}\right) v _ {2} ^ {2} + \frac {8 3}{8} \left(\vec {n} _ {1 2} \cdot \vec {v} _ {2}\right) v _ {2} ^ {2}\right) \\ + \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {2}} \left(4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + 5 (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) \\ + \frac {m _ {1} m _ {2}}{r _ {1 2} ^ {2}} \left(- \frac {3 0 7}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) + \frac {4 7 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + \frac {1 2 3 \pi^ {2}}{3 2} (\vec {n} _ {1 2} \cdot \vec {V})\right) \\ \left. + \frac {m _ {1} ^ {2}}{r _ {1 2} ^ {2}} \left(\frac {3 1 1}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) - \frac {3 5 7}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) \right] \\ + \mathcal {O} (\epsilon^ {7}), \tag {181} \\ \end{array}
$$

in the harmonic gauge.

Now we list some features of our 3 PN equations of motion. In the test-particle limit, our 3 PN equations of motion coincide with a geodesic equation for a test-particle in the Schwarzschild metric in harmonic coordinates (up to 3 PN order). Suppose that star 1 is a test particle, star 2 is represented by the Schwarzschild metric, and $\vec{v}_{2}=0$ . Then the geodesic equation for star 1 becomes

$$
a _ {1} ^ {i} = - \frac {m _ {2}}{r _ {1 2} ^ {2}} n _ {1 2} ^ {i} + \epsilon^ {2} \frac {m _ {2}}{r _ {1 2} ^ {2}} \left(- v _ {1} ^ {2} n _ {1 2} ^ {i} + \frac {4 m _ {2}}{r _ {1 2}} n _ {1 2} ^ {i} + 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) v _ {1} ^ {i}\right)
$$

$$
+ \epsilon^ {4} \frac {m _ {2} ^ {2}}{r _ {1 2} ^ {3}} \left(2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} n _ {1 2} ^ {i} - 9 \frac {m _ {2}}{r _ {1 2}} n _ {1 2} ^ {i} - 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) v _ {1} ^ {i}\right)
$$

$$
+ \epsilon^ {6} \frac {m _ {2} ^ {3}}{r _ {1 2} ^ {4}} \left(- (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} n _ {1 2} ^ {i} + \frac {1 6 m _ {2}}{r _ {1 2}} n _ {1 2} ^ {i} + 4 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) v _ {1} ^ {i}\right) + \mathcal {O} (\epsilon^ {7}) \tag {182}
$$

in the harmonic gauge. Thus, in the test particle limit Equation (181) coincides with the geodesic equation for a test particle in the Schwarzschild metric up to 3 PN order.

With the help of the formulas developed in $[28]$ , we have checked the Lorentz invariance of Equation (181) (in the post-Newtonian perturbative sense). Also, we have checked that our 3 PN acceleration admits a conserved energy of the binary orbital motion (modulo the 2.5 PN radiation reaction effect). In fact, the energy E of the binary associated with Equation (181) is

$$
\begin{array}{l} E = \frac {1}{2} m _ {1} v _ {1} ^ {2} - \frac {m _ {1} m _ {2}}{2 r _ {1 2}} \\ + \epsilon^ {2} \left[ \frac {3}{8} m _ {1} v _ {1} ^ {4} + \frac {m _ {1} ^ {2} m _ {2}}{2 r _ {1 2} ^ {2}} + \frac {m _ {1} m _ {2}}{2 r _ {1 2}} \left(3 v _ {1} ^ {2} - \frac {7}{2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {1}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2})\right) \right] \\ + \epsilon^ {4} \left[ \frac {5}{1 6} m _ {1} v _ {1} ^ {6} - \frac {m _ {1} ^ {3} m _ {2}}{2 r _ {1 2} ^ {3}} - \frac {1 9 m _ {1} ^ {2} m _ {2} ^ {2}}{8 r _ {1 2} ^ {3}} \right. \\ + \frac {m _ {1} ^ {2} m _ {2}}{2 r _ {1 2} ^ {2}} \left(- 3 v _ {1} ^ {2} + \frac {7}{2} v _ {2} ^ {2} + \frac {2 9}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - \frac {1 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2}\right) \\ + \frac {m _ {1} m _ {2}}{4 r _ {1 2}} \left(\frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + \frac {3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - \frac {9}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} \right. \\ - \frac {1 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} + \frac {2 1}{2} v _ {1} ^ {4} + \frac {1 3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ \left. + \left. 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {5 5}{2} v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {1 7}{2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {3 1}{4} v _ {1} ^ {2} v _ {2} ^ {2}\right) \right] \\ + \epsilon^ {6} \left[ \frac {3 5}{1 2 8} m _ {1} v _ {1} ^ {8} + \frac {3 m _ {1} ^ {4} m _ {2}}{8 r _ {1 2} ^ {4}} + \frac {4 6 9 m _ {1} ^ {3} m _ {2} ^ {2}}{1 8 r _ {1 2} ^ {4}} \right. \\ + \frac {m _ {1} ^ {2} m _ {2} ^ {2}}{2 r _ {1 2} ^ {3}} \left(\frac {5 4 7}{6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} - \frac {3 1 1 5}{2 4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - \frac {1 2 3 \pi^ {2}}{3 2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {V})\right) \\ - \left. \frac {5 7 5}{9} v _ {1} ^ {2} + \frac {4 1 \pi^ {2}}{3 2} (\vec {V} \cdot \vec {v} _ {2}) + \frac {4 4 2 9}{7 2} (\vec {v} _ {1} \cdot \vec {v} _ {2})\right) \\ + \frac {m _ {1} ^ {3} m _ {2}}{2 r _ {1 2} ^ {3}} \left(- \frac {4 3 7}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} + \frac {3 1 7}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) + 3 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + \frac {3 0 1}{1 2} v _ {1} ^ {2} \right. \\ - \left. \frac {3 3 7}{1 2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {5}{2} v _ {2} ^ {2}\right) \\ + \frac {m _ {1} m _ {2}}{r _ {1 2}} \left(- \frac {5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {5} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - \frac {5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} - \frac {5}{3 2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3}\right) \\ + \frac {1 9}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} + \frac {1 5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} \\ + \frac {3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} v _ {1} ^ {2} + \frac {1 9}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {4} v _ {1} ^ {2} - \frac {2 1}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {4} \\ - 2 (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {4} + \frac {5 5}{1 6} v _ {1} ^ {6} - \frac {1 9}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {4} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ - \frac {1 5}{3 2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {4 5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ \end{array}
$$

$$
\begin{array}{l} + \frac {5}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {1 1}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {1 3 9}{1 6} v _ {1} ^ {4} (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ - \frac {3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {4 1}{8} v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} \\ + \frac {1}{1 6} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {3} - \frac {4 5}{1 6} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {1} ^ {2} v _ {2} ^ {2} - \frac {2 3}{3 2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} v _ {2} ^ {2} + \frac {7 9}{1 6} v _ {1} ^ {4} v _ {2} ^ {2} \\ \left. - \frac {1 6 1}{3 2} v _ {1} ^ {2} v _ {2} ^ {2} \left(\vec {v} _ {1} \cdot \vec {v} _ {2}\right)\right) \\ + \frac {m _ {1} ^ {2} m _ {2}}{r _ {1 2} ^ {2}} \left(- \frac {4 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {4} + \frac {7 5}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {3} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) - \frac {1 8 7}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} + \frac {1 1}{2} v _ {1} ^ {4}\right) \\ + \frac {2 4 7}{2 4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {3} + \frac {4 9}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {1} ^ {2} + \frac {8 1}{8} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {1} ^ {2} \\ - \frac {2 1}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {1} ^ {2} - \frac {1 5}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - \frac {3}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) (\vec {v} _ {1} \cdot \vec {v} _ {2}) \\ + \frac {2 1}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) - 2 7 v _ {1} ^ {2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) + \frac {5 5}{2} (\vec {v} _ {1} \cdot \vec {v} _ {2}) ^ {2} + \frac {4 9}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) ^ {2} v _ {2} ^ {2} \\ - \frac {2 7}{2} (\vec {n} _ {1 2} \cdot \vec {v} _ {1}) (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) v _ {2} ^ {2} + \frac {3}{4} (\vec {n} _ {1 2} \cdot \vec {v} _ {2}) ^ {2} v _ {2} ^ {2} + \frac {5 5}{4} v _ {1} ^ {2} v _ {2} ^ {2} - 2 8 (\vec {v} _ {1} \cdot \vec {v} _ {2}) v _ {2} ^ {2} \\ \left. \right.\left. + \left. \frac {1 3 5}{1 6} v _ {2} ^ {4}\right)\right] \\ + (1 \leftrightarrow 2) + \mathcal {O} (\epsilon^ {7}). \tag {183} \\ \end{array}
$$

This orbital energy of the binary is computed based on that one found in Blanchet and Faye [27], the relation between their 3 PN equations of motion and our result described in Section 8.5 below, and Equation (167). (After constructing $E$ given as in Equation (183), we have checked that our 3 PN equations of motion make $E$ to be conserved.)

We note that Equation (171) as well gives a correct geodesic equation in the test-particle limit, is Lorentz invariant, and admits the conserved energy. These facts can be seen by the form of $a_1^i|_{\delta_{A\ln}}$ , Equation (175); it is zero when $m_1 \to 0$ , is Lorentz invariant up to 3 PN order, and is the effect of the mere redefinition of the dipole moments which does not break energy conservation.

Finally, we here mention one computational detail. We have retained during our calculation R-dependent terms with positive powers of R or logarithms of R. As stated below Equation (76), it is a good computational check to show that our equations of motion do not depend on R physically. In fact, we found that the R-dependent terms cancel each other out in the final result. There is no need to employ a gauge transformation to remove such an R dependence. As for terms with negative powers of R, we simply assume that those terms cancel out the R dependent terms from the far zone contribution. Indeed, Pati and Will [129, 130], whose method we have adopted to compute the far zone contribution, have proved that all the R-dependent terms cancel out between the far zone and the near zone contributions through all post-Newtonian orders.

# 8.5 Comparison

By comparing Equation (181) with the Blanchet and Faye 3 PN equations of motion [27], we find the following relation:

$$
m _ {1} \vec {a} _ {1} ^ {\text { this   work }} = m _ {1} (\vec {a} _ {1} ^ {\mathrm{BF}}) _ {\lambda = - \frac {1 9 8 7}{3 0 8 0}} + m _ {1} \vec {a} _ {1} | _ {\delta_ {A \ln}} + m _ {1} \vec {a} _ {1} | _ {\delta_ {A, \mathrm{BF}}}, \tag {184}
$$

where $m_{1}\vec{a}_{1}^{this work}$ is the 3 PN acceleration given in Equation (181), $(\vec{a}_{1}^{\mathrm{BF}})_{\lambda=-1987/3080}$ is the Blanchet and Faye 3 PN acceleration with $\lambda=-1987/3080$ , and $m_{1}\vec{a}_{1}|_{\delta_{A\ln}}$ is given in Equation (175) with $\epsilon R_{A}$ replaced by $r_{A}^{\prime}$ for notational consistency with the Blanchet and Faye 3 PN

equations of motion shown in [27]. $m_{1}\vec{a}_{1}|_{\delta_{A,\mathrm{BF}}}$ is an acceleration induced by the following dipole moments of the stars:

$$
\delta_ {A, \mathrm{BF}} ^ {i} = - \frac {3 7 0 9}{1 2 6 0} m _ {A} ^ {3} a _ {A} ^ {i}. \tag {185}
$$

We can compute $m_{1}a_{1}^{i}|_{\delta_{A,\mathrm{BF}}}$ by substituting $\delta_{A,\mathrm{BF}}^{i}$ instead of $\delta_{A\Theta}^{i}$ into Equation (166). Thus, by choosing the dipole moments,

$$
D _ {A \Theta , \mathrm{BF}} ^ {i} = \epsilon^ {4} \delta_ {A \Theta} ^ {i} - \epsilon^ {4} \delta_ {A, \mathrm{BF}} ^ {i}, \tag {186}
$$

we have the 3 PN equations of motion in completely the same form as $(\vec{a}_{1}^{\mathrm{BF}})_{\lambda=-1987/3080}$ . In other words, our 3 PN equations of motion physically agree with $(\vec{a}_{1}^{\mathrm{BF}})_{\lambda=-1987/3080}$ modulo the definition of the dipole moments (or equivalently, the coordinate transformation under the harmonic coordinates condition). In [93], we have shown some arguments that support this conclusion.

The value of $\lambda$ that we found, $\lambda = -1987 / 3080$ , is perfectly consistent with the relation (1) and the result of [54] ( $\omega_{\mathrm{static}} = 0$ ).

Finally, let us discuss the ambiguity in the 3 PN equations of motion previously derived by Blanchet and Faye in [25, 27]. In their formalism, Dirac delta distributions are used to achieve the point particle limit. The (Lorentz invariant generalized) Hadamard partie finie regularization has been extensively employed to regularize divergences caused by their use of a singular source. In fact, unless regularization is employed, divergences occur both in the evaluation of the 3 PN field where (Poisson) integrals diverge at the location of the stars and in their derivation of the equations of motion where a substitution of the metric into a geodesic equation causes divergences.

When we regularize some terms at a point, say, $z_{A}^{i}$ , where the terms are singular, using the Hadamard partie finite regularization, roughly speaking we take an angular average of the finite part of the terms in the neighborhood of the singular point. Then if there are logarithmic terms such as $\ln(|\vec{x}-\vec{z}_{A}|/r_{12})$ , we should take an angular average over some sphere centered on $z_{A}^{i}$ with a finite radius. The radius of the sphere is arbitrary but we do not ignore it because we should ensure the argument of the logarithms to be dimensionless.

The problem here is that there is a priori no reason to expect that the radius for each star introduced to regularize the field and another radius for that star introduced to regularize the geodesic equation coincide with each other. Thus the Blanchet and Faye 3 PN equations of motion have four arbitrary constants instead of two in our equations of motion. In our framework, we can see the origin of the number if we assume that we have defined a different body zone $B_{A}^{\prime}$ in the derivation of the equations of motion from $B_{A}$ used in the derivation of the 3 PN field. However, in reality, we have only one body zone for each star. In our formalism the field is expressed in terms of the four-momentum (and multipole moments) which are defined as volume integrals over the body zone. On the other hand our general form of the equations of motion has been derived based on the conservation law of the four-momentum, and thus we evaluate the surface integrals in the general form of the equations of motion over the boundary of the body zone.

In fact, [25, 27] have shown that two of the four arbitrary constants can be removed by using a gauge freedom remaining in the harmonic gauge condition; the two places where the singular points exist are in some sense ambiguous. The remaining two turn out to appear as the ratios $\zeta_A = r_A' / s_A$ ( $A = 1,2$ ) where $r_A'$ and $s_A$ are the four regularization parameters (roughly speaking, the radii of $B_A'$ and $B_A$ in the terminology in the previous paragraph). Blanchet and Faye [25, 27] then proved that assuming the equations of motion are polynomials of the two masses of the stars, those two ratios should satisfy $\ln \zeta_A = \sigma + \lambda(m_1 + m_2) / m_A$ where $\sigma$ and $\lambda$ are pure numbers. Then they showed that in order for their equations of motion to admit conserved energy, then $\sigma = 159/308$ , while no argument was found to fix $\lambda$ .

The above argument in turn means that the Blanchet and Faye 3 PN equations of motion do not give a conserved energy unless $B_A'$ is different from $B_A$ . Damour, Jaranowski, and Schäfer [54] pointed out that there is an unsatisfactory feature in the generalized partie finie regularization

which contradicts with the mathematical structure of general relativity. Indeed, by using dimensional regularization which is pointed out by them to be more satisfactory in this regard, [54] derived an unambiguous ADM Hamiltonian in the ADM transeverse traceless gauge. Later, Blanchet et al. [22] used dimensional regularization and found that their new equations of motion physically agree with ours and admit a conserved energy.

# 8.6 Summary

To deal with strongly self-gravitating objects such as neutron stars, we have used the surface integral approach with the strong field point particle limit. The surface integral approach is achieved by using the local conservation of the energy momentum, which led us to the general form of the equations of motion that are expressed entirely in terms of surface integrals. The use of the strong field point particle limit and the surface integral approach makes our 3 PN equations of motion applicable to inspirating compact binaries which consist of strongly self-gravitating regular stars (modulo the scalings imposed on the initial hypersurface). Our 3 PN equations of motion depend only on the masses of the stars and are independent of their internal structure such as their density profiles or radii. Thus our result supports the strong equivalence principle up to 3 PN order.

At 3 PN order, it does not seem possible to derive the field in a closed form. This is because not all the super-potentials required are available, and thus we could not evaluate all the Poisson-type N/B integrals. Some of the integrands allow us to derive super-potentials in a series form in the neighborhood of the stars. For others, we have adopted an idea that Blanchet and Faye have used in $[25, 26, 27]$ . The idea is that while abandoning the complete derivation of the 3 PN gravitational field valid throughout N/B, one exchanges the order of integrals $^{15}$ . We first evaluate the surface integrals in the evolution equation for the energy of a star and the general form of equations of motion, and then we evaluate the remaining volume integrals. Using these methods, we first derived the 3 PN mass-energy relation and the momentum-velocity relation. The 3 PN mass-energy relation admits a natural interpretation. We then evaluated the surface integrals in the general form of equations of motion, and obtained the equations of motion up to 3 PN order of accuracy.

At 3 PN order, our equations of motion contain logarithms of the body zone radii $R_{A}$ . We showed that we could remove the logarithmic terms by a suitable redefinition of the representative points of the stars. Thus we could transform our 3 PN equations of motion into unambiguous equations which do not contain any arbitrarily introduced free parameters.

Our so-obtained 3 PN equations of motion agree physically (modulo a definition of the representative points of the stars) with the result derived by Blanchet and Faye [27] with $\lambda = -1987/3080$ , which is consistent with Equation (1) and $\omega_{\mathrm{static}} = 0$ reported by Damour, Jaranowski, and Schäfer [54]. This result indirectly supports the validity of the dimensional regularization in the ADM canonical approach in the ADMTT gauge.

Blanchet and Faye $[25, 27]$ introduced four arbitrary parameters. In the Hadamard partie finie regularization, one has to introduce a sphere around each singular point (representing a point mass) whose radius is a free parameter. In their framework, regularizations are employed in the evaluation of both the gravitational field having two singular points and the two equations of motion. Since, in their formalism, there is a priori no reason to expect that the spheres introduced for the evaluation of the field and the equations of motion coincide, there arise four arbitrary parameters. This is in contrast to our formalism where each body zone introduced in the evaluation of the field is

inevitably the same as the body zone with which we defined the energy and the three-momentum of each star for which we derived our equations of motion.

Actually, the redefinition of the representative points in our formalism corresponds to the gauge transformation in $[27]$ , and only two of the four parameters remain in $[27]$ . Then they have used one of the remaining two free parameters to ensure the energy conservation, and there remains only one arbitrary parameter $\lambda$ which they could not fix in their formalism.

On the other hand, our 3 PN equations of motion have no ambiguous parameter, admit conservation of an orbital energy of the binary system (when we neglect the 2.5 PN radiation reaction effect), and respect Lorentz invariance in the post-Newtonian perturbative sense. We emphasize that we do not need to a posteriori adjust some parameters to make our 3 PN equations of motion to satisfy the above three physical features.

We here note that Blanchet et al. [22], who computed the 3 PN equations of motion in the harmonic gauge using the dimensional regularization, have recently obtained the same value for $\lambda$ .

The gauge condition in a harmonic gauge is related to the equations of motion. One may ask if the 3 PN equations of motion that have been derived so far guarantee the harmonic gauge condition through the corresponding post-Newtonian accuracy. This has not been tested yet. Let us call the n PN accurate metric components to be the components that are needed to compute the n PN equations of motion. Then the harmonic condition for the n PN field requires that matter obeys the n - 1 PN equations of motion. Thus, we need the 4 PN field to check if our resulting 3 PN equations of motion are a necessary condition to fulfil the harmonic gauge condition. This is beyond our current knowledge.

# 8.7 Going further

The 4 PN templates may be required to detect gravitational wave directly with high signal to noise ratio and extract astrophysical information from the wave. We have to derive the 4 PN equations of motion to derive the 4 PN templates if we use the energy balance.

At this moment, we should say that it is difficult to derive the 4 PN equations of motion. The technical obstacles regarding the derivation of the 4 PN equations of motion are the following. First, to derive the 4 PN equations of motion, we have to derive the 4 PN gravitational field at least in the neighborhood of a star. This requires the 3 PN gravitational field valid throughout the near zone. As seen in this article, however, it seems impossible to derive the 3 PN accurate gravitational field in harmonic coordinates in a closed form completely. We have not yet found the super-potentials necessary for us to evaluate the 3 PN Poisson integrals (or, retarded integrals).

Second, the amounts of calculations required would be too large to derive the 4 PN equations of motion successfully. For example, the Newtonian field consists of only two terms. The Landau–Lifshitz pseudotensor at the Newtonian order consists of basically one term. The gravitational field at 1 PN order consists of about 10 terms, while the Landau–Lifshitz pseudotensor has about 5 terms and thus we have to handle 50 terms to derive the 1 PN equations of motion. Next, the 2 PN field has about $10^{2}$ terms. The Landau–Lifshitz pseudotensor in terms of $h^{\mu\nu}$ consists of about 50 terms and thus $\sim10^{3}$ terms must be treated to derive the 2 PN equations of motion. The number of terms in the 3 PN field are of order $10^{3}\sim10^{4}$ , while Landau–Lifshitz pseudotensor has about 100 terms. Thus we may encounter $\sim10^{5}$ terms to derive the 3 PN equations of motion. Then at the 4 PN order, we may expect $10^{4}\sim10^{5}$ terms for the field, 500 terms for the integrand, and $\sim10^{6}\sim10^{7}$ terms for the equations of motion. Furthermore, for each term spatial and temporal derivative generate additional terms. The number of newly generated terms are about three or four for each term. Thus the number of terms in the intermediate expressions to be dealt with are about ten-fold the number quoted above at each order $^{16}$ . Such an enormous

number of terms forces us to use algebraic computing softwares to deal with them. In this work, we have extensively used the algebraic computing software Maple [118], Mathematica [166], and MathTensor [120] to deal with tensors. However, quantity changes quality. When some equations are produced through a sequence of black-boxes (that is, calculations done by computers) we could not check the equations term by term even if the calculations follow trivial procedures such as taking a Taylor expansion of equations around a star and extracting off some coefficients from the Taylor-expanded equations. Then, how could one confirm one's result? Some possible tests include the following: Do the resulting equations of motion respect Lorentz invariance? Do the equations of motion admit the existence of a conserved energy? Does the gravitational field (if obtainable) satisfy the harmonic condition? Do the results obtained by more than two groups agree with each other? Then do all these consist of a set of necessary and sufficient criteria for confirmation?

One way to derive the 4 PN equations of motion is to use brute force (if the first difficulty – how to derive the required super-potentials at 3 PN order – was successfully dealt with). On the other hand, one may find a scaling appropriate to the late inspiralling phase and construct an approximation scheme based on such a scaling, as the post-Newtonian approximation is based on the Newtonian scaling. In the post-Newtonian approximation, the lowest order term is just the Newtonian term, and the first order term is the 1 PN correction. In the post-Minkowskian approximation, (the lowest order is just a straight line and) the first order correction is valid for any velocity but only for a weak field. Likewise, a new approximation scheme (if any) would give (the lowest order of the conservative dynamics and) the first order correction to the radiation reaction effect. Such an approximation may produce a smaller number of terms than the post-Newtonian approximation and thus give easier-to-treat equations of motion. Also the relatively lower order equations of motion obtained by such an approximation are expected to give templates with the same accuracy as the accuracy achieved by the relatively higher order post-Newtonian equations of motion. Thus, such an approximation scheme is an attractive alternative to the post-Newtonian approximation. The construction of such an approximation remains to be done in future work.

# 9 Acknowledgements

The authors are greatly indebted to Bernard F. Schutz for his helpful comments and valuable discussions while they visited the Max-Planck-Institut für Gravitationsphysik (Albert-Einstein-Institut), Germany. When the authors wrote this article, Y.I. stayed at Tokyo University, for which he particularly thanks Masaru Shibata for his hospitality. The authors would like to thank Takashi Fukumoto for fruitful discussion.

# A Far Zone Contribution

A formal retardation expansion gives a divergent integral when we evaluate the gravitational field. Schematically the field $h(\tau, x^{i})$ at the field point $(\tau, x^{i})$ is given by (we omit the indices of the field for simplicity)

$$
h (\tau , x ^ {i}) \sim \int d ^ {3} y \frac {f (\tau - \epsilon | \vec {x} - \vec {y} | , \vec {y})}{| \vec {x} - \vec {y} |} \sim \sum_ {n} ^ {\infty} \frac {(- 1) ^ {n}}{n !} \epsilon^ {n} \int d ^ {3} y | \vec {x} - \vec {y} | ^ {n - 1} \frac {d ^ {n}}{d t ^ {n}} f (t, \vec {y}). \tag {187}
$$

This integral is divergent if we set the upper bound of the integral to infinity. In our formalism we take the upper bound of the integral as $\sim R/\epsilon$ and keep the R dependence in the field. In the last step of the derivation, we let R go to infinity if and only if this procedure gives finite result at least up to the relevant post-Newtonian order. At a high order of the retardation expansion approximation, however, some terms must have a $\ln R$ , and/or a $\mathcal{R}^{n}(n>0)$ dependence. In this case, we can not let R go infinite.

To solve this problem, it is important to realize that the field at $(\tau, x^{i})$ consists of two contributions: the retarded integral inside the near zone (the near zone contribution) and the retarded integral outside the near zone (the far zone contribution).

Since R is introduced artificially, we expect that the far zone contribution to the near zone field must have some R dependent terms which cancel out completely the R dependent terms in the near zone. This expectation is the case and was proved to any post-Newtonian order by Pati and Will $[129]$ within their formalism. They have shown that the total field (which is the sum of the near zone contribution plus the far zone contribution) is finite and independent of R.

In this section we calculate the far zone contribution to the 3 PN equations of motion to make this article self-contained. We entirely follow $[129, 162, 165]$ , and the result is the same as theirs: The far zone field does not have any influence on the equations of motion up to 3 PN order inclusively.

For the near zone field, we write the field as

$$
h _ {N (C)} ^ {\mu \nu} = h _ {N (N)} ^ {\mu \nu} + h _ {N (F)} ^ {\mu \nu} + h _ {\mathrm{H}} ^ {\mu \nu}, \tag {188}
$$

$$
h _ {N (N)} ^ {\mu \nu} = 4 \int_ {N = \{y: | y | \leq \mathcal {R} / \epsilon \}} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |}, \tag {189}
$$

$$
h _ {N (F)} ^ {\mu \nu} = 4 \int_ {F = \{y: | y | > \mathcal {R} / \epsilon \}} d ^ {3} y \frac {\Lambda^ {\mu \nu} (\tau - \epsilon | \vec {x} - \vec {y} | , y ^ {k} ; \epsilon)}{| \vec {x} - \vec {y} |}, \tag {190}
$$

where $h_{N}^{\mu\nu}$ is the field in the near zone, $h_{N(N)}^{\mu\nu}$ is the near zone integral contribution to the near zone field, and $h_{N(F)}^{\mu\nu}$ is the far zone integral contribution to the near zone field. Our task here is to evaluate $h_{N(F)}^{\mu\nu}$ to the 3 PN order. In turn, this means that we should derive $h_{F}^{\mu\nu}$ to lower order because the integrand $\Lambda^{\mu\nu}$ in the retarded integral for $h_{N(F)}^{\mu\nu}$ consists of $h_{F}^{\mu\nu}$ .

Only in this section, we do not use our bookkeeping parameter $\epsilon$ while it should be understood that $\vec{v}$ is of order $\epsilon$ , and $m$ of order $\epsilon^2$ .

Now we evaluate the near zone contribution to the far zone field using a multipole expansion:

$$
h _ {F (N)} ^ {\mu \nu} = 4 \int_ {N} \frac {d ^ {3} y}{| \vec {x} - \vec {y} |} \Lambda^ {\mu \nu} (t - | \vec {x} - \vec {y} |, \vec {y})
$$

$$
= 4 \sum_ {l = 0} \frac {(- 1) ^ {l}}{l !} \partial_ {K _ {l}} \left(\frac {1}{r} M ^ {K _ {l} \mu \nu} (u)\right), \tag {191}
$$

where $K_{l}$ is a collective multi-index, $r \equiv |\vec{x}|$ is the distance from the near zone center to the field point, and u = t - r. The near zone multipole moments $M^{K_{l}\mu\nu}$ are defined as

$$
M ^ {K _ {l} \mu \nu} (u) \equiv \int_ {N} d ^ {3} y   \Lambda^ {\mu \nu} (u, \vec {y})   y ^ {K _ {l}}. \tag {192}
$$

Next, the far zone contribution to the far zone field point can be evaluated as follows. The key idea would be an introduction of a new time u and may be seen from the following transformation of the retarded integral where the radial integral is transformed into a temporal integral from the past infinity to u = t - r:

$$
\begin{array}{l} h _ {F (F)} ^ {\mu \nu} (t, \vec {x}) = 4 \int_ {F} d ^ {3} x ^ {\prime} \frac {\Lambda^ {\mu \nu} (t - | \vec {x} - \vec {x} ^ {\prime} | , \vec {x} ^ {\prime})}{| \vec {x} - \vec {x} ^ {\prime} |} \\ = 4 \int_ {- \infty} ^ {u} d u ^ {\prime} \oint_ {F} \frac {\Lambda^ {\mu \nu} (u ^ {\prime} + r ^ {\prime} , \vec {x} ^ {\prime})}{t - u ^ {\prime} - \vec {n} ^ {\prime} \cdot \vec {x}} [ r ^ {\prime} (u ^ {\prime}, \Omega^ {\prime}) ] ^ {2} d \Omega , \tag {193} \\ \end{array}
$$

where

$$
r ^ {\prime} (u ^ {\prime}, \Omega^ {\prime}) = \frac {(t - u ^ {\prime}) ^ {2} - r ^ {2}}{2 (t - u ^ {\prime} - \vec {n} ^ {\prime} \cdot \vec {x})}. \tag {194}
$$

We then make STF decomposition of the integrand $\Lambda^{\mu\nu}$ in the retarded integral as $\Lambda^{\mu\nu} \sim f_{B,L}r^{-B}n^{\langle L\rangle}$ , where $\vec{n} = \vec{r}/r$ . The integrand in Equation (193) becomes a summation of terms, each of which consists of (some algebraic combinations of) near zone multipole moments (defined by Equation (192)) multiplied by terms which explicitly depend on t, $u'$ , and $\Omega'$ . Roughly speaking, the idea is that we do integral by parts many times, each time increasing the number of the time derivatives of the multipole moments, and assume that the system is sufficiently stationary in the past so that the contributions from the past infinity disappear. We then have for the far zone contribution to the far zone field point:

$$
\begin{array}{l} h_{F(F)}^{\mu \nu}(u,x^{i}) = \sum_{\substack{B\neq 2\\ l}}\left(\frac{2}{r}\right)^{B - 2}n^{\langle L\rangle}\sum_{q = 0}\mathcal{D}_{B,L}^{q}(z)r^{q}\frac{d^{q}f_{B,L}(u)}{du^{q}} \\ + \frac {n ^ {\langle L \rangle}}{r} \int_ {0} ^ {\infty} f _ {2, L} (u - s) Q _ {L} \left(1 + \frac {s}{r}\right) + n ^ {\langle L \rangle} \sum_ {q = 0} \mathcal {D} _ {2, L} ^ {q} (z) r ^ {q} \frac {d ^ {q} f _ {2 , L} (u)}{d u ^ {q}}, \tag {195} \\ \end{array}
$$

with $z = R/r^{17}$ , and

$$
\mathcal {D} _ {B, L} ^ {q} (z) = \frac {(- 1) ^ {q}}{q !} \int_ {1} ^ {1 + 2 z} \frac {(\zeta - 1) ^ {q}}{(\zeta^ {2} - 1) ^ {B - 2}} A _ {B, L} (\zeta , \alpha) d \zeta - \sum_ {p = 0} ^ {q} k _ {B, L} ^ {(q - p + 1)} (1 + 2 z) \frac {(- 2 z) ^ {p}}{p !} \tag {196}
$$

for $B \neq 2$ and

$$
\mathcal {D} _ {2, L} ^ {q} (z) = \frac {(- 1) ^ {q + 1}}{2 q !} \int_ {1} ^ {1 + 2 z} (\zeta - 1) ^ {q} d \zeta \int_ {- 1} ^ {1 - \alpha} \frac {\mathrm{P} _ {l} (y)}{\zeta - y} d y \tag {197}
$$

for $B = 2$ . Other quantities in the above equations are given by

$$
A _ {B, L} (\zeta , \alpha) \equiv \frac {1}{2} \int_ {1 - \alpha} ^ {1} \frac {\mathrm{P} _ {l} (y)}{\zeta - y} d y, \tag {198}
$$

$$
\alpha \equiv (\zeta - 1) (\zeta + 1 - 2 z) / (2 z), \tag {199}
$$

$$
\frac {d k _ {B , L} ^ {(m)} (\zeta)}{d \zeta} = k _ {B, L} ^ {(m - 1)} (\zeta) \quad \text {   for   } m \geq 1, \tag {200}
$$

$$
k _ {B, L} ^ {(0)} (\zeta) \equiv \frac {A _ {B , L} (\zeta , 2)}{(\zeta^ {2} - 1) ^ {B - 2}} \quad \text { for } m \geq 1. \tag {201}
$$

Here $P_{l}$ and $Q_{l}$ are the Legendre function of the first and the second kind. $\alpha$ represents an angular defect due to the fact that the far zone integral does not cover the whole spacetime due to the near zone. The function $k(t)$ and the retarded time multi-derivatives of the STF coefficient $f_{B,L}$ comes from the recursive integrals by parts.

Then combining the near zone and the far zone contributions to the far zone field point, we have to the required order:

$$
h _ {F} ^ {0 0} = 4 \left[ \frac {P _ {N} ^ {0}}{r} - \partial_ {k} \left(\frac {D _ {N} ^ {k} (u)}{r}\right) + \frac {1}{2} \partial_ {k l} \left(\frac {I _ {N} ^ {k l} (u)}{r}\right) - \frac {1}{6} \partial_ {k l m} \left(\frac {I _ {N} ^ {k l m} (u)}{r}\right) \right] + 7 \left(\frac {P _ {N} ^ {0}}{r}\right) ^ {2}, \tag {202}
$$

$$
h _ {F} ^ {0 i} = 4 \left[ \frac {P _ {N} ^ {i} (u)}{r} - \partial_ {k} \left(\frac {J _ {N} ^ {k i} (u)}{r}\right) + \frac {1}{2} \partial_ {k l} \left(\frac {J _ {N} ^ {k l i} (u)}{r}\right) \right], \tag {203}
$$

$$
h _ {F} ^ {i j} = 4 \left[ \frac {Z _ {N} ^ {i j} (u)}{r} - \partial_ {k} \left(\frac {Z _ {N} ^ {k i j} (u)}{r}\right) + \frac {1}{2} \partial_ {k l} \left(\frac {Z _ {N} ^ {k l i j} (u)}{r}\right) \right] + \left(\frac {P _ {N} ^ {0}}{r}\right) ^ {2} n ^ {i j}, \tag {204}
$$

where $P_N^0 = M^{00}$ , $P_N^i = M^{0i}$ , $D_N^i = M^{i00}$ , $I_N^{ij} = M^{ij00}$ , $J_N^{ki} = M^{k0i}$ , $J_N^{kli} = M^{kl0i}$ , $Z_N^{ij} = M^{ij}$ , $Z_N^{kij} = M^{kij}$ , and $Z_N^{klij} = M^{klij}$ .

Next we evaluate the far zone contribution to the near zone field point. A transformation of the retarded integral (193) is again used and similar arguments below Equation (193) lead to the following formulae that we use

$$
\begin{array}{l} h_{N(F)}^{\mu \nu}(t,x^{i}) = \sum_{\substack{B\neq 2\\ l}}\left(\frac{2}{r}\right)^{B - 2}n^{\langle L\rangle}\sum_{q = 0}\mathcal{E}_{B,L}(z)^{q}  r^{q}  \frac{d^{q}f_{B,L}(t)}{dt^{q}} \\ + \frac {n ^ {\langle L \rangle}}{r} \int_ {0} ^ {\infty} f _ {2, L} (u - s) \mathrm{Q} _ {l} \left(1 + \frac {s}{r}\right) + n ^ {\langle L \rangle} \sum_ {q = 0} \mathcal {E} _ {2, L} ^ {q} (z) r ^ {q} \frac {d ^ {q} f _ {2 , L} (t)}{d t ^ {q}}, \tag {205} \\ \end{array}
$$

with

$$
\mathcal {E} _ {B, L} ^ {q} (z) = \frac {(- 1) ^ {q}}{q !} \int_ {2 z - 1} ^ {2 z + 1} \frac {\zeta^ {q}}{(\zeta^ {2} - 1) ^ {B - 2}} A _ {B, L} (\zeta , \alpha) d \zeta - \sum_ {p = 0} ^ {q} k _ {B, L} ^ {(q - p + 1)} (1 + 2 z) \frac {(- 1 - 2 z) ^ {p}}{p !} \tag {206}
$$

for $B \neq 2$ and

$$
\mathcal {E} _ {2, L} ^ {q} (z) = \frac {(- 1) ^ {q + 1}}{q !} \left\{\int_ {1} ^ {1 + 2 z} \zeta^ {q} \mathrm{Q} _ {l} (\zeta) d \zeta + \frac {1}{2} \int_ {2 z - 1} ^ {2 z + 1} \zeta^ {q} d \zeta \int_ {- 1} ^ {1 - \alpha} \frac {\mathrm{P} _ {l} (y)}{\zeta - y} d y \right\} \tag {207}
$$

for B = 2.

Evaluating the coefficients $\mathcal{E}_{B,L}^{q}$ , we finally have

$$
\begin{array}{l} h _ {N (F)} ^ {\tau \tau} (\tau , x ^ {i}) = \epsilon^ {1 0} \left[ - 8 P _ {N} ^ {\tau} n ^ {\langle k l \rangle} \int_ {1} ^ {\infty} ^ {(4)} I _ {N} ^ {k l} (t - r \zeta)   \mathrm{Q} _ {2} (\zeta)   d \zeta - \frac {8}{3} P _ {N} ^ {\tau} \int_ {1} ^ {\infty} ^ {(4)} I _ {N} ^ {k k} (t - r \zeta)   \mathrm{Q} _ {0} (\zeta)   d \zeta \right. \\ \left. + \frac {4}{3} P _ {N} ^ {\tau (4)} I _ {N} ^ {k l} (t) n ^ {\langle k l \rangle} + \frac {8}{3} P _ {N} ^ {\tau (4)} I _ {N} ^ {k k} (t) \left(1 + \ln \left(\frac {\mathcal {R}}{\epsilon r}\right)\right) \right] + \mathcal {O} (\epsilon^ {1 1}), (208) \\ h _ {N (F)} ^ {\tau i} = \mathcal {O} (\epsilon^ {9}), (209) \\ \end{array}
$$

$$
h _ {N (F)} ^ {k k} = \epsilon^ {8} \left[ - 1 6 P _ {N} ^ {\tau} \int_ {1} ^ {(2)} Z _ {N k} ^ {k} (t - r \zeta)   \mathrm{Q} _ {0} (\zeta)   d \zeta + 1 6 P _ {N} ^ {\tau (2)} Z _ {N k} ^ {k} \left(1 + \ln \left(\frac {\mathcal {R}}{\epsilon r}\right)\right) \right] + (\epsilon^ {9} 2 1 0)
$$

where we reintroduce our bookkeeping parameter $\epsilon$ , and $Z_{N}^{ij}$ is written with the near zone quadrupole moment as $Z_{N}^{ij} = (1/2)d^{2}I_{N}^{ij}/d\tau^{2}$ . We note that the fields $h^{\tau\tau}$ of $\mathcal{O}(\epsilon^{10})$ and $h^{\mu i}$ of $\mathcal{O}(\epsilon^{8})$ are the 3 PN fields in our formalism. Finally assuming that in the distant past the binary was sufficiently stationary, we find that the far zone contribution becomes a function of time only at the 3 PN order. It turns out that only the spatial derivative of those 3 PN fields contributes to the 3 PN equations of motion. Thus the far zone contribution does not affect the equations of motion up to 3 PN order inclusively.

In this section, we have given a highly rough sketch about the method developed by Pati and Will $[129]$ which is based on their previous work $[165]$ . Readers may consult $[129, 130, 162, 163, 165]$ for more details.

# B Effects of Extendedness of Stars

In this section, we derive the spin-orbit coupling force, the quadrupole-orbit coupling force, the spin-spin coupling force, and the spin geodesic precession equation to the lowest order.

Our order-counting of the multipole couplings may need an explanation. The magnitude of a mass multipole moment and a current multipole moment of order l of a star are roughly $(\text{mass}) \times (\text{radius of the star})^{l}$ and $(\text{mass}) \times (\text{radius of the star})^{l} \times (\text{velocity of steller internal motion})$ . Since we assume slow stellar rotation where (velocity of steller internal motion) = $\mathcal{O}(\epsilon)$ and the strong field point particle limit where (radius of the star) = $\mathcal{O}(\epsilon^{2})$ , the mass multipole moment and the current multipole moment are of order $\mathcal{O}(\epsilon^{2l+2})$ and $\mathcal{O}(\epsilon^{2l+3})$ , respectively. For example, the spin-orbit coupling force which takes a form of $(\text{mass}) \times (\text{orbital velocity}) \times (\text{spin})$ appears at $\mathcal{O}(\epsilon^{4})$ , that is, 2 PN order, not at the usual 1.5 PN order where rapid stellar rotation is assumed.

In summary the structure of the equations of orbital motion is written schematically as

$$
\begin{array}{l} m a ^ {i} = F _ {\mathrm{Newton}} ^ {i} + \epsilon^ {2} F _ {\mathrm{1PN}} ^ {i} + \epsilon^ {4} F _ {\mathrm{2PN}} ^ {i} + \epsilon^ {4} F _ {\mathrm{SO}} ^ {i} + \epsilon^ {4} F _ {\mathrm{QO}} ^ {i} + \epsilon^ {5} F _ {\mathrm{RR}} ^ {i} \\ + \epsilon^ {6} F _ {\mathrm{3PN}} ^ {i} + \epsilon^ {6} F _ {\mathrm{1PNSO}} ^ {i} + \epsilon^ {6} F _ {\mathrm{1PN,QO}} ^ {i} + \epsilon^ {6} F _ {\mathrm{OO}} ^ {i} + \epsilon^ {6} F _ {\mathrm{SS}} ^ {i} + \epsilon^ {6} F _ {\mathrm{TO}} ^ {i} + \mathcal {O} (\epsilon^ {7}). (2 1 1) \\ \end{array}
$$

Here $F_{Newton}^{i}$ , $F_{1 PN}^{i}$ , $F_{2 PN}^{i}$ , $F_{RR}^{i}$ , and $F_{3 PN}^{i}$ , respectively, are the Newtonian force, the 1 PN force, the 2 PN force, the 2.5 PN radiation reaction force, and the 3 PN force. $F_{SO}^{i}$ and $F_{1 PNSO}^{i}$ are the spin-orbit coupling force and its 1 PN correction, while $F_{QO}^{i}$ and $F_{1 PN QO}^{i}$ are the quadrupole-orbit coupling force and its 1 PN correction. $F_{OO}^{i}$ , $F_{SS}^{i}$ , and $F_{TO}^{i}$ are the octupole-orbit coupling force, spin-spin coupling force, and tidal-orbit coupling force, respectively $^{18}$ .

Though formally the effects of the 1 PN spin-orbit coupling, the 1 PN quadrupole-orbit, the octupole-orbit coupling, and the tidal-orbit coupling appear up to 3 PN order in our ordering, we focus our attention onto the lowest order spin-orbit coupling, the spin-spin coupling, and the quadrupole-orbit coupling forces. The 1 PN spin-orbit coupling force was derived by Tagoshi, Ohashi, and Owen [148].

Before investigating the multipole-orbit coupling forces, it is worth noticing that our definition of multipole moments is operational and the relation between these multipole moments and the intrinsic multipole moments of a star has not been given. Let us discuss briefly about this problem.

First, for an appropriate frame for the definition of multipole moments it is natural to define the multipole moments in a frame attached to the star, and nonrotating with respect to an asymptotic inertial frame (see $[37]$ for the case of an earth-satellite system in the solar system). If we do not define the multipole moments in an appropriate frame, for example, an apparent quadrupole moment would be produced by Lorentz contraction caused by the orbital motion of the star and an apparent spin would be produced by the Thomas precession. One realization of an appropriate frame are the (generalized) Fermi normal coordinates $[13]$ . To derive the spin-orbit coupling force in the same form as in previous works $[46, 108, 152]$ , it is sufficient to assume the coordinate transformation of $\Lambda_{N}^{\tau\tau}$ in the near zone coordinates to the $\Lambda_{A}^{\mu'\nu'}$ in the (generalized) Fermi normal coordinates in the following implicit form ([13, 37, 57, 58, 59, 60] (see also Appendix B.4):

$$
\Lambda_ {N} ^ {\tau \tau} = (\Gamma_ {A \tau^ {\prime}} ^ {\tau}) ^ {2} \Lambda_ {A} ^ {\tau^ {\prime} \tau^ {\prime}} + 2 \epsilon^ {2} \Gamma_ {A \tau^ {\prime}} ^ {\tau} \Gamma_ {A \underline {{i}} ^ {\prime}} ^ {\tau} \Lambda_ {A} ^ {\tau^ {\prime} \underline {{i}} ^ {\prime}} + \mathcal {O} (\epsilon^ {2}), \tag {212}
$$

with $\Gamma_{A^{\tau}}^{\tau}=1+\mathcal{O}(\epsilon^{2})$ and $\Gamma_{Ai}^{\tau}=v_{A}^{i}+\mathcal{O}(\epsilon^{2})$ (think of a Lorentz transformation). The $\epsilon^{2}$ in front of the second term arises from the body zone coordinates rescaling $(x_{A}^{i}=\epsilon^{2}\alpha_{A}^{i})$ . An explicit expression of $\Gamma_{A^{\nu}}^{\mu}$ is not required for our purpose. To 2 PN order, which is the sufficient order to derive the lowest order spin-orbit, spin-spin, and quadrupole-orbit coupling forces, the transformation

changes only $_{8}h^{\tau\tau}$ :

$$
h ^ {\tau \tau} = 4 \epsilon^ {4} \sum_ {A = 1, 2} \left[ \frac {P _ {A} ^ {\tau}}{r _ {A}} + \epsilon^ {2} \frac {r _ {A} ^ {i}}{r _ {A} ^ {3}} (d _ {A} ^ {i} + \epsilon^ {2} M _ {A} ^ {i k} v _ {A} ^ {k}) \right] + \dots , \tag {213}
$$

where “...” denotes irrelevant terms and

$$
d _ {A} ^ {i} = \epsilon^ {2} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \left(\Gamma_ {A \tau^ {\prime}} ^ {\tau}\right) ^ {2} \alpha_ {A} ^ {\frac {i}{\tau^ {\prime}}} \Lambda_ {A} ^ {\tau^ {\prime} \tau^ {\prime}}, \tag {214}
$$

$$
m _ {A} ^ {i j} = 2 \epsilon^ {4} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \Gamma_ {A \tau^ {\prime}} ^ {\tau} \alpha_ {A} ^ {[ i} \Lambda_ {A} ^ {j ^ {\prime} ] \tau^ {\prime}} \mathcal {O} (\epsilon) = M _ {A} ^ {i j} + \mathcal {O} (\epsilon), \tag {215}
$$

where $M_A^{ij}$ is given by Equation (90).

Second, we discuss the $\chi$ part of our multipole moments. We have used $\Lambda_{N}^{\mu\nu} = \Theta_{N}^{\mu\nu} + \chi_{N}^{\mu\nu\alpha\beta}, \alpha\beta$ in the definition of our multipole moments since we could not evaluate $\chi$ parts of multipole moments separately except for some low order moments. However, we have to take into account carefully the fact that $\chi$ parts of our higher multipole moments can affect the equations of motion for two point masses. An obvious example can be found from the definition of the energy and the mass. It is natural to define the mass as a volume integral of $\Theta_{N}^{\tau\tau}$ . In fact the $\chi$ part of the energy, $P_{A\chi}^{\tau}$ , appears at 2 PN order. As another example, the $\chi$ part of our dipole moment can be evaluated directly as

$$
D _ {A \chi} ^ {i} = \epsilon^ {2} \int_ {B _ {A}} d ^ {3} \alpha_ {A} \alpha_ {A} ^ {\underline {{i}}} \chi_ {N} ^ {\tau \tau \alpha \beta}, _ {\alpha \beta} = \epsilon^ {4} \frac {1 7 5 m _ {1} ^ {3} m _ {2} r _ {1 2} ^ {i}}{1 8 r _ {1 2} ^ {3}} + \mathcal {O} (\epsilon^ {5}). \tag {216}
$$

Thus if we define the center of mass of the star not by $D_{A}^{i}=0$ but by $D_{A\Theta}^{i}=0$ , the form of the equations of motion for the two point masses would change $^{19}$ .

Finally, we list the relevant field and $Q^{i}$ up to the required order to derive the lowest order spin-orbit, the spin-spin, and the quadrupole-orbit coupling forces and the spin geodesic precession equation.

$$
\begin{array}{l} h ^ {\tau \tau} = 4 \epsilon^ {6} \sum_ {A = 1, 2} \frac {r _ {A} ^ {k}}{r _ {A} ^ {3}} D _ {A} ^ {k} + 6 \epsilon^ {8} \sum_ {A = 1, 2} \frac {r _ {A} ^ {k} r _ {A} ^ {l}}{r _ {A} ^ {5}} I _ {A} ^ {k l} \\ = 4 \epsilon^ {6} \sum_ {A = 1, 2} \frac {r _ {A} ^ {k}}{r _ {A} ^ {3}} d _ {A} ^ {k} + 6 \epsilon^ {8} \sum_ {A = 1, 2} \frac {r _ {A} ^ {k} r _ {A} ^ {l}}{r _ {A} ^ {5}} I _ {A} ^ {k l} + 4 \epsilon^ {8} \sum_ {A = 1, 2} \frac {r _ {A} ^ {k}}{r _ {A} ^ {3}} M _ {A} ^ {k i} v _ {A} ^ {i} + \mathcal {O} (\epsilon^ {9}), \tag {217} \\ \end{array}
$$

$$
{ } _ { 6 } h ^ { \tau i } = 2 \sum _ { A = 1 , 2 } \frac { r _ { A } ^ { k } } { r _ { A } ^ { 3 } } M _ { A } ^ { k i } , \tag {218}
$$

$$
_ 6 h ^ {i j} = 4 \sum_ {A = 1, 2} \frac {r _ {A} ^ {k}}{r _ {A} ^ {3}} M _ {A} ^ {k (i} v _ {A} ^ {j)}. \tag {219}
$$

$$
Q _ {1} ^ {i} = \epsilon^ {4} \frac {2 m _ {2} M _ {1} ^ {i k}}{3 r _ {1 2} ^ {3}} r _ {1 2} ^ {k}. \tag {220}
$$

# B.1 Spin-orbit coupling force

It is well known that the definition of a dipole moment of the star, which we equate to zero to determine the center of mass of the star, affects the appearance of the spin-orbit coupling force

(see e.g. [108]). If one chooses $d_A^i = 0$ as in [94], the corresponding spin-orbit coupling force takes the usual form [46, 108, 152]

$$
\begin{array}{l} \left. F _ {1 \mathrm{SO}} ^ {i} \right| _ {d _ {A} ^ {i} = 0} = - \epsilon^ {4} \frac {V ^ {k}}{r _ {1 2} ^ {3}} \left[ \left(2 m _ {1} M _ {2} ^ {i l} + m _ {2} M _ {1} ^ {i l}\right) \Delta^ {l k} + 2 \left(m _ {1} M _ {2} ^ {l k} + m _ {2} M _ {1} ^ {l k}\right) \Delta^ {l i} \right] \\ = \epsilon^ {4} \frac {m _ {1}}{r _ {1 2} ^ {3}} \left[ 6 (\vec {s} _ {2} \times \vec {n} _ {1 2}) \cdot \vec {V} n _ {1 2} ^ {i} + 4 \vec {s} _ {2} \times \vec {V} - 6 \vec {s} _ {2} \times \vec {n} _ {1 2} (\vec {n} _ {1 2} \cdot \vec {V}) \right] \\ + \epsilon^ {4} \frac {m _ {2}}{r _ {1 2} ^ {3}} \left[ 6 (\vec {s} _ {1} \times \vec {n} _ {1 2}) \cdot \vec {V} n _ {1 2} ^ {i} + 3 \vec {s} _ {1} \times \vec {V} - 3 \vec {s} _ {1} \times \vec {n} _ {1 2} (\vec {n} _ {1 2} \cdot \vec {V}) \right], \tag {221} \\ \end{array}
$$

where $\Delta^{ij} \equiv \delta^{ij} - 3n_{12}^i n_{12}^j$ and $\times$ in this section denotes the outer product for the usual Euclidean spatial three-vectors. The spin vector $s_A^i$ is defined by

$$
s _ {A} ^ {i} \equiv \frac {1}{2} \epsilon_ {i j k} M _ {A} ^ {j k}, \tag {222}
$$

where $\epsilon_{ijk}$ is the totally antisymmetric symbol.

# B.2 Spin-spin coupling force

It is straightforward to derive the spin-spin coupling force. The definition of the center of mass does not change the form of the spin-spin coupling force as expected.

We evaluate the following surface integral,

$$
F _ {\mathrm{1SS}} ^ {i} = - \oint_ {\partial B _ {1}} d S _ {j 1 0} [ (- g) t _ {\mathrm{LL}} ^ {i j} ] + \dots , \tag {223}
$$

with 10 $\left[(-g)t_{\mathrm{LL}}^{ij}\right]=\delta^{ij}_{6}h^{\tau k,l}_{6}h^{\tau}_{[k,l]}+2_{6}h^{\tau[k,i]}_{6}h^{\tau j}_{,k}+2_{6}h^{\tau[k,j]}_{6}h^{\tau i}_{,k}+\ldots$ . The result is

$$
\begin{array}{l} F _ {1 \mathrm{SS}} ^ {i} = \epsilon^ {6} \left[ - \frac {1 5 M _ {1} ^ {j k} M _ {2} ^ {j l} r _ {1 2} ^ {k} r _ {1 2} ^ {l} r _ {1 2} ^ {i}}{r _ {1 2} ^ {7}} + \frac {3 M _ {1} ^ {j k} M _ {2} ^ {j k} r _ {1 2} ^ {i}}{r _ {1 2} ^ {5}} - \frac {3 M _ {1} ^ {i j} M _ {2} ^ {j k} r _ {1 2} ^ {k}}{r _ {1 2} ^ {5}} - \frac {3 M _ {1} ^ {j k} M _ {2} ^ {k i} r _ {1 2} ^ {j}}{r _ {1 2} ^ {5}} \right] \\ = \epsilon^ {6} \frac {1}{r _ {1 2} ^ {4}} \left[ 1 5 (\vec {n} _ {1 2} \cdot \vec {s} _ {1}) (\vec {n} _ {1 2} \cdot \vec {s} _ {2}) n _ {1 2} ^ {i} - 3 s _ {1} ^ {i} (\vec {n} _ {1 2} \cdot \vec {s} _ {2}) - 3 s _ {2} ^ {i} (\vec {n} _ {1 2} \cdot \vec {s} _ {1}) - 3 n _ {1 2} ^ {i} (\vec {s} _ {1} \cdot \vec {s} _ {2}) \right], (2 2 4) \\ \end{array}
$$

which perfectly agrees with the previous result (see e.g. [46, 108, 152]).

# B.3 Quadrupole-orbit coupling force

The quadrupole-orbit coupling force can be derived by evaluating the following integral

$$
F _ {1 \mathrm{QO}} ^ {i} = - \oint_ {\partial B _ {1}} d S _ {j 8} [ (- g) t _ {\mathrm{LL}} ^ {i j} ] + \dots , \tag {225}
$$

with $8[(-g)t_{\mathrm{LL}}^{ij}]=(\delta_{k}^{i}\delta_{l}^{j}+\delta_{k}^{j}\delta_{l}^{i}+-\delta^{ij}\delta_{kl})_{4}h^{\tau\tau,k}_{8}h^{\tau\tau,l}/4+\ldots$ . Then we have the quadrupole-orbit coupling force in the same form as the result in [152],

$$
F _ {1 \mathrm{QO}} ^ {i} = \epsilon^ {4} \frac {3}{2 r _ {1 2} ^ {4}} \left(m _ {1} I _ {2} ^ {\langle k l \rangle} + m _ {2} I _ {1} ^ {\langle k l \rangle}\right) \left(2 \delta^ {i l} n _ {1 2} ^ {k} - 5 n _ {1 2} ^ {i} n _ {1 2} ^ {k} n _ {1 2} ^ {l}\right). \tag {226}
$$

This result agrees formally with the Newtonian quadrupole-orbit coupling force, see Equation (58).

# B.4 Spin geodesic precession

The spin precession can be evaluated using the following equation:

$$
\frac {d M _ {A} ^ {i j}}{d \tau} = - 2 \epsilon^ {- 2} v _ {A} ^ {[ i} P _ {A} ^ {j ]} - 2 \epsilon^ {- 2} R _ {A} ^ {[ i j ]}. \tag {227}
$$

Evaluating the surface integrals in $R_{A}^{ij}$ and $Q_{A}^{i}$ (appearing through the momentum-velocity relation), we have up to $\epsilon^{2}$

$$
\begin{array}{l} \frac {d}{d \tau} \left(M _ {1} ^ {i j} + 2 v _ {1} ^ {[ i} D _ {1} ^ {j ]}\right) = \epsilon^ {2} \frac {m _ {2}}{r _ {1 2}} \left(2 n _ {1 2} ^ {[ i} M _ {1} ^ {j ] k} v _ {1} ^ {k} - 4 v _ {1} ^ {[ i} M _ {1} ^ {j ] k} n _ {1 2} ^ {k} - 4 n _ {1 2} ^ {[ i} M _ {1} ^ {j ] k} v _ {2} ^ {k} + 4 v _ {2} ^ {[ i} M _ {1} ^ {j ] k} n _ {1 2} ^ {k}\right) \\ + \epsilon^ {2} \frac {6 m _ {2}}{r _ {1 2} ^ {3}} n _ {1 2} ^ {k} n _ {1 2} ^ {[ i} I _ {1} ^ {j ] k} \\ + \epsilon^ {2} \frac {2 m _ {2}}{r _ {1 2} ^ {3}} \left(n _ {1 2} ^ {k} n _ {1 2} ^ {i} Z _ {1} ^ {k [ j l ] l} - n _ {1 2} ^ {k} n _ {1 2} ^ {j} Z _ {1} ^ {k [ i l ] l} + n _ {1 2} ^ {k} n _ {1 2} ^ {i} Z _ {1} ^ {j [ k l ] l} - n _ {1 2} ^ {k} n _ {1 2} ^ {i} Z _ {1} ^ {j [ k l ] l}\right) \\ + \mathcal {O} (\epsilon^ {3}). \tag {228} \\ \end{array}
$$

Note that there is no monopole-monopole coupling.

In our formalism, the above form is sufficient since we use $M_{A}^{ij}$ , not the spin vector $s_{A}^{i}$ . To transform the above equation into the usual form, we take a “crude” method; we shall treat one star, say, the star 1 as if it felt only the gravitational field of the companion star. Motivated by the formalism on extended bodies by Dixon [69], we introduce the intrinsic spin four-vector $S_{A\mu}$ and the intrinsic spin tensor $M_{A}^{\mu\nu}$ as

$$
\mathcal {M} _ {A} ^ {\mu \nu} = \frac {2 \epsilon^ {- 6}}{\sqrt {- g}} \int_ {\mathcal {B} _ {A}} d ^ {3} \Sigma_ {\rho} \sigma_ {A} ^ {[ \mu} \Lambda_ {N} ^ {\nu ] \rho}, \tag {229}
$$

$$
\mathcal {S} _ {A \mu} = \frac {1}{2} \epsilon_ {\alpha \rho \sigma \mu} \mathcal {M} _ {A} ^ {\rho \sigma} u _ {A} ^ {\alpha}, \tag {230}
$$

where $\epsilon_{\alpha \rho \sigma \mu}$ is the totally anti-symmetric symbol with $\epsilon_{0123} = 1$ . $d^3\Sigma_\alpha$ is the proper volume element satisfying $d^3\Sigma_{[\alpha}u_{A\beta]} = 0$ . $\mathcal{B}_A$ is a three-sphere surrounding star $A$ whose normal is $u_{A\alpha}$ and whose radius is $\epsilon R_A$ . $\sigma_A^\mu$ is the spacelike four-vector satisfying $g_{\mu \nu}\sigma_A^\mu u_A^\mu = 0$ . The four-momentum is normalized as $g_{\mu \nu}u_A^\mu u_A^\nu = \epsilon^{-2}$ . The above definitions imply the spin supplementary condition

$$
\mathcal {S} _ {A \mu} u _ {A} ^ {\mu} = 0, \quad \text { or   equivalently }, \quad \mathcal {D} _ {A} ^ {\mu} = - \mathcal {M} _ {A} ^ {\mu \nu} u _ {A \mu} = 0, \tag {231}
$$

where $D_{A}^{\mu}$ is the intrinsic dipole moment. Now we construct a coordinate transformation from the near zone $x^{\mu} = (\tau, x^{i})$ to the Fermi normal coordinates $\sigma_{A}^{\hat{\rho}} = (\hat{\tau}, \sigma^{\hat{i}})$ (see e.g. Section 40.7 in [122]),

$$
\sigma_ {A} ^ {\hat {\rho}} = e _ {A \mu} ^ {\hat {\rho}} x ^ {\mu}, \tag {232}
$$

$$
e _ {1 \tau} ^ {\hat {\tau}} = 1 + \frac {1}{2} \epsilon^ {2} v _ {1} ^ {2} - \epsilon^ {2} \frac {m _ {2}}{r _ {1 2}}, \tag {233}
$$

$$
e _ {1 i} ^ {\hat {\tau}} = - \epsilon^ {2} v _ {1 i}, \tag {234}
$$

$$
e _ {1 j} ^ {\hat {\mathrm{i}}} = \left(1 + \epsilon^ {2} \frac {m _ {2}}{r _ {1 2}}\right) \delta_ {j} ^ {i} + \frac {1}{2} \epsilon^ {2} v _ {1} ^ {i} v _ {1 j}. \tag {235}
$$

Then we express the intrinsic spin tensor in the Fermi normal coordinates in terms of the moments in the near zone. Using $d^{3}\Sigma_{\alpha} = -u_{A\alpha}(1 + \epsilon^{2}m_{2}/r_{12} - \epsilon^{2}v_{1}^{2}/2)$ , we have

$$
\mathcal {M} _ {1} ^ {\hat {\mathrm{i}} \hat {\tau}} = \left[ \left(1 + \frac {1}{2} \epsilon^ {2} v _ {1} ^ {2} - \epsilon^ {2} \frac {2 m _ {2}}{r _ {1 2}}\right) \delta^ {i j} - \frac {1}{2} \epsilon^ {2} v _ {1} ^ {i} v _ {1} ^ {j} \right] D _ {1} ^ {j} - \epsilon^ {2} M _ {1} ^ {i k} v _ {1} ^ {k} + \mathcal {O} (\epsilon^ {3}), \tag {236}
$$

$$
\mathcal {M} _ {1} ^ {\hat {i j}} = M _ {1} ^ {i j} + \epsilon^ {2} v _ {1} ^ {k} v _ {1} ^ {[ i} M _ {1} ^ {j ] k} + \mathcal {O} (\epsilon^ {3}). \tag {237}
$$

Now defining the intrinsic center of mass by setting $\mathcal{D}_A^\mu = \mathcal{M}_{A}^{\hat{\mathrm{i}}\hat{\tau}} = 0$ , we obtain

$$
D _ {A} ^ {i} = \epsilon^ {2} M _ {1} ^ {i k} v _ {1} ^ {k}. \tag {238}
$$

Notice that this relation provides the spin-orbit coupling force in the previous form (see Appendix B.1; thus we get $d_{A}^{i} = D_{A}^{i}$ in the present treatment). Using Equation (228), we finally obtain the spin geodesic equation in the usual form (we omit the quadrupole and $Z_{1}^{ijkl}$ term for simplicity),

$$
\frac {d \vec {\mathcal {S}} _ {1}}{d \tau} = \epsilon^ {2} \frac {m _ {2}}{r _ {1 2} ^ {2}} \left[ \left(2 \vec {v} _ {2} - \frac {3}{2} \vec {v} _ {1}\right) \times \vec {n} _ {1 2} \right] \times \vec {\mathcal {S}} _ {1} + \mathcal {O} (\epsilon^ {3}), \tag {239}
$$

where $\vec{S}_{1}$ is the spatial part of the intrinsic spin four-vector $S_{1\hat{i}} = \epsilon_{ijk} M_{1}^{jk}/2$ . Equation (239) is the geodesic precession equation, or called the de Sitter-Fokker precession (see e.g. Section 40.7 in [122] and [36, 145]).

# B.5 Remarks

We have derived the spin-orbit coupling force, the spin-spin coupling force, and the quadrupole-orbit coupling force to the lowest order. The spin geodesic equation could also be derived in a cruder way. Our results agree with the previous results modulo the definition of the center of mass. It should be noted, however, that we have defined our multipole moments $M_{A}^{ij}$ and $I_{A}^{ij}$ as a volume integral of $\Lambda_{N}^{\mu\nu}$ which manifestly includes the effect of strong internal gravity of the star. Thus, our results support the applicability of the spin-orbit, spin-spin, and quadrupole-orbit coupling forces to a relativistic compact binary where the component stars have a strong internal gravitational field. On the effect of these multipole moments in the orbital evolution and/or the gravitational waveform (see e.g. [6, 7, 8, 9, 15, 44, 108, 109, 110, 112, 128, 132, 148]).

# C A Generalized Equivalence Principle Including The Emission of Gravitational Wave

In this section, we give a simple and divergence free derivation of the equations of motion for a small compact object with mass m around any sort of massive body, following the paper by Fukumoto and co-authors $[78]$ . The object moves along the geodesic determined by the smooth part of the geometry around the object up to order m. We note that the smooth part includes the gravitational waves emitted by the orbital motion of the object. Thus it generalizes the equivalence principle for such a compact object to the emission of gravitational waves. Also our derivation of the equations of motion provides a useful method to explicitly calculate the radiation reaction in the fast motion of a compact body.

# C.1 Introduction

The idea that an apple and the Moon fall with the same acceleration led Newton to the discovery of the law of gravity. This is not trivial because the Moon is self-gravitating and an apple is not. That any test particle moves on a geodesic of an external gravitational field is called the weak equivalence principle (WEP). Including self-gravitating objects in this statement is leads to the strong equivalence principle (SEP). The SEP is now experimentally verified with an accuracy of better than $1.5 \times 10^{-13}$ [4]. The verification of WEP and SEP is regarded as one of the most important experiments in physics because it plays a fundamental role in any theory of gravity. In fact, Einstein regarded the WEP as the starting point of his theory of gravity. However it is not obvious that a theory constructed that way respects the SEP, and there have been investigations to confirm this. It is then natural to ask how far one can generalize the principle within the framework of general relativity and other theories of gravity.

This is not only of academic interest but also has practical importance. There are already several detectors of gravitational waves around the world and we are expecting to directly detect gravitational waves from astronomical sources in the near future and hopefully to open a new window to the universe by gravitational waves. In order to use gravitational waves as a practical tool in astronomy, it is definitely necessary to have a good understanding of the equations of motion for systems with more general situations such as small compact objects like a neutron star/black hole moving at an arbitrary speed in an arbitrary external field. In such a situation the perturbation of the external field including gravitational waves generated by the orbital motion is not negligible. This is exactly the situation we have in mind here and for which we would like to generalize the equivalence principle. In this respect it should be mentioned that Mino et al. and others derived the equations of motion for a point particle with mass m which is represented by a Dirac delta distribution source in an arbitrary background. The equation is interpreted as the geodesic equation on the geometry determined by the external field and the so-called tail part of the self-field of the particle in the first order in m $[67, 121, 133]$ . Furthermore, Mino et al. used another approach, the matched asymptotic expansion, to obtain the equations of motion without employing the concept of a point particle and thus avoiding divergences in their derivation.

We avoid using a singular source and make use of the point particle limit to derive directly the geodesic equations on the smooth part of the geometry around the object. Here the point particle limit is the strong field point particle limit $[81]$ . The smooth part includes the gravitational waves emitted by the orbital motion of the object, and thus the equivalence principle is generalized to including the emission of gravitational waves. We believe that our approach simplifies the proof that the Mino–Sasaki–Tanaka equations of motion are applicable to a nonsingular source where Mino et al. used the matched asymptotic expansion for their proof.

# C.2 Equations of motion

Let us start explaining again our situation and discuss the strong field point particle limit [81]. A small spherical compact object with mass $m$ moves at an arbitrary speed around a massive body with mass $M$ . We would like to find the equations of motion for the object including radiation reaction. We assume that the object is stationary except for higher order tidal effects, so that we can safely neglect the emission of gravitational waves from the object itself, but of course we cannot neglect the gravitational waves emitted by the orbital motion of the object. We denote the world line of the center of mass of the object as $z^{\mu}(\tau)$ and define the body zone of the object as follows. We imagine a spherical region around $z^{\mu}(\tau)$ and a radius that scales as $\epsilon$ . At the same time we scale the linear dimension of the object as $\epsilon^2$ so that the boundary of the body zone is located at the far zone of the object. Namely we are able to have a multipole expansion of the field generated by the object at the surface of the body zone. We also implicitly assume that the mass of the object scales as $\epsilon$ so that the compactness of the object remains constant in the point particle limit as $\epsilon \to 0$ . This is why we call this limit the strong field point particle limit. Then we calculate the metric perturbation induced by the small object in this limit. The smallness parameter $\epsilon$ has a dimension of length which characterizes the smallness of the object. One may regard it as the ratio between the physical scale of the object and the characteristic scale of the background curvature. We assume that the background metric $g_{\mu\nu}$ satisfies the Einstein equations in vacuum. So the Ricci tensor of the background vanishes. Since we have assumed that the mass scale of the small object is much smaller than the scale of the gravitational field of the background geometry, we approximate the metric perturbation by the linear perturbation of the small particle $h_{\mu\nu}$ .

We will work in the harmonic gauge,

$$
\bar {h} _ {\nu} ^ {\mu \nu} = 0, \tag {240}
$$

where the semicolon means that the covariant derivative with respect to the background metric and the trace-reversed variable is defined as usual,

$$
\bar {h} _ {\mu \nu} = h _ {\mu \nu} - \frac {1}{2} g _ {\mu \nu} g ^ {\rho \lambda} h _ {\rho \lambda}. \tag {241}
$$

Then the linearized Einstein equations take the following form:

$$
- \frac {1}{2} \bar {h} ^ {\mu \nu ; \xi} _ {\xi} (x) - R ^ {\mu} _ {\xi} ^ {\nu} _ {\rho} (x) \bar {h} ^ {\xi \rho} (x) = 8 \pi T ^ {\mu \nu} (x). \tag {242}
$$

This can be solved formally as follows,

$$
\bar {h} ^ {\mu \nu} (x) = 8 \pi \int d ^ {4} y \sqrt {- g} G _ {\alpha \beta} ^ {\mu \nu} (x, y) T ^ {\alpha \beta} (y), \tag {243}
$$

where we have used the retarded tensor Green's function defined by

$$
G ^ {\mu \nu \alpha \beta} (x, y) = \frac {1}{4 \pi} \theta (\Sigma (x), y) \left[ u ^ {\mu \nu \alpha \beta} (x, y) \delta (\sigma (x, y)) + v ^ {\mu \nu \alpha \beta} (x, y) \theta (- \sigma (x, y)) \right]. \tag {244}
$$

For the general tensor Green's function and the definition of $\sigma(x,y)$ , $\bar{g}^{\mu\alpha}(x,y)$ , $u^{\mu\nu\alpha\beta}(x,y)$ , $v^{\mu\nu\alpha\beta}(x,y)$ , and $\Sigma(x)$ , please refer to Mino et al. [121] and DeWitt and Brehme [68].

Now we take the point particle limit. In this limit the above metric perturbation contains terms of different $\epsilon$ dependence which makes the calculation of the equations of motion simple. For example, terms with negative powers of $\epsilon$ appear from the Dirac delta distribution part of the Green's function:

$$
{ } _ { s } \bar { h } ^ { \mu \nu } ( x ) = 2 \int d ^ { 4 } y \sqrt { - g } u _ { \alpha \beta } ^ { \mu \nu } ( x , y ) \delta ( \sigma ( x , y ) ) T ^ { \alpha \beta } ( y ) . ( 2 4 5 )
$$

As explained below we only need the field on the boundary of the body zone which is the far zone of the body itself, and thus we may make use of the multipole expansion for the field. For this purpose we choose our coordinate system as follows,

$$
y ^ {\alpha} = z ^ {\alpha} (\tau) + \delta y ^ {\alpha}, \tag {246}
$$

$$
z ^ {0} (\tau) = \tau , \tag {247}
$$

$$
\delta y ^ {0} = 0, \tag {248}
$$

where $z^{\alpha}(\tau)$ is the world line of the center of the object. The center is assumed to be always inside the body in the point particle limit, and thus there is no ambiguity for the choice of the center. In this coordinate system the volume element satisfies $d^{4}y = d\tau d^{3}\delta y$ . Thus we have

$$
{ } _ { s } \bar { h } ^ { \mu \nu } ( x ) = 2 \int d ^ { 3 } \delta y \sqrt { - g } u ^ { \mu \nu } { } _ { \alpha \beta } ( x , z ( \tau _ { y } ) + \delta y ) \frac { 1 } { \dot { \sigma } | _ { \tau = \tau _ { y } } } T ^ { \alpha \beta } ( y ) , \tag {249}
$$

where $\tau_{y}$ is the retarded time of each point y. Then the multipole expansion is obtained by expanding the above expression at the retarded time of the center of the object $\tau_{z}$ defined by $\sigma(x,z(\tau_{z}))=0$ . This can be easily done by noticing the condition $\sigma(x,z(\tau_{y})+\delta y)=0$ . Then the difference between $\tau_{z}$ and $\tau_{y}$ is given by

$$
\delta \tau = - \frac {\sigma_ {; \alpha} (x , z (\tau_ {z})) \delta y ^ {\alpha}}{\dot {\sigma} (x , z (\tau_ {z}))} + \mathcal {O} (\delta y ^ {2}). \tag {250}
$$

Using this $\delta\tau$ we can expand $u^{\mu\nu}_{\alpha\beta}(x,z(\tau_{y})+\delta y)$ around $\tau_{z}$ . In principle we can calculate arbitrarily high multipole moments in this way. Here we only calculate the leading term. Then we only need $u^{\mu\nu}_{\alpha\beta}(x,z(\tau_{y})+\delta y)=u^{\mu\nu}_{\alpha\beta}(x,z(\tau_{z}))+\mathcal{O}(\delta y)$ , and $\dot{\sigma}(x,z(\tau_{y})+\delta y)=\dot{\sigma}(x,z(\tau_{z}))+\mathcal{O}(\delta y)$ . By defining the mass as follows,

$$
m \dot {z} ^ {\alpha} (\tau_ {z}) \dot {z} ^ {\beta} (\tau_ {z}) = \int d ^ {3} \delta y \sqrt {- g} T ^ {\alpha \beta} (y), \tag {251}
$$

we finally obtain the following expression for $_{s}\bar{h}^{\mu\nu}$ :

$$
{ } _ { s } \bar { h } ^ { \mu \nu } ( x ) = \frac { 2 m } { \dot { \sigma } ( x , z ( \tau _ { z } ) ) } u ^ { \mu \nu } { } _ { \alpha \beta } ( x , z ( \tau _ { z } ) ) \dot { z } ^ { \alpha } ( \tau _ { z } ) \dot { z } ^ { \beta } ( \tau _ { z } ) . \tag {252}
$$

Now we derive the equations of motion using this expression. First we define the $\epsilon$ dependent four-momentum of the object as the volume integral of the effective stress-energy tensor $\Theta^{\mu \nu}$ over the body zone $B(\tau)$ ,

$$
P ^ {\mu} (\tau) = - \int_ {B (\tau)} \Theta^ {\mu \nu} d \Sigma_ {\nu}. \tag {253}
$$

Since the effective stress-energy tensor satisfies the conservation law $\Theta^{\mu\nu},_{\nu}=0$ , the change of the four-momentum defined above may be expressed as the surface integral over the boundary of the body zone $\partial B$ ,

$$
\frac {d P ^ {\mu}}{d \tau} = - \oint_ {\partial B} d \Omega \epsilon^ {2} n _ {\nu} (1 + \epsilon a \cdot n) (- g) t _ {\mathrm{LL}} ^ {\mu \nu}, \tag {254}
$$

where $n^{\mu}$ is the unit normal to the surface and $a^{\mu} = du^{\mu}/d\tau$ is the four-acceleration. The equations of motion are obtained by taking the point particle limit of $\epsilon \rightarrow 0$ . Thus we need to calculate $(-g)t_{\mathrm{LL}}^{\mu\nu}$ on the body zone boundary (field points) on which the multipole expansion of the self-field and the Taylor expansion of the nonsingular part of the field are available, and to pick up only terms of order $\epsilon^{-2}$ in the expression. All the other terms but one term, which is proportional

to $m^{2}a^{\mu}/\epsilon$ and can be renormalized to the mass of the small object, vanish in the limit or the angular integration. Remembering that the LL tensor is bilinear in the Christoffel symbols and the Christoffel symbols are the derivatives of the metric tensor, one may realize that the only remaining terms come from the combination of the 0th order of the smooth part of the metric and $1/\epsilon$ part of the self-field which is given by Equation (252). The remaining part of the self-field is the so-called tail part that is regular at the object which is the only relevant point in our calculation. The field point $(x,\tau_{x})$ is now on the body zone boundary, which is defined by $\sqrt{2\sigma(x,z(\tau_{x}))}=\epsilon$ and

$$
\left[ \sigma_ {; \alpha} (x, z (\tau)) \dot {z} ^ {\alpha} (\tau) \right] _ {\tau = \tau_ {x}} = 0. \tag {255}
$$

Using this expression we have the following result in the point particle limit,

$$
\frac {d P ^ {\mu}}{d \tau} = - m \Gamma_ {\alpha \beta} ^ {\mu} (g _ {\mathrm{s}}) u ^ {\alpha} u ^ {\beta} - m \frac {1}{2} \Gamma_ {\alpha \beta} ^ {\alpha} (g _ {\mathrm{s}}) u ^ {\beta} u ^ {\mu}, \tag {256}
$$

where the Christoffel symbols here are calculated in terms of the smooth part of the metric $g_{s}$ . Then the ADM mass is related to the four-momentum as follows, which is supported by the higher order post-Newtonian approximation [91, 95, 91]:

$$
P ^ {\mu} = \sqrt {- g _ {\mathrm{s}}} m u ^ {\mu}. \tag {257}
$$

Finally we have

$$
\frac {d u ^ {\mu}}{d \tau} = - \Gamma_ {\alpha \beta} ^ {\mu} (g _ {\mathrm{s}}) u ^ {\alpha} u ^ {\beta}, \tag {258}
$$

which is the geodesic equation on the geometry determined by the smooth part of the metric around the compact object.

In fact, the spin effect on the equations of motion can be derived in a similar way and the standard result [152] can be obtained.

We have proved that a small compact object moves on the geodesic determined by only the smooth part of the geometry around the object. Thus the equations of motion are automatically obtained by determining the geometry around the object which is of course an implicit functional of the world line of the object. The smooth part contains the gravitational waves emitted by the orbital motion so that this equation includes the damping force due to radiation reaction. Our method avoids using a singular source in the first place by making use of the strong field point particle limit. All the quantities should be evaluated at the surface of the body zone boundary and thus we only need the dependence of the distance from the center of the object, namely the $\epsilon$ dependence of the field. In this way we are able to avoid using any divergent quantities in any part of our calculation. This strongly suggests that our method may be used to get unique equations of fast motion with radiation reaction. This will be investigated in future publications.

In this section, we have assumed spherical symmetry of the compact object except for the tidal effect. It is straightforward to generalize the case to multipole moments in our formalism. This will also be studied in future publications.

# References

[1] Abramovici, A., Althouse, W.E., Drever, R.W.P., Gürsel, Y., Kawamura, S., Raab, F.J., Shoemaker, D.H., Sievers, L., Spero, R.E., Thorne, K.S., Vogt, R.E., Weiss, R., Whitcomb, S.E., and Zucker, M.E., “LIGO: The Laser Interferometer Gravitational-Wave Observatory”, Science, 256, 325–333, (1992). 1.1   
[2] Ajith, P., Iyer, B.R., Robinson, C.A.K., and Sathyaprakash, B.S., “Erratum: A new class of post-Newtonian approximants to the dynamics of inspiralling compact binaries: Test-mass in the Schwarzschild spacetime”, Phys. Rev. D, 72, 049902, (2005). 1.1   
[3] Ajith, P., Iyer, B.R., Robinson, C.A.K., and Sathyaprakash, B.S., “A new class of post-Newtonian approximants to the dynamics of inspiralling compact binaries: Test-mass in the Schwarzschild spacetime”, Phys. Rev. D, 71, 044029, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0412033. 1.1   
[4] Anderson, J.D., and Williams, J.G., “Long-range tests of the equivalence principle”, Class. Quantum Grav., 18, 2447–2456, (2001). C.1   
[5] Anderson, J.L., and DeCanio, T.C., “Equations of hydrodynamics in general relativity in the slow motion approximation”, Gen. Relativ. Gravit., 6, 197–237, (1975). 1.1, 2.3   
[6] Apostolatos, T.A., “Search templates for gravitational waves from precessing, inspirating binaries”, Phys. Rev. D, 52, 605–620, (1995). B.5   
[7] Apostolatos, T.A., “Construction of a template family for the detection of gravitational waves from coalescing binaries”, Phys. Rev. D, 54, 2421–2437, (1996). B.5   
[8] Apostolatos, T.A., “The Influence of spin spin coupling on inspirating compact binaries with $M_{1}=M_{2}$ and $S_{1}=S_{2}$ ”, Phys. Rev. D, 54, 2438–2441, (1996). B.5   
[9] Apostolatos, T.A., Cutler, C., Sussman, G.J., and Thorne, K.S., “Spin induced orbital precession and its modulation of the gravitational wave forms from merging binaries”, Phys. Rev. D, 49, 6274–6297, (1994). B.5   
[10] Arun, K.G., Iyer, B.R., Sathyaprakash, B.S., and Sundararajan, P.A., “Parameter estimation of inspiralling compact binaries using 3.5 post-Newtonian gravitational wave phasing: The nonspinning case”, Phys. Rev. D, 71, 084008, 1–16, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0411146. 1.1   
[11] Asada, H., and Futamase, T., “Post-Newtonian Approximation”, Prog. Theor. Phys. Suppl., 128, 123–181, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9806108. 3   
[12] Asada, H., and Futamase, T., “Propagation of gravitational waves from slow motion sources in a Coulomb type potential”, Phys. Rev. D, 56, 6062–6066, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9711009. 1.1, 2.3   
[13] Ashby, N., and Bertotti, B., “Relativistic effects in local inertial frames”, Phys. Rev. D, 34, 2246–2259, (1986). 3.3, 4.3, B

[14] Bel, L., Deruelle, N., Damour, T., Ibañez, J., and Martin, J., “Poincaré-Invariant Gravitational Field and Equations of Motion of two Pointlike Objects: The Postlinear Approximation of General Relativity”, Gen. Relativ. Gravit., 13, 963–1004, (1981). 1.2   
[15] Bildsten, L., and Cutler, C., “Tidal interactions of inspirating compact binaries”, Astrophys. J., 400, 175–180, (1992). 3.1, B.5   
[16] Blanchet, L., “Gravitational Radiation from Relativistic Sources”, in Marck, J.-A., and Lasota, J.P., eds., Relativistic Gravitation and Gravitational Radiation, Proceedings of the Les Houches School of Physics, held in Les Houches, Haute Savoie, France 26 September–6 October, 1995, 33–66, (Cambridge University Press, Cambridge, U.K., 1995). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9607025.3   
[17] Blanchet, L., “Gravitational radiation reaction and balance equations to post-Newtonian order”, Phys. Rev. D, 55, 714–732, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9609049. 1.2   
[18] Blanchet, L., “Post-Newtonian Gravitational Radiation”, in Schmidt, B.G., ed., Einstein’s Field Equations and Their Physical Implications: Selected Essays in Honour of Jürgen Ehlers, vol. 540 of Lecture Notes in Physics, 225–271, (Springer, Berlin, Germany; New York, U.S.A., 2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0004012.3   
[19] Blanchet, L., “Gravitational Radiation from Post-Newtonian Sources and Inspiralling Compact Binaries”, Living Rev. Relativity, 9, lrr-2006-4, (2006). URL (cited on 3 August 2006): http://www.livingreviews.org/lrr-2006-4. 2.2, 3   
[20] Blanchet, L., and Damour, T., “Tail-transported temporal correlations in the dynamics of a gravitating system”, Phys. Rev. D, 37, 1410–1435, (1988). 2.2, 4.1   
[21] Blanchet, L., and Damour, T., “Post-Newtonian generation of gravitational waves”, Ann. Inst. Henri Poincare A, 50, 377–408, (1989). 1.1   
[22] Blanchet, L., Damour, T., and Esposito-Farèse, G., “Dimensional regularization of the third post-Newtonian dynamics of point particles in harmonic coordinates”, Phys. Rev. D, 69, 124007, 1–51, (2004). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0311052. 1.1, 1.2, 1.2, 11, 8.5, 8.6   
[23] Blanchet, L., Damour, T., Esposito-Farèse, G., and Iyer, B.R., “Gravitational radiation from inspiralling compact binaries completed at the third post-Newtonian order”, Phys. Rev. Lett., 93, 091101, (2004). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0406012. 1.1   
[24] Blanchet, L., Damour, T., Esposito-Farèse, G., and Iyer, B.R., “Dimensional regularization of the third post-Newtonian gravitational wave generation from two point masses”, Phys. Rev. D, 71, 124004, 1–36, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0503044. 1.1   
[25] Blanchet, L., and Faye, G., “Equations of motion of point-particle binaries at the third post-Newtonian order”, Phys. Lett. A, 271, 58–64, (2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0004009. 1.2, 1.2, 8.5, 8.6, 15

[26] Blanchet, L., and Faye, G., “Hadamard regularization”, J. Math. Phys., 41, 7675–7714, (2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0004008. 1.2, 6.1, 8.6, 15   
[27] Blanchet, L., and Faye, G., “General relativistic dynamics of compact binaries at the third post-Newtonian order”, Phys. Rev. D, 63, 062005, 1–43, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0007051. 1.2, 1.2, 5, 5.1, 8.2, 8.4, 8.5, 8.5, 8.5, 8.6, 15   
[28] Blanchet, L., and Faye, G., “Lorentzian regularization and the problem of point-like particles in general relativity”, J. Math. Phys., 42, 4391–4418, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0006100. 1.2, 4.2.2, 8.4   
[29] Blanchet, L., Faye, G., Iyer, B.R., and Joguet, B., “Gravitational-wave inspiral of compact binary systems to 7/2 post-Newtonian order”, Phys. Rev. D, 65, 061501, 1–5, (2002). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0105099. 1.1   
[30] Blanchet, L., Faye, G., and Ponsot, B., “Gravitational field and equations of motion of compact binaries to 5/2 post-Newtonian order”, Phys. Rev. D, 58, 124002, 1–20, (1998). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9804079. 1.2, 1.2, 4.1, 5, 5.1, 6.1   
[31] Blanchet, L., and Iyer, B.R., “Hadamard regularization of the third post-Newtonian gravitational wave generation of two point masses”, Phys. Rev. D, 71, 024004, 1–20, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0409094. 1.1   
[32] Blanchet, L., Iyer, B.R., and Joguet, B., “Gravitational waves from inspiralling compact binaries: Energy flux to third post-Newtonian order”, Phys. Rev. D, 65, 064005, 1–41, (2002). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0105098. 1.1   
[33] Blanchet, L., and Schäfer, G., “Gravitational wave tails and binary star systems”, Class. Quantum Grav., 10, 2699–2721, (1993). 1.1   
[34] Blandford, R., and Teukolsky, S.A., “Arrival-time analysis for a pulsar in a binary system”, Astrophys. J., 205, 580–591, (1976). 1.1   
[35] Bradaschia, C., Del Fabbro, R., Di Virgilio, A., Giazotto, A., Kautzky, H., Montelatici, V., Passuello, D., Brillet, A., Cregut, O., Hello, P., Man, C.N., Manh, P.T., Marraud, A., Shoemaker, D.H., Vinet, J.-Y., Barone, F., di Fiore, L., Milano, L., Russo, G., Aguirregabiria, J.M., Bel, H., Duruisseau, J.P., Le Denmat, G., Tourenc, P., Capozzi, M., Longo, M., Lops, M., Pinto, I., Rotoli, G., Damour, T., Bonazzola, S., Marck, J.-A., Gourghoulon, Y., Holloway, L.E., Fuligni, F., Iafolla, V., and Natale, G., “The VIRGO Project: A wide band antenna for gravitational wave detection”, Nucl. Instrum. Methods A, 289, 518–525, (1990). 1.1   
[36] Brumberg, V.A., Essential Relativistic Celestial Mechanics, (Adam Hilger, Bristol, U.K.; Philadelphia, U.S.A., 1991). B.4   
[37] Brumberg, V.A., and Kopeikin, S.M., “Relativistic Reference Systems and Motion of Test Bodies in the Vicinity of the Earth”, Nuovo Cimento B, 103, 63–98, (1989). B

[38] Burgay, M., D'Amico, N., Possenti, A., Manchester, R.N., Lyne, A.G., Joshi, B.C., McLaughlin, M.A., Kramer, M., Sarkissian, J.M., Camilo, F., Kalogera, V., Kim, C., and Lorimer, D.R., "An increased estimate of the merger rate of double neutron stars from observations of a highly relativistic system", Nature, 426, 531–533, (2003). 1.1   
[39] Burke, W.L., “Gravitational Radiation Damping of Slowly Moving Systems Calculated Using Matched Asymptotic Expansions”, J. Math. Phys., 12, 401–418, (1971). 1.1   
[40] Chandrasekhar, S., “The Post-Newtonian Equations of Hydrodynamics in General Relativity”, Astrophys. J., 142, 1488–1540, (1965). 1.1   
[41] Chandrasekhar, S., “Conservation Laws in General Relativity and in the Post-Newtonian Approximations”, Astrophys. J., 158, 45, (1969). 1.1   
[42] Chandrasekhar, S., and Esposito, F.P., “The $2\frac{1}{2}$ -Post-Newtonian Equations of Hydrodynamics and Radiation Reaction in General Relativity”, Astrophys. J., 160, 153–179, (1970). 1.1   
[43] Chandrasekhar, S., and Nutku, Y., “The Second Post-Newtonian Equations of Hydrodynamics in General Relativity”, Astrophys. J., 158, 55–79, (1969). 1.1   
[44] Cutler, C., Apostolatos, T.A., Bildsten, L., Finn, L.S., Flanagan, É.É., Kennefick, D., Marković, D.M., Ori, A., Poisson, E., and Sussman, G.J., “The Last Three Minutes: Issues in Gravitational-Wave Measurements of Coalescing Compact Binaries”, Phys. Rev. Lett., 70, 2984–2987, (1993). 1.1, 3.1, B.5   
[45] Cutler, C., and Thorne, K.S., “An Overview of Gravitational-Wave Sources”, in Bishop, N.T., and Maharaj, S.D., eds., General Relativity and Gravitation, Proceedings of the 16th International Conference on General Relativity and Gravitation, Durban, South Africa, 15–21 July, 2001, 72–111, (World Scientific, Singapore; River Edge, U.S.A., 2002). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0204090. 1.1   
[46] Damour, T., “Problème des deux corps et freinage de rayonnement en relativité générale”, C. R. Acad. Sci. Ser. II, 294, 1355–1357, (1982). 1.1, 1.2, 4.4, B, B.1, B.2   
[47] Damour, T., “Gravitational radiation and the motion of compact bodies”, in Deruelle, N., and Piran, T., eds., Gravitational Radiation, NATO Advanced Study Institute, Centre de Physique des Houches, France, 2–21 June, 1982, 59–144, (North-Holland; Elsevier, Amsterdam, Netherlands; New York, U.S.A., 1983). 1.1, 1.2, 1.2, 3, 3.1   
[48] Damour, T., “An Introduction to the Theory of Gravitational Radiation”, in Carter, B., and Hartle, J.B., eds., Gravitation in Astrophysics: Cargèse 1986, Proceedings of a NATO Advanced Study Institute on Gravitation in Astrophysics, Cargèse, France, 15–31 July, 1986, vol. 156 of NATO ASI Series B, 3–62, (Plenum Press, New York, U.S.A., 1987). 1.1, 3   
[49] Damour, T., “The problem of motion in Newtonian and Einsteinian gravity”, in Hawking, S.W., and Israel, W., eds., Three Hundred Years of Gravitation, 128–198, (Cambridge University Press, Cambridge, U.K.; New York, U.S.A., 1987). 1.1, 3   
[50] Damour, T., and Deruelle, N., “Lagrangien généralisé du système de deux masses ponctuelles, à l’approximation post-post-newtonienne de la relativité générale”, C. R. Acad. Sci. Ser. II, 293, 537–540, (1981). 1.2

[51] Damour, T., and Deruelle, N., “Lois de conservation d’un système de deux masses ponctuelles en relativité générale”, C. R. Acad. Sci. Ser. II, 293, 877–880, (1981). 1.2   
[52] Damour, T., and Deruelle, N., “Radiation reaction and angular momentum loss in small angle gravitational scattering”, Phys. Lett. A, 87, 81–84, (1981). 1.2   
[53] Damour, T., Jaranowski, P., and Schäfer, G., “Poincaré invariance in the ADM Hamiltonian approach to the general relativistic two-body problem”, Phys. Rev. D, 62, 021501, 1–5, (2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0003051. Erratum: Phys. Rev. D 63 (2001) 029903. 1.2   
[54] Damour, T., Jaranowski, P., and Schäfer, G., “Dimensional regularization of the gravitational interaction of point masses”, Phys. Lett. B, 513, 147–155, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0105038. 1.2, 1.2, 8.5, 8.6   
[55] Damour, T., Jaranowski, P., and Schäfer, G., “Equivalence between the ADM-Hamiltonian and the harmonic-coordinates approaches to the third post-Newtonian dynamics of compact binaries”, Phys. Rev. D, 63, 044021, 1–11, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0010040. Erratum: Phys. Rev. D 63 (2001) 044021. 1.2   
[56] Damour, T., and Schäfer, G., “Lagrangians for n Point Masses at the Second Post-Newtonian Approximation of General Relativity”, Gen. Relativ. Gravit., 17, 879–905, (1985). 1.2   
[57] Damour, T., Soffel, M., and Xu, C., “General-relativistic celestial mechanics. I. Method and definition of reference systems”, Phys. Rev. D, 43, 3273–3307, (1991). B   
[58] Damour, T., Soffel, M., and Xu, C., “General-relativistic celestial mechanics. II. Translational equations of motion”, Phys. Rev. D, 45, 1017–1044, (1992). B   
[59] Damour, T., Soffel, M., and Xu, C., “General-relativistic celestial mechanics. III. Rotational equations of motion”, Phys. Rev. D, 47, 3124–3135, (1993). B   
[60] Damour, T., Soffel, M., and Xu, C., “General-relativistic celestial mechanics. IV. Theory of satellite motion”, Phys. Rev. D, 49, 618–635, (1994). B   
[61] Damour, T., and Taylor, J.H., “On the orbital period change of the binary pulsar PSR 1913+16”, Astrophys. J., 366, 501–511, (1991). 1.1   
[62] Danzmann, K. et al., “The GEO-Project. A Long-Baseline Laser Interferometer for the Detection of Gravitational Waves”, in Ehlers, J., and Schäfer, G., eds., Relativistic Gravity Research with Emphasis on Experiments and Observations, Proceedings of the 81 WE-Heraeus-Seminar held at the Physikzentrum, Bad Honnef, Germany, 2–6 September, 1991, vol. 410 of Lecture Notes in Physics, 184–209, (Springer, Berlin, Germany; New York, U.S.A., 1992). 1.1   
[63] Dautcourt, G., “Post-Newtonian extension of the Newton-Cartan theory”, Class. Quantum Grav., 14, A109–A118, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9610036. 2   
[64] de Andrade, V.C., Blanchet, L., and Faye, G., “Third post-Newtonian dynamics of compact binaries: Noetherian conserved quantities and equivalence between the harmonic coordinate and ADM-Hamiltonian formalisms”, Class. Quantum Grav., 18, 753–778, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0011063. 1.2

[65] D'Eath, P.D., “Dynamics of a small black hole in a background universe”, Phys. Rev. D, 11, 1387–1403, (1975). 3.1   
[66] D'Eath, P.D., "Interaction of two black holes in the slow-motion limit", Phys. Rev. D, 12, 2183-2199, (1975). 1.2, 3.1   
[67] Detweiler, S., and Whiting, B.F., “Self-force via a Green’s function decomposition”, Phys. Rev. D, 67, 024025, (2003). Related online version (cited on 23 March 2006): http://arXiv.org/abs/gr-qc/0202086. C.1   
[68] DeWitt, B.S., and Brehme, R.W., “Radiation Damping in a Gravitational Field”, Ann. Phys. (N.Y.), 9, 220–259, (1960). C.2   
[69] Dixon, W.G., “Extended bodies in general relativity: Their description and motion”, in Ehlers, J., ed., Isolated Gravitating Systems in General Relativity (Sistemi gravitazionali isolati in relatività generale), Proceedings of the International School of Physics “Enrico Fermi”, Course 67, Varenna on Lake Como, Villa Monastero, Italy, 28 June–10 July, 1976, 156–219, (North-Holland, Amsterdam, Netherlands; New York, U.S.A., 1979). 3.1, B.4   
[70] Ehlers, J., “Examples of Newtonian limits of relativistic spacetimes”, Class. Quantum Grav., 14, A119–A126, (1997). 2   
[71] Ehlers, J., Rosenblum, A., Goldberg, J.N., and Havas, P., “Comments on gravitational radiation damping and energy loss in binary systems”, Astrophys. J. Lett., 208, L77–L81, (1976). 1.1   
[72] Einstein, A., “Explanation of the Perihelion Motion of Mercury from the General Theory of Relativity”, Sitzungsber. Preuss. Akad. Wiss., 1915, 831–839, (1915). 1.1   
[73] Einstein, A., Infeld, L., and Hoffmann, B., “The Gravitational Equations and the Problem of Motion”, Ann. Math., 39, 65–100, (1938). 1.1, 1.2, 3.1, 3.2, 4.5.2   
[74] Epstein, R., “The binary pulsar: Post-Newtonian timing effects”, Astrophys. J., 216, 92–100, (1977). Related online version (cited on 3 August 2006): http://adsabs.harvard.edu/abs/1977ApJ...216...92E. 1.1   
[75] Finn, L.S., “Binary inspiral, gravitational radiation, and cosmology”, Phys. Rev. D, 53, 2878–2894, (1996). 1.1   
[76] Fock, V.A., “On motion of finite masses in general relativity”, J. Phys. (Moscow), 1(2), 81–116, (1939). 1.2   
[77] Fock, V.A., Theory of space, time and gravitation, (Pergamon Press, London, U.K., 1959). 1.1, 4.1, 4.7   
[78] Fukumoto, T., Futamase, T., and Itoh, Y., “On the Equation of Motion for a Fast Moving Small Object in the Strong Field Point Particle Limit”, Prog. Theor. Phys., 116, 423–428, (2006). Related online version (cited on 5 July 2006): http://arXiv.org/abs/gr-qc/0606114. C   
[79] Futamase, T., “Gravitational radiation reaction in the Newtonian limit”, Phys. Rev. D, 28, 2373–2381, (1983). 1.1, 2.2, 3.3, 4.1   
[80] Futamase, T., “Point particle limit and the far zone quadrupole formula in general relativity”, Phys. Rev. D, 32, 2566–2574, (1985). 3.3, 3.3

[81] Futamase, T., “The strong field point particle limit and the equations of motion in the binary system”, Phys. Rev. D, 36, 321–329, (1987). 1.2, 3.1, 3.3, 4.2.1, C.1, C.2   
[82] Futamase, T., and Schutz, B.F., “Newtonian and post-Newtonian approximation are asymptotic to general relativity”, Phys. Rev. D, 28, 2363–2372, (1983). 1.1, 2, 2.2, 3.3   
[83] MPI for Gravitational Physics (Albert Einstein Institute), “GEO 600: The German-British Gravitational Wave Detector”, project homepage. URL (cited on 7 March 2006): http://geo600.aei.mpg.de.1.1   
[84] Geroch, R., “Limits of Spacetimes”, Commun. Math. Phys., 13, 180–193, (1969). 2.1   
[85] Gopakumar, A., Iyer, B.R., and Iyer, S., “Second post-Newtonian gravitational radiation reaction for two-body systems: Nonspinning bodies”, Phys. Rev. D, 55, 6030–6053, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9703075. 1.2   
[86] Gopakumar, A., Iyer, B.R., and Iyer, S., “Erratum: Second post-Newtonian gravitational radiation reaction for two-body systems: Nonspinning bodies”, Phys. Rev. D, 57, 6562, (1998). 1.2   
[87] Grishchuk, L.P., and Kopeikin, S.M., “The motion of a pair of gravitating bodies, including the radiation reaction force”, Sov. Astron. Lett., 9, 230–232, (1983). 1.2   
[88] Hadamard, J., Le probèm de Cauchy et les équation aux dérivées partielles linéaries hyperboliques, (Hermann, Paris, France, 1932). 6.1   
[89] Hulse, R.A., and Taylor, J.H., “Discovery of a pulsar in a binary system”, Astrophys. J. Lett., 195, L51–L53, (1975). 1.1   
[90] Isaacson, R.A., Welling, J.S., and Winicour, J., “Extension of the Einstein quadrupole formula”, Phys. Rev. Lett., 53, 1870–1872, (1984). 1.1   
[91] Itoh, Y., “Equation of motion for relativistic compact binaries with the strong field point particle limit: Third post-Newtonian order”, Phys. Rev. D, 69, 064018, 1–43, (2004). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0310029. 1.1, 1.2, 1.2, 3, 4.2.1, 4.2.2, 4.3, 5, 5.1, 5.2, 5.2, 5.3, 7, 12, C.2   
[92] Itoh, Y., “On the equation of motion of compact binaries in Post-Newtonian approximation”, Class. Quantum Grav., 21, S529–S534, (2004). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0401059. 1.2, 3   
[93] Itoh, Y., and Futamase, T., “New derivation of a third post-Newtonian equation of motion for relativistic compact binaries without ambiguity”, Phys. Rev. D, 68, 121501(R), (2003). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0310028. 1.1, 1.2, 1.2, 3, 8.5   
[94] Itoh, Y., Futamase, T., and Asada, H., “Equation of motion for relativistic compact binaries with the strong field point particle limit: Formulation, the first post-Newtonian and multipole terms”, Phys. Rev. D, 62, 064002, 1–12, (2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9910052. 1.2, 3, 3.3, 4, 4.4, B.1

[95] Itoh, Y., Futamase, T., and Asada, H., “Equation of motion for relativistic compact binaries with the strong field point particle limit: The second and half post-Newtonian order”, Phys. Rev. D, 63, 064038, 1–21, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0101114. 1.2, 1.2, 3, 4, 8.1, C.2   
[96] Iyer, B.R., and Will, C.M., “Post-Newtonian gravitational radiation reaction for two-body systems: Nonspinning bodies”, Phys. Rev. D, 52, 6882–6893, (1995). 1.2   
[97] Jaranowski, P., and Schäfer, G., “Radiative 3.5 post-Newtonian ADM Hamiltonian for many-body point-mass systems”, Phys. Rev. D, 55, 4712–4722, (1997). 1.2   
[98] Jaranowski, P., and Schäfer, G., “Non-uniqueness of the third post-Newtonian binary point-mass dynamics”, Phys. Rev. D, 57, 5948–5949, (1998). Related online version (cited on 7 June 2006): http://arXiv.org/abs/gr-qc/9802030. 1.2   
[99] Jaranowski, P., and Schäfer, G., “Third post-Newtonian higher order ADM Hamilton dynamics for two-body point-mass systems”, Phys. Rev. D, 57, 7274–7291, (1998). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9712075. Erratum: Phys. Rev. D 63 (2001) 029902. 1.2, 5.1, 5.1   
[100] Jaranowski, P., and Schäfer, G., “The binary black-hole problem at the third post-Newtonian approximation in the orbital motion: Static part”, Phys. Rev. D, 60, 124003, 1–7, (1999). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9906092. 1.2   
[101] Jaranowski, P., and Schäfer, G., “The binary black-hole dynamics at the third post-Newtonian order in the orbital motion”, Ann. Phys. (Berlin), 9, 378–383, (2000). Related online version (cited on 12 December 2006): http://arXiv.org/abs/gr-qc/0003054. 1   
[102] Kalogera, V., Kim, C., Lorimer, D.R., Burgay, M., D'Amico, N., Possenti, A., Manchester, R.N., Lyne, A.G., Joshi, B.C., McLaughlin, M.A., Kramer, M., Sarkissian, J.M., and Camilo, F., "The Cosmic Coalescence Rates for Double Neutron Star Binaries", Astrophys. J. Lett., 601, L179–L182, (2004). 1.1   
[103] Kalogera, V., Kim, C., Lorimer, D.R., Burgay, M., D'Amico, N., Possenti, A., Manchester, R.N., Lyne, A.G., Joshi, B.C., McLaughlin, M.A., Kramer, M., Sarkissian, J.M., and Camilo, F., "Erratum: The Cosmic Coalescence Rates for Double Neutron Star Binaries", Astrophys. J. Lett., 614, L137–L138, (2004). 1.1   
[104] Kates, R.E., “Gravitational radiation damping of a binary system containing compact objects calculated using matched asymptotic expansions”, Phys. Rev. D, 22, 1871–1878, (1980). 1.1, 3.1   
[105] Kates, R.E., “Motion of a small body through an external field in general relativity calculated by matched asymptotic expansions”, Phys. Rev. D, 22, 1853–1870, (1980). 1.1, 3.1   
[106] Kerlick, G.D., “Finite reduced hydrodynamic equations in the slow-motion approximation to general relativity. Part I. First post-Newtonian equations”, Gen. Relativ. Gravit., 12, 467–482, (1980). 1.1, 3

[107] Kerlick, G.D., “Finite reduced hydrodynamic equations in the slow-motion approximation to general relativity. Part II. Radiation reaction and higher-order divergent terms”, Gen. Relativ. Gravit., 12, 521–543, (1980). 1.1   
[108] Kidder, L.E., “Coalescing binary systems of compact objects to (post) $^{5/2}$ -Newtonian order. V. Spin effects”, Phys. Rev. D, 52, 821–847, (1995). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9506022. 4.4, B, B.1, B.2, B.5   
[109] Kidder, L.E., Will, C.M., and Wiseman, A.G., “Spin effects in the inspiral of coalescing compact binaries”, Phys. Rev. D, 47, R4183–R4187, (1993). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9211025. B.5   
[110] Kochanek, C.S., “Coalescing Binary Neutron Stars”, Astrophys. J., 398, 234–247, (1992). 3.1, B.5   
[111] Königsdörffer, C., Faye, G., and Schäfer, G., “Binary black-hole dynamics at the third-and-a-half post-Newtonian order in the ADM formalism”, Phys. Rev. D, 68, 044004, 1–19, (2003). Related online version (cited on 12 December 2006): http://arXiv.org/abs/gr-qc/0305048. 1.1, 1.2   
[112] Königsdörffer, C., and Gopakumar, A., “Post-Newtonian accurate parametric solution to the dynamics of spinning compact binaries in eccentric orbits: The leading order spin-orbit interaction”, Phys. Rev. D, 71, 024039, 1–18, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0501011. B.5   
[113] Kopeikin, S.M., “General-relativistic equations of binary motion for extended bodies, with conservative corrections and radiation damping”, Sov. Astron., 29, 516–524, (1985). 1.2   
[114] Kuroda, K. et al., “Status of TAMA”, in Ciufolini, I., and Fidecaro, F., eds., Gravitational Waves: Sources and Detectors, Proceedings of the International Conference, Cascina (Pisa), Italy, 19–23 March 1996, vol. 2 of Edoardo Amaldi Foundation Series, 100, (World Scientific, Singapore; River Edge, U.S.A., 1997). 1.1   
[115] Landau, L.D., and Lifshitz, E.M., The Classical Theory of Fields, vol. 2 of Course of Theoretical Physics, (Pergamon Press, Oxford, U.K.; New York, U.S.A., 1975), 4th edition. 2.3, 4.1, 7   
[116] California Institute of Technology, “LIGO Laboratory Home Page”, project homepage. URL (cited on 7 March 2006):
http://www.ligo.caltech.edu.1.1   
[117] Lorentz, H.A., and Droste, J., “The motion of a system of bodies under the influence of their mutual attraction, according to Einstein’s theory”, in Zeeman, P., and Fokker, A.D., eds., The Collected Papers of H.A. Lorentz, Vol. 5, 330–355, (Nijhoff, The Hague, Netherlands, 1937). English translation of Versl. K. Akad. Wet. Amsterdam, 26, 392 and 649, (1917). 1.1, 1.2   
[118] Maplesoft, “Maple: Math and Engineering Software by Maplesoft”, institutional homepage. URL (cited on 7 March 2006): http://www.maplesoft.com/. 8.7

[119] Marković, D., “Possibility of determining cosmological parameters from measurements of gravitational waves emitted by coalescing, compact binaries”, Phys. Rev. D, 48, 4738–4756, (1993). 1.1   
[120] MathTensor, Inc., “MathTensor for Mathematica”, institutional homepage. URL (cited on 7 March 2006):
http://smc.vnet.net/MathTensor.html. 8.7   
[121] Mino, Y., Sasaki, M., and Tanaka, T., “Gravitational radiation reaction to a particle motion”, Phys. Rev. D, 55, 3457–3476, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9606018. C.1, C.2   
[122] Misner, C.W., Thorne, K.S., and Wheeler, J.A., Gravitation, (W.H. Freeman, San Francisco, U.S.A., 1973). B.4, B.4   
[123] Nissanke, S., and Blanchet, L., “Gravitational radiation reaction in the equations of motion of compact binaries to 3.5 post-Newtonian order”, Class. Quantum Grav., 22, 1007–1032, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0412018. 1.1, 1.2   
[124] Ó Murchadha, N., and York Jr, J.W., “Gravitational energy”, Phys. Rev. D, 10, 2345–2357, (1974). 2.3   
[125] Ohta, T., Okamura, H., Kimura, T., and Hiida, K., “Physically acceptable solution of Einstein’s equation for many-body system”, Prog. Theor. Phys., 50, 492–514, (1973). 1.2   
[126] Ohta, T., Okamura, H., Kimura, T., and Hiida, K., “Coordinate Condition and Higher Order Gravitational Potential in Canocical Formalism”, Prog. Theor. Phys., 51, 1598–1612, (1974). 1.2   
[127] Ohta, T., Okamura, H., Kimura, T., and Hiida, K., “Higher-order gravitational potential for many-body system”, Prog. Theor. Phys., 51, 1220–1238, (1974). 1.2   
[128] Owen, B.J., Tagoshi, H., and Ohashi, A., “Nonprecessional spin-orbit effects on gravitational waves from inspirating compact binaries to second post-Newtonian order”, Phys. Rev. D, 57, 6168–6175, (1998). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9710134. B.5   
[129] Pati, M.E., and Will, C.M., “Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations: Foundations”, Phys. Rev. D, 62, 124015, 1–28, (2000). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0007087. 1.1, 1.2, 1.3, 2.2, 4.1, 4.2, 4.2.1, 8.4, A, A   
[130] Pati, M.E., and Will, C.M., “Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations. II. Two-body equations of motion to second post-Newtonian order, and radiation-reaction to 3.5 post-Newtonian order”, Phys. Rev. D, 65, 104008, 1–21, (2002). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0201001. 1.1, 1.2, 1.2, 8.4, A   
[131] Plebański, J.F., and Bażański, S.L., “The general Fokker action principle and its application in general relativity theory”, Acta Phys. Pol., 18, 307, (1959). 1.1   
[132] Poisson, E., “Gravitational waves from inspirating compact binaries: The quadrupole-moment term”, Phys. Rev. D, 57, 5287–5290, (1998). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9709032. 3.1, B.5

[133] Quinn, T.C., and Wald, R.M., “An axiomatic approach to electromagnetic and gravitational radiation reaction of particles in curved spacetime”, Phys. Rev. D, 56, 3381–3394, (1997). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9610053. C.1   
[134] Rendall, A.D., “On the definition of post-Newtonian approximations”, Proc. R. Soc. London, Ser. A, 438, 341–360, (1992). 2   
[135] Schäfer, G., “The Gravitational Quadrupole Radiation-Reaction Force and the Canonical Formalism of ADM”, Ann. Phys. (N.Y.), 161, 81–100, (1985). 1.1, 1.2   
[136] Schäfer, G., “The ADM Hamiltonian at the Postlinear Approximation”, Gen. Relativ. Gravit., 18, 255–270, (1986). 1.2   
[137] Schäfer, G., “Three-body hamiltonian in general relativity”, Phys. Lett. A, 123, 336–339, (1987). 1.2   
[138] Schutz, B.F., “Statistical formulation of gravitational radiation reaction”, Phys. Rev. D, 22, 249–259, (1980). 1.1, 3.3, 3.3, 4.1   
[139] Schutz, B.F., “The Use of Perturbation and Approximation Methods in General Relativity”, in Fustero, X., and Verdaguer, E., eds., Relativistic Astrophysics and Cosmology, Proceedings of the XIVth GIFT International Seminar on Theoretical Physics, Sant Feliu de Guixols, Spain, 27 June–1 July, 1983, 35, (World Scientific, Singapore, 1984). 1.1   
[140] Schutz, B.F., “Motion and radiation in general relativity”, in Bressan, O., Castagnino, M., and Hamity, V., eds., Relativity, Supersymmetry and Cosmology, Proceedings of the 5th Simposio Latino Americano de Relatividad y Gravitación – SILARG V, 3–80, (World Scientific, Singapore; Philadelphia, U.S.A., 1985). 3   
[141] Schutz, B.F., “Determining the Hubble constant from gravitational wave observations”, Nature, 323, 310–311, (1986). 1.1   
[142] Schutz, B.F., “Lighthouses of gravitational wave astronomy”, in Gilfanov, M., Sunyaev, R.A., and Churazov, E., eds., Lighthouses of the Universe: The Most Luminous Celestial Objects and Their Use for Cosmology, Proceedings of the MPA/ESO/MPE/USM Joint Astronomy Conference held in Garching, Germany, 6–10 August 2001, (Springer, Berlin, Germany; New York, U.S.A., 2002). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0111095. 1.1   
[143] Schwartz, L., Théorie des distributions, (Hermann, Paris, France, 1966). 6.1   
[144] Sobolev, S.L., Partial Differential Equations of Mathematical Physics, (Dover, New York, U.S.A., 1989). Reprint. Originally published in 1964 by Pergamon Press, London. 2.3, 4.1   
[145] Soffel, M.H., Relativity in Astrometry, Celestial Mechanics and Geodesy, (Springer, Berlin, Germany; New York, U.S.A., 1989). B.4   
[146] Stewart, J.M., and Walker, M., “Perturbations of space-times in general relativity”, Proc. R. Soc. London, Ser. A, 341, 49–74, (1974). 2.1, 2.1   
[147] Tagoshi, H., and Nakamura, T., “Gravitational waves from a point particle in circular orbit around a black hole: Logarithmic terms in the post-Newtonian expansion”, Phys. Rev. D, 49, 4016–4022, (1994). 1.1

[148] Tagoshi, H., Ohashi, A., and Owen, B.J., “Gravitational field and equations of motion of spinning compact binaries to 2.5-post-Newtonian order”, Phys. Rev. D, 63, 044006, 1–14, (2001). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0010014. B, B.5   
[149] National Astronomical Observatory, “TAMA Project”, project homepage. URL (cited on 7 March 2006):
http://tamago.mtk.nao.ac.jp.1.1   
[150] Thorne, K.S., “Multipole expansions of gravitational radiation”, Rev. Mod. Phys., 52, 299–339, (1980). 4.2.1   
[151] Thorne, K.S., “LIGO, VIRGO, and the international network of laser-interferometer gravitational-wave detectors”, in Sasaki, M., ed., Relativistic Cosmology, Proceedings of the 8th Nishinomiya-Yukawa Memorial Symposium, Shukugawa City Hall, Nishinomiya, Hyogo, Japan, October 28–29, 1993, vol. 8 of NYMSS, (Universal Academy Press, Tokyo, Japan, 1994). 1.1   
[152] Thorne, K.S., and Hartle, J.B., “Laws of motion and precession for black holes and other bodies”, Phys. Rev. D, 31, 1815–1837, (1985). 1.2, 4.4, B, B.1, B.2, B.3, C.2   
[153] INFN, “The Virgo Project”, project homepage. URL (cited on 7 March 2006): http://www.virgo.infn.it.1.1   
[154] Wald, R.M., General Relativity, (University of Chicago Press, Chicago, U.S.A., 1984). 2.1   
[155] Walker, M., “Isolated Systems in Relativistic Gravity”, in de Sabbata, V., and Karade, T.M., eds., Relativistic Astrophysics and Cosmology, Vol. 1, Proceedings of the Sir Arthur Eddington Centenary Symposium, Nagpur, India, 99–134, (World Scientific, Singapore, 1984). 1.1   
[156] Walker, M., and Will, C.M., “The approximation of radiative effects in relativistic gravity: Gravitational radiation reaction and energy loss in nearly Newtonian systems”, Astrophys. J., 242, L129–L133, (1980). 1.1   
[157] Walker, M., and Will, C.M., “Gravitational radiation quadrupole formula is valid for gravitationally interacting systems”, Phys. Rev. Lett., 45, 1741–1744, (1980). 1.1   
[158] Wang, Y., Stebbins, A., and Turner, E.L., “Gravitational Lensing of Gravitational Waves from Merging Neutron Star Binaries”, Phys. Rev. Lett., 77, 2875–2878, (1996). 1.1   
[159] Will, C.M., “Experimental gravitation from Newton’s Principia to Einstein’s general relativity”, in Hawking, S.W., and Israel, W., eds., Three Hundred Years of Gravitation, 80–127, (Cambridge University Press, Cambridge, U.K.; New York, U.S.A., 1987). 1.1, 3.1   
[160] Will, C.M., Theory and experiment in gravitational physics, (Cambridge University Press, Cambridge, U.K.; New York, U.S.A., 1993), 2nd edition. 1.1, 3.1   
[161] Will, C.M., “Gravitational Waves from Inspirating Compact Binaries: A Post-Newtonian Approach”, in Sasaki, M., ed., Relativistic Cosmology, Proceedings of the 8th Nishinomiya-Yukawa Memorial Symposium, Shukugawa City Hall, Nishinomiya, Hyogo, Japan, 28–29 October, 1993, vol. 8 of NYMSS, 83–98, (Universal Academy Press, Tokyo, Japan, 1994). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9403033. 1.1, 3.1

[162] Will, C.M., “Generation of post-Newtonian gravitational radiation via direct integration of the relaxed Einstein equations”, Prog. Theor. Phys. Suppl., 136, 158–167, (1999). Related online version (cited on 16 March 2006): http://arXiv.org/abs/gr-qc/9910057. 1.3, 2.2, A, A   
[163] Will, C.M., “Post-Newtonian gravitational radiation and equations of motion via direct integration of the relaxed Einstein equations. III. Radiation reaction for binary systems with spinning bodies”, Phys. Rev. D, 71, 084027, 1–15, (2005). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/0502039. 1.2, A   
[164] Will, C.M., “The Confrontation between General Relativity and Experiment”, Living Rev. Relativity, 9, lrr-2006-3, (2006). URL (cited on 5 July 2006): http://www.livingreviews.org/lrr-2006-3. 1.1, 3.1   
[165] Will, C.M., and Wiseman, A.G., “Gravitational radiation from compact binary systems: Gravitational waveforms and energy loss to second post-Newtonian order”, Phys. Rev. D, 54, 4813–4848, (1996). Related online version (cited on 7 March 2006): http://arXiv.org/abs/gr-qc/9608012. 1.3, 2.2, 4.2.1, A, A   
[166] Wolfram, S., “Mathematica: The Way the World Calculates”, institutional homepage, Wolfram Research, Inc. URL (cited on 7 March 2006): http://www.wolfram.com/products/mathematica/. 8.7