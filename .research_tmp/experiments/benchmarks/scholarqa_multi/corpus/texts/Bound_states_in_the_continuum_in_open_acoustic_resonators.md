We consider bound states in the continuum (BSC) or embedded trapped modes in two- and three-dimensional acoustic axisymmetric duct-cavity structures. We demonstrate numerically that under variation of the length of the cavity multiple BSCs occur due to the Friedrich-Wintgen two-mode full destructive interference mechanism. The BSCs are detected by tracing the resonant widths to the points of the collapse of Fano resonances where one of the two resonant modes acquires infinite life-time. It is shown that the approach of the acoustic coupled mode theory cast in the truncated form of a two-mode approximation allows us to analytically predict the BSC frequencies and shape functions to a good accuracy in both two and three dimensions.

Key words: aeroacoustics, noise control, wave scattering

# Bound states in the continuum in open acoustic resonators

A. A. Lyapina $^{1,2}$ , D. N. Maksimov $^{1\dagger}$ , , A. S. Pilipchuk $^{1,2}$ and A. F. Sadreev $^{1}$ ,

$^{1}$ L.V. Kirensky Institute of Physics, Krasnoyarsk, 660036, Russia $^{2}$ Siberian Federal University, Krasnoyarsk, 660041, Russia

(Received 24 June 2015)

# 1. Introduction

Bound states in the continuum (BSCs) also known as embedded trapped modes are localized solutions which correspond to discrete eigenvalues coexisting with extended modes of a continuous spectrum in resonator-waveguide configurations. The existence of trapped solutions residing in the continuum was first reported by von Neumann & Wigner (1929) at the dawn of quantum mechanics. To the best of our knowledge, the term bound state (embedded) in the continuum was introduced by Fonda (1961) in the context of resonance reactions in the presence continuous channels. Since then bound state in the continuum has been universally used to designate a BSC in quantum mechanics Stillinger & Herrick (1975). In the field of fluid mechanics, Parker (1966, 1967) is credited to be the first to encounter resonances of pure acoustic nature in air flow over a cascade of flat parallel plates. Nowadays, the BSCs are known to exist in various waveguide structures ranging from quantum wires Shahbazyan & Raikh (1994); Kim et al. (1999); Olendski & Mikhailovska (2002); Sadreev et al. (2006); Cattapan & Lotti (2007), acoustic waveguides Linton & McIver (2007); Duan et al. (2007); Hein & Koch (2008); Hein et al. (2012), and photonic crystals Shipman & Venakides (2005); Bulgakov & Sadreev (2008); Marinica et al. (2008); Yang et al. (2014). The BSCs are of immense interest, specifically, in optics thanks to experimental opportunity to confine light in optical microcavities despite that outgoing waves are allowed in the surrounding medium Plotnik et al. (2011); Corrielli et al. (2013); Weimann et al. (2013); Chia Wei Hsu et al. (2013); Zhang & Zhang (2015). At the same time, in aerodynamics, trapped and nearly trapped modes are known to cause severe vibrations and noise problems in gas and steam pipelines Jungowski et al. (1989); Ziada & Bühlmann (1992); Kriesels et al. (1995); Ziada & Shine (1999); Tonon et al. (2011).

The most trivial mechanism of BSC formation is due to the symmetry of the structure i.e. the BSC and the continuous-spectrum modes have incompatible symmetries. Typically, when symmetric and antisymmetric problems are separated the first symmetric and antisymmetric propagating modes have different cut-off frequencies. Thus, the difference between the cut-off frequencies provides a window for an antisymmetric BSC to reside in. Such BSCs are omnipresent in wave-related set-ups including quantum wires Schult et al. (1989), fluid in a wave tank Evans & Linton (1991), acoustic Evans et al. (1994); Sugimoto & Imahori (2005) and elastic Maksimov & Sadreev (2006) waveguides.

By breaking the axial symmetry of the structure the symmetry protected BSCs become quasi-trapped leaky modes coupled to the propagating modes of the waveguides Aslanyan et al. (2000). Nöckel (1992); Ladrón de Guevara et al. (2003) demonstrated that if the symmetry is broken under variation of a control parameter the symmetry protected BSCs reveal themselves in form of ghost Fano resonances. Conversely, as the symmetry is recovered the Fano resonance collapses with one of the two resonant modes acquiring an infinite live-time. The collapse of Fano resonance is a signature of full destructive interference of two degenerate eigenmodes of the resonator on the waveguide-resonator interface Kim et al. (1999). In the consequence of this a trapped mode is formed as a localized solution decoupled from the extended modes of the waveguides. Importantly, the full destructive interference of two degenerate eigenmodes leaking into waveguides represents a generic mechanism of BSC formation Friedrich & Wintgen (1985) whose implementations go far beyond the above symmetry arguments by Ladrón de Guevara et al. (2003). In particular, this mechanism allows the formation of BSCs with the same symmetry about the duct axis as the coexisting scattering solution Hein et al. (2012). The BSCs which can be attributed to a full destructive interference are multiply reported in quantum mechanics Kim et al. (1999), optics Bulgakov & Sadreev (2014), and acoustics Hein et al. (2010, 2012). The same interference mechanism was recently employed by Lepetit et al. (2010); Lepetit & Kanté (2014) for experimental observation of BSCs in microwave set-ups. Throughout this paper the full destructive interference of two degenerate modes resulting in the formation of BSCs in multimodal cavities will be invoked to establish two-mode approximation that will be used for finding BSC frequencies and shape functions.

In our recent work Maksimov et al. (2015) we showed that the approach known in quantum mechanics as formalism of the effective non-Hermitian Hamiltonian Dittes (2000); Pichugin et al. (2001) could be adapted for solving hard-wall acoustic cavity-duct problem. The essential feature of the approach is that it allows to recover transmission spectra of open-boundary systems relying on the spectral properties of their closed counterparts. Technically, the method is the acoustic coupled mode theory in which the pressure field within the cavity is represented by the modal variables corresponding to the amplitudes of the eigenmodes of the closed cavity decoupled from waveguides. We also address the reader to the paper by Racec et al. (2009) for a similar approach for scattering in cylindrical nanowire heterostructures.

In this paper, we consider acoustical problem of axisymmetric side-branch cavities in two- and three-dimensional ducts of infinite length. Our primary goal is to explore the interference phenomena resulting in the formation of BSCs localized within two- and three-dimensional cavities. The BSCs will be analysed in the framework of the acoustic coupled mode theory Maksimov et al. (2015). We will demonstrate that our approach not only represents an alternative numerical technique for finding BSCs but also provides an efficient analytical tool for construction the BSC shape function in the two-mode approximation. In particular, we will see that the two-mode approximation is able to a good accuracy reproduce the BSC pressure field within two- and three-dimensional acoustic cavities.

The paper is organized as follows. In Section 2 we briefly overview the acoustic coupled mode theory which is our major tool for analysing the scattering problem. The numerical techniques for finding BSCs are also discussed. In Section 3 we present our results on BSCs in two-dimensional duct-cavity systems. Three-dimensional systems are considered in Section 4. Finally, we conclude in Section 5.

# 2. Acoustic Coupled Mode Theory

Let us start with the formulation of the problem. Throughout this paper we adopt non-dimensionalized quantities. After the time dependence is removed through the substitution $\psi(t)=\psi\exp(-i\omega t)$ the pressure field $\psi$ within a cavity-duct structure is controlled by the non-dimensional stationary Helmholtz equation

$$
\left(\frac {\partial^ {2}}{\partial x ^ {2}} + \frac {\partial^ {2}}{\partial y ^ {2}} + \frac {\partial^ {2}}{\partial z ^ {2}}\right) \psi + \omega^ {2} \psi = 0, \tag {2.1}
$$

where $\omega$ is the non-dimensional frequency. The x-axis is aligned with the center line of the duct-cavity structure. The problem could be reduced to two dimensions under assumption that the structure has constant thickness along the z-axis which allows us to present the solution of Eq. (2.1) as $\psi = \psi(x, y)\psi(z)$ . In what follows we take $\psi(z) = const$ which means that only the first propagating mode is involved.

Eq. (2.1) has to be solved subject to Neumann boundary conditions on all sound-hard boundaries of the cavity and reflectionless boundary conditions on the cavity-waveguide interfaces. There are numerous numerical techniques to impose reflectionless conditions on waveguide-cavity interfaces the most efficient of which implement layers of absorbing medium with complex refractivity giving rise to perfectly matched layers Berenger (1994) or complex scaling Moiseyev (1998) methods. For detailed information on the use of those methods for finding acoustic BSCs we address the reader to the papers by Hein & Koch (2008); Hein et al. (2012) which thoroughly review the subject. In this paper we will use an alternative method, namely, acoustic coupled mode theory Maksimov et al. (2015). The method is a recent adaptation of the approach known in quantum mechanics as formalism of the effective non-Hermitian Hamiltonian. In this section we only briefly review the essentials and introduce the basic concepts that will be needed later on for understanding the mechanism of BSC formation.

To be consistent with the further applications our approach is exemplified on the two-dimensional duct-cavity structure depicted in Fig. 1 where we show a two-dimensional cavity of length $L_{x}$ and width $L_{y}$ coupled to symmetrically positioned waveguides of width d. In what follows we take d = 1. As it was mentioned in the Introduction the coupled mode theory relies on the spectral properties of a closed cavity decoupled from waveguides. In that context closed means that in the first step the eigenmodes and eigenfrequencies of the cavity are computed with the Neumann boundary conditions

$$
\left. \frac {\partial \psi}{\partial \sigma} \right| _ {\partial \Omega} = 0
$$

on the boundary of the resonator $\partial\Omega$ including the waveguide-cavity interfaces as well as the physical boundaries; $\sigma$ being the local coordinate normal the boundary of the closed resonator. In the case of a rectangular cavity the eigenvalue problem is easily solved analytically to yield the spectrum as

$$
\omega_ {n, m} ^ {2} = \left(\frac {\pi (n - 1)}{L _ {x}}\right) ^ {2} + \left(\frac {\pi (m - 1)}{L _ {y}}\right) ^ {2}; n, m = 1, 2, \dots \tag {2.2}
$$

At the same time the eigenmodes are found as products of two factors

$$
\psi_ {n, m} = \sqrt {\frac {(2 - \delta_ {n} ^ {1}) (2 - \delta_ {m} ^ {1})}{L _ {x} L _ {y}}} \cos \left(\frac {\pi (n - 1) (2 x + L _ {x})}{2 L _ {x}}\right) \cos \left(\frac {\pi (m - 1) (2 y + L _ {y})}{2 L _ {y}}\right), (2. 3)
$$

where $\delta_{n'}^{n}$ is the Kronecker delta. Notice that the eigenfunctions are unit-normalized, so

![](images/75a74c299c50f61b19597d429737d285c978dc1349c0df811dcbbe8b651816a3.jpg)

<details>
<summary>text_image</summary>

resonator
d
y
0
x
L_y
L_x
waveguides
</details>

FIGURE 1. (Color on-line) Structure lay-out. d = 1.

that

$$
\int \psi_ {n, m} ^ {2} d x d y = 1. \tag {2.4}
$$

The propagating solutions in the waveguides have a form of plane waves

$$
\phi_ {p} ^ {(\pm)} (x, y) = \sqrt {\frac {1}{4 \pi k _ {p}}} \exp {(\pm i k _ {p} x)} \chi_ {p} (y), \tag {2.5}
$$

where symbol $+(-)$ stands for the waves propagating to the right (left), and the functions $\chi_p(y)$ are defined as

$$
\chi_ {p} (y) = \sqrt {2 - \delta_ {p} ^ {1}} \cos \left(\pi \frac {(p - 1) (2 y + 1)}{2}\right). \tag {2.6}
$$

The dispersion relation reads

$$
k _ {p} = \sqrt {\omega^ {2} - \pi^ {2} (p - 1) ^ {2}}; p = 1, 2, \dots . \tag {2.7}
$$

The pressure field within the cavity $\psi_{c}$ could be constructed as a modal expansion

$$
\psi_ {c} = \sum_ {n, m} g _ {n, m} \psi_ {n, m}. \tag {2.8}
$$

over eigenfunctions Eq. (2.3). The key equation for finding the unknowns $g_{n,m}$ for the cavity excited by an incoming wave has the following form Maksimov et al. (2015)

$$
(D - \omega^ {2} I) \mathbf {g} = - i \sum_ {p = 1} ^ {\infty} \sqrt {\frac {k _ {p}}{\pi}} \sum_ {C = L, R} \mathbf {v} _ {C, p} a _ {C, p} \tag {2.9}
$$

where I is the identity matrix, the subscript C = L, R specifies the coupling with the left and right waveguides, and the unknown vector g contains the coefficients $g_{n,m}$ . Analogously, the amplitudes $a_{C,p}$ are the coefficients of the modal expansion of the incident wave field in the waveguides over the eigenfunctions (2.5), while the vector $v_{C,p}$ contains the coupling constants $v_{C,p}^{m,n}$ between the specific eigenmode $(m,n)$ (2.3) and $(C,p)$ propagating mode of the waveguides (2.5). Finally, the matrix D has the following form

$$
D = \Omega^ {2} - \sum_ {p} ^ {\infty} i k _ {p} \sum_ {C = L, R} \mathbf {v} _ {C, p} ^ {\dagger} \mathbf {v} _ {C, p}, \tag {2.10}
$$

where $\Omega^{2}$ is a diagonal matrix which carries squared eigenfrequencies $\omega_{n,m}^{2}$ Eq. (2.2) on the main diagonal and the symbol $\dagger$ stands for Hermitian transpose. Notice that the coupling to the evanescent modes $(p-1)\pi > \omega$ of the waveguides is accounted for in Eq. (2.10) by the terms with imaginary wavenumbers $k_{p}$ found from Eq. (2.7). At this point we only have to specify the coupling constants $v_{C,p}^{n,m}$ . According to Pichugin et al. (2001); Maksimov et al. (2015) they are evaluated as overlapping integrals between the functions (2.6) and the eigenfunctions (2.3) on the waveguide-cavity interface

$$
v _ {C, p} ^ {m, n} = \int \chi_ {p} \psi_ {m, n} d \tau_ {C}, \tag {2.11}
$$

where $\tau_{C}$ is the local interface coordinate orthogonal to the wave vector. The generalization to the three-dimensional case is straightforward; the coupling constant is a surface integral Maksimov et al. (2015) with $\tau_{C}$ representing the local coordinate set on a flat waveguide-resonator interface. Notice that Eq. (2.9) formally contains an infinite number of unknowns. However, as it was demonstrated in Maksimov et al. (2015) for computational sake the basis can be truncated to a reasonably small number of modes in a given frequency window. After the unknown vector g is obtained from Eq. (2.9) the coefficients of the modal expansion of the outgoing field in the waveguide $b_{C,p}$ are found as

$$
b _ {C, p} = - a _ {C, p} + \sqrt {4 \pi k _ {p}} {\bf v} _ {C, p} ^ {\dagger} {\bf g}. (2. 1 2)
$$

The above described approach is demonstrated to be applicable for finding scattering functions and reflection amplitudes in two- and three-dimensional cavities Maksimov et al. (2015). The goal of the present paper, however, is to adapt it for computing BSCs. The BSCs are localized solutions of infinite live-time which exist in the cavity even when no incident field is present in the waveguides. Thus, according to Eq. (2.9) one obtains an eigenvalue problem

$$
(D - z _ {n} ^ {2} I) \mathbf {g} _ {\mathbf {n}} = 0. \tag {2.13}
$$

Once, an eigenvalue $z_{n} = \omega_{n} + i\gamma_{n}$ found from Eq. (2.13) is real its eigenvector $g_{n}$ corresponds to an BSC. Unfortunately, Eq. (2.13) does not represent the standard eigenvalue problem because the matrix D Eq. (2.10) depends on the spectral parameters $k_{p}$ which are linked to the eigenvalue $z_{n}$ through the dispersion relation Eq. (2.7). One of the possible solutions Sadreev et al. (2006) is employing an iterative procedure of solving the fix-point equation for real eigenvalues of Eq. (2.13). Alternatively, one of the most efficient techniques for finding complex eigenvalues (2.13) is the harmonic inversion method which allows to extract the positions and life-times of the resonances from the response of the system to an external driving. In this paper we do not detail the Harmonic Inversion owing the credit to Wiersig & Main (2008) who nicely outlined the method in their paper on the fractal Weyl law for chaotic microcavities. In brief, the first step in the method is solving Eq. (2.9) with some, quite arbitrary, right hand part which, in particular, could be a point source placed within the open cavity. The response obtained

is then Fourier-transformed to yield a set of nonlinear equation for eigenvalues $z_{n}$ which is subsequently solved by the use of Padé approximants Wiersig & Main (2008).

To conclude this section we return to the Friedrich-Wintgen two-mode interference mechanism of BSC formation Friedrich & Wintgen (1985). Let us check whether it is possible that only two modes $\psi_{1}$ and $\psi_{2}$ of the closed cavity with eigenfrequencies $\omega_{1}$ and $\omega_{2}$ contribute to a BSC. Suppose that only the first scattering channel with the wave number $k_{1}$ is open at the frequency of interest. Then, under a further assumptions that the coupling to the evanescent channels p > 1 could be neglected we rewrite Eq. (2.13) as an eigenvalue problem for matrix D with real valued terms $\omega_{j}^{2}$ arranged along the main diagonal in the ascending order

$$
D = \left( \begin{array}{c c c c c c} \ddots & \vdots & \vdots & \vdots & \vdots & \ddots \\ \dots & \omega_ {0} ^ {2} - i k _ {1} 2 v _ {0} ^ {2} & - i k _ {1} 2 v _ {0} v _ {1} & - i k _ {1} 2 v _ {0} v _ {2} & - i k _ {1} 2 v _ {0} v _ {3} & \dots \\ \dots & - i k _ {1} 2 v _ {0} v _ {1} & \omega_ {1} ^ {2} - i k _ {1} 2 v _ {1} ^ {2} & - i k _ {1} 2 v _ {1} v _ {2} & - i k _ {1} 2 v _ {1} v _ {3} & \dots \\ \dots & - i k _ {1} 2 v _ {0} v _ {2} & - i k _ {1} 2 v _ {1} v _ {2} & \omega_ {2} ^ {2} - i k _ {1} 2 v _ {2} ^ {2} & - i k _ {1} 2 v _ {2} v _ {3} & \dots \\ \dots & - i k _ {1} 2 v _ {0} v _ {3} & - i k _ {1} 2 v _ {1} v _ {3} & - i k _ {1} 2 v _ {2} v _ {3} & \omega_ {3} ^ {2} - i k _ {1} 2 v _ {3} ^ {2} & \dots \\ \ddots & \vdots & \vdots & \vdots & \vdots & \ddots \end{array} \right), \tag {2.14}
$$

where $\omega_{0}$ and $\omega_{3}$ are the eigenfrequencies next to $\omega_{1}$ and $\omega_{2}$ , and $v_{j}=v_{L,1}^{m,n}$ is the coupling constant with the open propagation channel p=1. Notice, that above we assumed that the absolute values of the coupling constants on the left and the right are identical i.e. all eigenmodes have compatible symmetry about the y-axis. In fact, the matrix D is split into two decoupled subblocks corresponding to the y-symmetric and y-antisymmetric eigenmodes so that the y-symmetric and y-antisymmetric problems could be considered independently from one another. Under variation of some control parameter, say the length of the cavity $L_{x}$ , one can reach the degeneracy point where $\omega_{1}=\omega_{2}=\omega_{BSC}$ , then the matrix (2.14) has a real eigenvalue $\omega_{BSC}^{2}$ with the eigenvector

$$
\mathbf {g} = \frac {v _ {1} v _ {2}}{\sqrt {v _ {1} ^ {2} + v _ {2} ^ {2}}} \big (  \ldots ,    0,    1 / v _ {1},    - 1 / v _ {2},    0,    \ldots   \big) ^ {\dagger}
$$

containing only two non-zero entries. Thus, the corresponding BSC shape-function can be constructed as

$$
\psi_ {B S C} = g (1) \psi_ {1} + g (2) \psi_ {2}. \tag {2.15}
$$

The above equation represents the central mathematical result to be used for analyzing BSCs in open acoustic resonators. In essence, it manifests the well known interference mechanism Friedrich & Wintgen (1985). In fact, both eigenmodes $\psi_{1}$ and $\psi_{2}$ albeit coupled to the waveguides contribute to Eq. (2.15) in the proportion providing a full destructive interference at the degeneracy point so that the resulting BSC function $\psi_{BSC}$ is decoupled from the open channel. For details on the realization of the above mechanism in the field of quantum mechanics we address the reader to the paper by Sadreev et al. (2006). We mention in passing that thanks to the symmetry about the duct axis the same arguments are valid for the duct-axis antisymmetric modes with $k_{1}$ replaced by $k_{2}$ according to Eq. (2.7).

The above result Eq. (2.15) is exact under assumption that the coupling to the evanescent channels could be neglected. However, we would like to remind the reader that the evanescent eigenmodes are present in summation over index p in the right hand part of Eq. (2.10). Therefore Eq. (2.15) is not an exact solution, and could only be seen as the basis for a two-mode approximation for BSCs occurring at the degeneracy points. Some remarks are due on the validity of the two-mode approximation. As it will be seen later on

<table><tr><td>p \ (n,m)</td><td>(1,1)</td><td>(1,2)</td><td>(1,3)</td><td>(1,4)</td><td>(4,1)</td><td>(4,2)</td><td>(2,3)</td><td>(2,4)</td></tr><tr><td>1</td><td>0.408</td><td>0</td><td>-0.367</td><td>0</td><td>0.577</td><td>0</td><td>-0.520</td><td>0</td></tr><tr><td>2</td><td>0</td><td>0.245</td><td>0</td><td>-0.441</td><td>0</td><td>0.346</td><td>0</td><td>-0.624</td></tr><tr><td>3</td><td>0</td><td>0</td><td>0.173</td><td>0</td><td>0</td><td>0</td><td>0.245</td><td>0</td></tr><tr><td>4</td><td>0</td><td>0.021</td><td>0</td><td>0.082</td><td>0</td><td>0.030</td><td>0</td><td>0.116</td></tr><tr><td>5</td><td>0</td><td>0</td><td>0.034</td><td>0</td><td>0</td><td>0</td><td>0.049</td><td>0</td></tr><tr><td>6</td><td>0</td><td>0.007</td><td>0</td><td>0.024</td><td>0</td><td>0.011</td><td>0</td><td>0.034</td></tr></table>

TABLE 1. Numerical values of coupling constants $v_{L,p}^{n,m}$ at $L_{x} = 3, L_{y} = 2$ .

due to the structure of Eq. (2.10) the evanescent coupling results in the presence of other than the two basic modes in the modal expansion of the BSC shape function, as well as in a shift of the true BSC point from the degeneracy point. Nevertheless, one can argue that the coupling to the evanescent channels is small relative to the coupling to the open channel as we consider the eigenmodes residing in the frequency window where only one symmetric and one antisymmetric propagating channels are open. In order to illustrate this we collected in Table 1 the numerical values of the coupling constants for some relevant eigenmodes evaluated on the basis of the analytical results from Maksimov et al. (2015). One can see from Table 1 that the coupling constants rapidly decay with the growth of the channel number $p$ . In fact, in can be demonstrated that any coupling constant decays away from its maximal value with the asymptotic law $v_{L,p}^{n,m} \sim 1 / p^2$ .

Finally, notice that the two-mode approximation has no restrictions related to the size of the cavity. One can see from Eq. (2.14) that once a degeneracy occurs in the closed cavity it always results in the formation of a BSC through the Friedrich-Wintgen mechanism irrelevant to the values of $L_{x}$ , and $L_{y}$ . If the size of the cavity is isometrically increased whilst the width of the waveguide is kept unchanged, the next to the BSC eigenfrequencies $\omega_0, \omega_3$ would shift closer to the BSC eigenfrequency. The real valued diagonal term in Eq. (2.13) will drop as $(\omega_{0,3}^2 - \omega_{BSC}^2) \sim 1 / S, S = L_x L_y$ because the mean distance between the eigenfrequencies is inversely proportional to the width/length of the cavity. On the other hand the coupling constants $v_{j}$ in matrix $D$ (2.14) will drop as $v_{j} \sim 1 / \sqrt{S}$ because the eigenfunctions Eq.(2.3) contributing to the overlapping integrals Eq. (2.11) have the normalization constant $\sqrt{S}$ in denominator. In consequence of this the off-diagonal terms $v_{j}v_{j'}$ in matrix (2.14) drop as $v_{j}v_{j'} \sim 1 / S$ . Therefore, although in larger cavities the next to the BSC eigenvalues appear closer to the BSC point their coupling to the basic BSC modes through the evanescent channels drops at the same rate as the resonant term $\omega_{0,3}^2 - \omega_{BSC}^2$ . This leads us to a conclusion that the two-mode approximation does not break down as the size of the cavity is increased. Notice, that the same holds true in the three-dimensional case where the coupling constant drops as $v_{j} \sim 1 / \sqrt{V}$ with $V$ as the volume of the cavity.

# 3. Bound states in two dimensional duct-cavity structure

We start with computing the transmittance $T = |b_{1,R}|^{2}$ for the two-dimensional cavity in Fig. 1 under assumption that an incident wave in the first propagation band p = 1 enters the cavity from the left waveguide. For solving the scattering problem we used the approach described in Section 2. The numerical values of the coupling constants $v_{C,p}^{n,m}$ in Eq. (2.9,2.10) are taken from our previous work Maksimov et al. (2015). In Fig. 2 we demonstrate the dependence of the transmittance T on the frequency of the incoming

![](images/9d94de789b0710c9c558667faa6a12871ddd07c9f0371fe04cced260b60e397e.jpg)

<details>
<summary>heatmap</summary>

| Point | ω    | Lx   |
|-------|------|------|
| 1     | 3.0  | 4.0  |
| 2     | 3.0  | 2.0  |
| 3     | 3.5  | 2.8  |
| 4     | 3.5  | 3.5  |
| 5     | 4.0  | 4.0  |
</details>

FIGURE 2. (Color on-line) Transmittance for a two-dimensional cavity of length $L_{y}=2$ . Only the first symmetric channel with p=1 is open. The dashed lines show eigenfrequencies of the modes symmetric about the y-axis; solid line - antisymmetric ones. The positions of the BSCs are shown by open circles and stars for y-symmetric and y-antisymmetric modes, correspondingly.

wave $\omega$ and the length of the cavity $L_{x}$ . Notice that the frequency varies in the range $[0,2\pi]$ which means according to Eq. (2.7) that only one symmetric scattering channel is open in each waveguide. In Fig. 2 we also depict the frequencies of the eigenmodes of the closed cavity symmetric about the duct axis Eq. (2.2) $m = 1,3,5,\ldots$ . The eigenfrequencies which correspond to the modes symmetric about the $y$ -axis, later to be referred to as $y$ -symmetric, are shown by dashed lines. The solid lines show the eigenfrequencies of $y$ -antisymmetric eigenmodes. One can see that the degeneracy points of eigenfrequencies with the same symmetry about the $y$ -axis adjoin to an abrupt change of the transmittance; the transmittance drops from unit to zero as the frequency is swept across the point where the collapse of Fano resonance occurs. The complex resonance frequencies for the cavity are computed by harmonic inversion method. In the given range of parameters we found five BSCs. Three of them originate from the modes symmetric about the $y$ -axis (shown by circles) and two from the antisymmetric ones (shown by stars). For further convenience we numerate the BSCs from 1 to 5. The complex eigenfrequencies are plotted against the length of the resonator $L_{x}$ in Fig. 3 where one can see that the imaginary part $\gamma_{n}$ turns to zero as one approaches the BSC point. In Fig. 4 we demonstrate the numerical solution for BSC 3 along with the corresponding coefficients of the modal expansion (2.8). One can clearly see that only two eigenmodes dominate in the BSC function $\psi_{BSC}$ . The $y$ -antisymmetric BSC 3 in Fig. 4 is formed by eigenmodes $\psi_{1} = \psi_{4,1}(x,y)$ , and $\psi_{2} = \psi_{2,3}(x,y)$ with the corresponding coupling constants $v_{1} = v_{L,1}^{4,1} = -v_{R,1}^{4,1} = 0.600$ , and $v_{2} = v_{L,1}^{2,3} = -v_{R,1}^{2,3} = -0.540$ . To assess the accuracy of

![](images/2f4078e99cf561c5142ee06b9fc324892535619968c5d4bd9200418e527b633b.jpg)

<details>
<summary>line</summary>

| L_x | ω_n (Line 1) | ω_n (Line 2) | ω_n (Line 3) | ω_n (Line 4) | ω_n (Line 5) |
|-----|--------------|--------------|--------------|--------------|--------------|
| 2   | ~1.8         | ~3.0         | ~3.8         | -            | -            |
| 3   | ~1.5         | -            | -            | ~3.5         | -            |
| 4   | ~1.0         | -            | -            | ~3.0         | ~4.0         |
</details>

![](images/ede51366ecafb0c97c46d3650e17910dc86b47847567f3963d244f938604d4c0.jpg)

<details>
<summary>line</summary>

| L_x | Curve 1 | Curve 2 | Curve 3 | Curve 4 | Curve 5 |
|-----|---------|---------|---------|---------|---------|
| 2   | -0.6    | -0.2    | -0.2    | -0.2    | -0.2    |
| 3   | -0.4    | 0.0     | 0.0     | 0.0     | 0.0     |
| 4   | -0.2    | 0.0     | 0.0     | 0.0     | 0.0     |
</details>

FIGURE 3. (Color on-line) Complex eigenfrequencies against the length of the two-dimensional resonator $L_{x}$ ; real parts - left panel; imaginary parts - right panel. The y-symmetric resonances are shown by dash brown lines; y-antisymmetric by blue solid lines. The positions of BSCs from Fig. 2 are shown by red open circles and stars for y-symmetric and y-antisymmetric modes, correspondingly.

the two-mode approximation we introduce parameters

$$
\triangle \omega = \frac {\left| \omega_ {B S C} - \omega^ {*} \right|}{\omega_ {B S C}}, \tag {3.1}
$$

and

$$
\triangle L _ {x} = \frac {\left| L _ {x} - L _ {x} ^ {*} \right|}{L _ {x}}, \tag {3.2}
$$

with $\omega^{*}$ , and $L_{x}^{*}$ as the frequency and the length of the cavity corresponding to the degeneracy point in which the BSC would occur according to the two-mode approximation Eq. (2.14). For BSC 3 we obtained $\triangle\omega = 0.019$ , and $\triangle L_{x} = 0.018$ . In Fig. 4 we also plot the BSC shape function constructed by Eq. (2.15). One can see from Fig. 4 that the two-mode approximation reproduces our numerical result to a good accuracy. We also found that the same holds true for the other four BSCs in Fig. 2. Therefore, the other BSC functions are not presented in this paper.

The same procedure was repeated in case of the incoming wave in the first antisymmetric channel for cavities with $L_{y} = 2$ . As before we observed that the two-mode approximation is valid in the situation when only one antisymmetric channel is open in each waveguide. For brevity we omit a detailed description of our findings restricting ourselves to one example of a duct axis antisymmetric BSC shown in Fig. 5. The $y$ -symmetric BSC in Fig. 5 is formed by eigenmodes $\psi_{1} = \psi_{5,2}(x,y)$ , and $\psi_{2} = \psi_{3,4}(x,y)$ with the corresponding coupling constants $v_{1} = v_{L,1}^{5,2} = v_{R,1}^{5,2} = -0.692$ , and $v_{2} = v_{L,1}^{3,4} = v_{R,1}^{3,4} = 0.384$ . For the corresponding relative errors Eqs. (3.1,3.2) we found $\triangle \omega = 0.004$ , and $\triangle L_{x} = 0.004$ . As it was mentioned in Section 2, we expect that the two-mode approximation could be applied irrelevant to the cavity size. This is confirmed by numerical tests for larger cavities $L_{y} = 4$ . As an example in Fig. 6 we present the duct-axis antisymmetric BSC formed by eigenmodes $\psi_{1} = \psi_{5,8}(x,y)$ , and $\psi_{2} = \psi_{7,6}(x,y)$ with the coupling constants $v_{1} = v_{L,1}^{5,8} = v_{R,1}^{5,8} = -0.371$ , and $v_{2} = v_{L,1}^{7,6} = v_{R,1}^{7,6} = 0.402$ . The values of the relative

a)   
![](images/9e50cd47afcfed6ef18064fa7da918d7d7ddf779a2c483901c48f0b3b44de6b4.jpg)

<details>
<summary>heatmap</summary>

| X Range | Y Range | Value |
|---------|---------|-------|
| Low     | Low     | -1    |
| Low     | High    | 1     |
| Mid     | Low     | -1    |
| Mid     | High    | 1     |
</details>

![](images/ca055758ee1233ead12638440a1ca4d07eae347982d749711cb3e6b249c2a502.jpg)

c)   
![](images/3c2124011403260a7e4eb7653d0e483ccb6e827e87756310dcab0198be3fdc4e.jpg)

<details>
<summary>natural_image</summary>

Abstract heat map pattern with blue, red, and yellow regions (no text or symbols)
</details>

FIGURE 4. (Color on-line) BSC 3 symmetric about the duct axis of a two-dimensional duct-cavity structure $\omega_{BSC} = 3.396, L_{y} = 2, L_{x} = 2.776$ ; a) Pressure field of the numerically computed BSC shape function; b) Absolute values of the modal coefficient of the numerically computed pressure field; c) BSC shape function found from the two-mode approximation Eq. (2.15).   
a)   
![](images/c4693b6dfd328cfde28c91b0a94061155fa6c0bd4011edc795602445c5ba4da3.jpg)

<details>
<summary>heatmap</summary>

| Position | Value |
|--------|-------|
| 1      | -1    |
| 2      | 0     |
| 3      | 1     |
</details>

![](images/d5441867e5fbc270aee019b7eb4e1d9e8553f1b77e908cece1f70cde8dbcebe7.jpg)

c)   
![](images/c6d8c310aebdc11bd3ed0640d52aa981010018807917da7c4b3e2ab42a1969f5.jpg)

<details>
<summary>natural_image</summary>

Abstract heat map pattern with blue, orange, and red gradients (no text or symbols)
</details>

FIGURE 5. (Color on-line) Bound state in the continuum antisymmetric about the duct axis of a two-dimensional duct-cavity structure $\omega_{BSC} = 5.385, L_{y} = 2, L_{x} = 2.441$ ; a) Pressure field of the numerically computed BSC shape function; b) Absolute values of the modal coefficient of the numerically computed pressure field; c) BSC shape function found from the two-mode approximation Eq. (2.15).

![](images/4d29163d43671f4ad6d9fcbca1960eb0102a3b9aef10ee140dfd9ba501ea71fa.jpg)

<details>
<summary>heatmap</summary>

| Value Range | Color Scale |
|-------------|-------------|
| -0.5        | Blue        |
| 0           | Yellow      |
| 0.5         | Red         |
</details>

![](images/64e50283f384fbe86c69aa0e015e3e918807a428c5471f7aaac3bab3601d409f.jpg)

<details>
<summary>bar</summary>

| n   | m   |
| --- | --- |
| 8   | 10  |
| 9   | 10  |
</details>

![](images/e1ac5db3a94ef892bb2afae15e7354bc742d920d5b93b4c5ff0533e19e13fb38.jpg)

<details>
<summary>natural_image</summary>

Colorful abstract pattern with symmetrical symmetrical shapes (no text or symbols)
</details>

FIGURE 6. (Color on-line) Bound state in the continuum antisymmetric about the duct axis of a two-dimensional duct-cavity structure $\omega_{BSC} = 6.502, L_{y} = 4, L_{x} = 3.627$ ; a) Pressure field of the numerically computed BSC shape function; b) Absolute values of the modal coefficient of the numerically computed pressure field; c) BSC shape function found from the two-mode approximation Eq. (2.15).

errors $\triangle \omega = 0.003$ , and $\triangle L_{x} = 0.007$ indicate that the two-mode approximation holds true for larger cavities.

As it was mentioned in the introduction the occurrence of BSC is associated with a collapse of Fano resonance. To illustrate this in Fig. 7 we plot the dependence transmittance vs. frequency for three different lengths of the resonator $L_{x}$ close to the BSCs point. It was recently demonstrated by Bulgakov & Sadreev (2014) that if the BSC point is approached in the parametric space along the certain path corresponding to the peak of the Fano resonance, the long-lived complex mode related to the BSC provides the dominant contribution to the scattering function. In consequence of this the scattering function near the BSC point bears the maximal resemblance of the BSC if the parameters are tuned to the peak of the Fano resonance. This is illustrated in Fig. 8 where one can see that the real part of the scattering function is almost identical to the BSC from Fig. 4 meanwhile the imaginary is present as a background providing the net power flux in the system. Thus, one can experimentally observe a signature of BSC in the form of a large amplitude acoustic resonance even though in a realistic set-up the parameters are always detuned from the true BSC point due, for example, to fabrication inaccuracies.

# 4. Bound states in the continuum in three-dimensional open cavities

Let us consider the axisymmetric cylindrical resonator with two coaxially attached cylindrical waveguides of radius a = 1 as depicted in Fig. 9,a). After the azimuthal variable is separated we can restrict ourself to the case of zero angular momentum bearing in mind that our results can be easily generalized to the non-zero case taking into account that by virtue of Eq. (2.11) only the modes of the same rotational number are coupled on the waveguide-cavity interface. In our case the cavity eigenfrequencies are given by

![](images/7743bd76e79aeeb5a7dc9d95fc96450612b928e789278b231ac02dbdf5b047e2.jpg)

<details>
<summary>line</summary>

| x     | Blue Line | Green Dashed Line | Red Dotted Line |
|-------|-----------|-------------------|-----------------|
| 3.35  | 1.0       | 1.0               | 1.0             |
| 3.4   | 0.9       | 0.9               | 0.9             |
| 3.45  | 0.85      | 0.8               | 0.7             |
| 3.5   | 0.8       | 0.8               | 0.75            |
</details>

ω

![](images/1217128ccf686020300ba3d101d58441a171f80906857fc6759645f2f0be5d36.jpg)

<details>
<summary>line</summary>

| x    | T (red dashed) | T (green dash-dot) | T (blue solid) |
|------|----------------|--------------------|----------------|
| 5.2  | 1.0            | 1.0                | 1.0            |
| 5.3  | 0.0            | 0.0                | 1.0            |
| 5.4  | 1.0            | 1.0                | 1.0            |
</details>

ω

FIGURE 7. (Color on-line) Collapse of Fano resonance in the vicinity of a BSC point under variation of the length of the resonator $L_{x}$ . a) BSC 4 in Fig. 4 $\omega_{BSC} = 3.396$ , $L_{y} = 2$ , $L_{x} = 2.776$ . Resonator lengths: $L_{x} = 2.75$ - solid blue line, $L_{x} = 2.70$ - dashed green line, $L_{x} = 2.60$ - dash-dot red line. b) Duct axis antisymmetric BSC in Fig. 5 $\omega_{BSC} = 3.384$ , $L_{y} = 2$ , $L_{x} = 2.441$ . Resonator lengths: $L_{x} = 2.460$ - solid blue line, $L_{x} = 2.485$ - dashed green line, $L_{x} = 2.510$ - dash-dot red line.   
![](images/c2de51902c5e359b75ae8799d213fd1847c5158af7d8eca07328a2d38c2650d4.jpg)

<details>
<summary>heatmap</summary>

| Region | Value |
|--------|-------|
| Left   | ~0.0  |
| Right  | ~0.2  |
</details>

b)   
![](images/4f8065cd449f2a6019b2dfdcfa85d132afc17409140e7b0aeb4adeddd87a773d.jpg)

<details>
<summary>heatmap</summary>

| X Range | Y Range | Value  |
|---------|---------|--------|
| Low     | Low     | -0.05  |
| Low     | High    | 0.05   |
| Mid     | Low     | -0.05  |
| Mid     | High    | 0.05   |
| High    | Low     | -0.05  |
| High    | High    | 0.05   |
</details>

FIGURE 8. (Color on-line) Scattering function at the peak $\omega = 3.410$ of the Fano resonance Fig. 7 related to BSC 3 shown in Fig. 4. $L_{y} = 2$ , $L_{x} = 2.75$ . a) Real part of the scattering function. b) Imaginary part of the scattering function.

![](images/6728c0c79d334778702a3fcdbcc0b4290133e234d43cfc310e24c8f4ef2a057f.jpg)  
FIGURE 9. Bound states in axisymmetric duct-cavity structure. a) Cylindrical resonator of radius R and length $L_{x}$ with two coaxially attached waveguides of radius a = 1. b) Pressure field isosurfaces of BSC 2 in Fig. 12 symmetric with respect to the central section of the resonator; dark blue $\psi_{BSC} = -0.1$ , light red $\psi_{BSC} = 0.1$ . c) Pressure field isosurfaces of BSC 7 in Fig. 13 antisymmetric with respect to the central section of the resonator; dark blue $\psi_{BSC} = -0.1$ , light red $\psi_{BSC} = 0.1$ .

$$
\omega_ {m, n} ^ {2} = \left[ \frac {\mu_ {m} ^ {2}}{R ^ {2}} + \frac {\pi^ {2} (n - 1) ^ {2}}{L _ {x} ^ {2}} \right] \tag {4.1}
$$

where R and $L_{x}$ are the radius and length of the cylindrical resonator, and $\mu_{m}$ is the m-th root of the equation $J_{0}^{\prime}(\mu_{m}) = J_{1}(\mu_{m}) = 0, m = 1, 2, \ldots$ for the derivative of the first order Bessel function. The corresponding eigenfunctions are

$$
\psi_ {m, n} (r, x) = \frac {1}{\sqrt {\pi} R J _ {0} (\mu_ {m})} J _ {0} \left(\frac {\mu_ {m} r}{R}\right) \psi_ {n} (x), \tag {4.2}
$$

![](images/fffdc8f644042f11271e983143e13338b32dc53d7bbfc0584d65928c5fec3a2c.jpg)

<details>
<summary>heatmap</summary>

| Point | ω    | Lx   |
|-------|------|------|
| 1     | 2.3  | 3.2  |
| 2     | 2.4  | 2.6  |
| 3     | 2.6  | 3.8  |
| 4     | 3.4  | 3.9  |
| 5     | 3.4  | 3.7  |
| 6     | 3.4  | 2.4  |
| 7     | 3.5  | 3.5  |
| 8     | 3.5  | 2.8  |
| 9     | 3.5  | 2.6  |
</details>

FIGURE 10. (Color on-line) Transmittance for a cylindrical resonator with the radius R = 2 with two coaxially attached cylindrical waveguides of radius a = 1 vs. frequency $\omega$ and the length of the resonator $L_{x}$ . The dashed lines show eigenfrequencies of the modes symmetric about the y0z-plane; solid line - antisymmetric ones. The positions of the BSCs are shown by open circles and stars for y0z-symmetric and y0z-antisymmetric modes, correspondingly.

where r is the radial coordinate in the y0z-plane while the axial modes are given by

$$
\psi_ {n} (x) = \sqrt {\frac {2 - \delta_ {n} ^ {1}}{L _ {x}}} \cos \left(\frac {\pi (n - 1) (2 x + L _ {x})}{2 L _ {x}}\right), n = 1, 2, \dots \tag {4.3}
$$

As before the transmission problem was solved by the use of the acoustic coupled mode theory Maksimov et al. (2015) and the complex eigenvalues were obtained by the harmonic inversion method. The frequency $\omega$ varies in the range $[0,\mu_{2}]$ , $\mu_{2}=3.831$ which means that only one propagating channel is open in each waveguide. The transmittance along with the spectrum of the closed resonator and the positions of BSCs is shown in Fig. 10. The complex eigenfrequencies are plotted in Fig. 11 against the length of the resonator $L_{x}$ . In Figs. 12,13 we demonstrate the shape functions of two BSCs occurring through the two mode interference mechanism. BSC 2 in Fig. 12 with $\triangle\omega=0.012$ , and $\triangle L_{x}=0.012$ is formed by eigenmodes $\psi_{1}=\psi_{1,3}(r,x)$ , and $\psi_{2}=\psi_{3,1}(r,x)$ with the corresponding coupling constants $v_{1}=v_{L,1}^{3,1}=v_{R,1}^{3,1}=0.310$ , and $v_{2}=v_{L,1}^{1,3}=v_{R,1}^{1,3}=0.289$ , while y0z-antisymmetric BSC 7 depicted in Fig. 13 is formed by $\psi_{1}=\psi_{2,4}(r,x)$ , and $\psi_{2}=\psi_{4,3}(r,x)$ with the coupling constants $v_{1}=v_{L,1}^{2,4}=-v_{R,1}^{2,4}=-0.078$ , and $v_{2}=v_{L,1}^{4,3}=-v_{R,1}^{4,3}=0.389$ . A three dimensional visualization of BSCs from Figs. 12,13 is offered in Fig. 9 b,c) in form of pressure field isosurfaces. One can see that of all BSCs in Fig. 10 BSC 7 deviates the most from the corresponding degeneracy point $\triangle\omega=0.016$ , and $\triangle L_{x}=0.066$ . The

![](images/615b1fc37c42eaa34eb6f84ba1e98c1042cc9f66b2ae695c7a45d6925768e0ee.jpg)

<details>
<summary>line</summary>

| L_x | θ_n (Line 1) | θ_n (Line 2) | θ_n (Line 3) |
|-----|--------------|--------------|--------------|
| 2.0 | 2.5          | 2.5          | 2.5          |
| 2.5 | 2.0          | 2.0          | 2.0          |
| 3.0 | 1.5          | 1.5          | 1.5          |
| 3.5 | 1.0          | 1.0          | 1.0          |
| 4.0 | 0.5          | 0.5          | 0.5          |
</details>

![](images/436f4d7ef0445df0e3b641567ff8ff20621b8ff495844b3227bccf146570ef5e.jpg)

<details>
<summary>line</summary>

| L_x | γ_n (Solid Blue) | γ_n (Dashed Blue) | γ_n (Solid Yellow) | γ_n (Dashed Yellow) |
|-----|------------------|-------------------|--------------------|---------------------|
| 2.0 | -0.45            | -0.05             | -0.25              | -0.15               |
| 2.5 | -0.35            | -0.08             | -0.20              | -0.10               |
| 3.0 | -0.25            | -0.10             | -0.15              | -0.05               |
| 3.5 | -0.15            | -0.12             | -0.10              | 0.00                |
| 4.0 | -0.10            | -0.15             | -0.05              | 0.05                |
</details>

FIGURE 11. (Color on-line) Complex eigenfrequencies against the length of the three-dimensional resonator $L_{x}$ ; real parts - left panel, imaginary parts - right panel. The y0z-symmetric resonances are shown by dash brown lines; y0z-antisymmetric by blue solid lines. The positions of BSCs from Fig. 10 are shown by red open circles and stars for y0z-symmetric and y0z-antisymmetric modes, correspondingly.

reason for this can be understood from Figs. 9 c), 13 where one can clearly see that BSC 7 extends far into the waveguides due to the coupling to the evanescent modes which is neglected by the two mode approximation. One can conclude that the coupling to the evanescent modes becomes important just below the next channel propagation threshold as the channel function extends far into the waveguide. Nevertheless, the two-mode approximation is still valid in the three-dimensional case reproducing the numerically computed shape functions to a good accuracy.

In this paper, we restricted ourself to coaxial systems in which the azimuthal number is a constant of motion so that the rotational vibration modes do not participate in formation of BSC depicted in Figs. 12, 13. It is worth noticing, however, that the cavity eigenmodes with azimuthal number $\nu \neq 0$ are symmetry protected BSCs as their eigenfrequencies are below the $\nu$ -th propagating threshold in cylindrical waveguides.

# 5. Conclusion

We considered sound transmission through two- and three-dimensional acoustic resonators and demonstrated the existence of singular points in which the unit transmittance coalesces with the zero transmittance in the points of collapse of Fano resonances. These singular points occur due to a full destructive interference of two degenerate modes of the same symmetry. As a result a certain superposition of these two modes is a trapped solution corresponding to a bound state in the continuum or embedded trapped mode localized within the resonator. The above mechanism of wave localization was first proposed by Friedrich & Wintgen (1985), and has been so far experimentally realized only in microwave set-ups Lepetit et al. (2010); Lepetit & Kanté (2014). In the present paper we invited the reader's attention to acoustic cylindrical resonators coaxially connected to semi-infinite waveguides where the degeneracy points could be accessed in the parametric

a)   
![](images/92dcd3c13ce4af12f821c0f494e848e992442a2c2a145e104cf46012e6c0bc0b.jpg)

<details>
<summary>heatmap</summary>

| Position | Value  |
|--------|--------|
| 1      | 0.35   |
| 2      | 0.2    |
| 3      | 0.1    |
| 4      | 0.0    |
| 5      | -0.1   |
| 6      | -0.2   |
</details>

b)   
![](images/c4c827eba97c1be2e355968bddf34572db1b9a3a0bca72a10d066849e015a909.jpg)

c)   
![](images/879879bfbd70b8a503c8037b16bfb127739e3a3b707207d2fddafd39d5827cfb.jpg)

<details>
<summary>natural_image</summary>

Abstract heatmap pattern with red center fading to blue edges (no text or symbols)
</details>

FIGURE 12. (Color on-line) BSC 2 in a three-dimensional duct-cavity structure $\omega_{BSC} = 2.367$ , $R = 2$ , $L_x = 2.655$ ; a) $x0y$ -plane section of the pressure field of the numerically computed BSC shape function; b) Absolute values of the modal coefficient of the numerically computed pressure field; c) BSC shape function found from the two-mode approximation Eq. (2.15).   
a)   
![](images/09c1fbad174f3cc54399d832edf9950a18084b3e49c5844088abe776114f7325.jpg)

<details>
<summary>heatmap</summary>

| X Range | Y Range | Value |
|---------|---------|-------|
| Low     | Low     | -0.5  |
| Low     | High    | 0.5   |
| Mid     | Low     | -0.5  |
| Mid     | High    | 0.5   |
</details>

b)   
![](images/ee6c39f2cfd757bec40f3b587150d3fefec04c5081f227053c5a3807cb9dc315.jpg)

c)   
![](images/4d4b7c2a9c1f64dd32eaeab53a661089a8d1f2b5063915a2ba00df815198dc5b.jpg)

<details>
<summary>natural_image</summary>

Abstract color gradient pattern with red, blue, and green hues (no text or symbols)
</details>

FIGURE 13. (Color on-line) BSC 7 in a three-dimensional duct-cavity structure $\omega_{BSC} = 3.557, R = 2, L_x = 3.387$ ; a) $x0y$ -plane section of the pressure field of the numerically BSC; b) Absolute values of the modal coefficient of the numerically computed pressure field; c) BSC shape function found from the two-mode approximation Eq. (2.15).

space under variation of the length of the resonator. We speculate that such a tuning could be performed in a realistic acoustic experiment by the use of piston-like hollow-stem waveguides tightly fit to the interior boundaries of a cylindric cavity. For analysing the bound states in the continuum we successfully employed the acoustic coupled mode theory Maksimov et al. (2015) which allowed us to analytically predict the eigenfrequencies of the bound states as well as their shape functions. Until recently Hein et al. (2012) the bound states in the continuum were independently studied in physics and fluid mechanics with little to none cross-fertilization. In the present paper we attempted to bridge that gap to the mutual benefit of the both communities.

Acknowledgement: This work has been supported by Russian Scientific Foundation through Grant 14-12-00266.

# REFERENCES

ASLANYAN, A., PARNOVSKI, L. & VASSILIEV, D. 2000 Complex resonances in acoustic waveguides. The Quarterly Journal of Mechanics and Applied Mathematics 53 (3), 429–447.   
BERENGER, J.-P. 1994 A perfectly matched layer for the absorption of electromagnetic waves. Journal of Computational Physics 144, 185200.   
BULGAKOV, E. & SADREEV, A. 2008 Bound states in the continuum in photonic waveguides inspired by defects. Phys. Rev. B 78 (7), 075105.   
BULGAKOV, E. N. & SADREEV, A. F. 2014 Robust bound state in the continuum in a nonlinear microcavity embedded in a photonic crystal waveguide. Optics Letters 39 (17), 5212-5215.   
CATTAPAN, G. & LOTTI, P. 2007 Fano resonances in stubbed quantum waveguides with impurities. Eur. Phys. J. B 60 (1), 51–60.   
CHIA WEI HSU, ZHEN, B., CHUA, S.-L., JOHNSON, S. G., JOANNOPOULOS, J. D. & SOLJAČIĆ, M. 2013 Bloch surface eigenstates within the radiation continuum. Light Sci Appl 2 (7), e84.   
CORRIELLI, G., DELLA VALLE, G., CRESPI, A., OSELLAME, R. & LONGHI, S. 2013 Observation of surface states with algebraic localization. Physical Review Letters 111 (22), 220403.   
DITTES, F. 2000 The decay of quantum systems with a small number of open channels. Physics Reports 339 (4), 215–316.   
DUAN, Y., KOCH, W., LINTON, C. M. & MCIVER, M. 2007 Complex resonances and trapped modes in ducted domains. J. Fluid Mech. 571, 119.   
EVANS, D. V., LEVITIN, M. & VASSILIEV, D. 1994 Existence theorems for trapped modes. J. Fluid Mech. 261 (1), 21.   
EVANS, D. V. & LINTON, C. M. 1991 Trapped modes in open channels. J. Fluid Mech. 225 (1), 153.   
FONDA, L. 1961 Resonance reactions and continuous channels. Annals of Physics 12 (3), 476-484.   
FRIEDRICH, H. & WINTGEN, D. 1985 Interfering resonances and bound states in the continuum. Phys. Rev. A 32 (6), 3231-3242.   
LADRÓN DE GUEVARA, M., CLARO, F. & ORELLANA, P. 2003 Ghost fano resonance in a double quantum dot molecule attached to leads. Phys. Rev. B 67 (19), 195335.   
HEIN, S. & KOCH, W. 2008 Acoustic resonances and trapped modes in pipes and tunnels. J. Fluid Mech. 605, 401-428.   
HEIN, S., KOCH, W. & NANNEN, L. 2010 Fano resonances in acoustics. J. Fluid Mech. 664, 238–264.   
HEIN, S., KOCH, W. & NANNEN, L. 2012 Trapped modes and fano resonances in two-dimensional acoustical duct-cavity systems. J. Fluid Mech. 692, 257–287.   
JUNGOWSKI, W. M., BOTROS, K. K. & STUDZINSKI, W. 1989 Cylindrical side-branch as tone generator. Journal of Sound and Vibration 131 (2), 265–285.   
KIM, C., SATANIN, A., JOE, Y. & COSBY, R. 1999 Resonant tunneling in a quantum waveguide: Effect of a finite-size attractive impurity. Phys. Rev. B 60 (15), 10962–10970.   
KRIESELS, P. C., PETERS, M. C. A. M., HIRSCHBERG, A., WIJNANDS, A. P. J., IAFRATI,

A., RICCARDI, G., PIVA, R. & BRUGGEMAN, J.C. 1995 High amplitude vortex-induced pulsations in a gas transport system. Journal of Sound and Vibration 184 (2), 343–368.   
LEPETIT, T., AKMANSOY, E., GANNE, J.-P. & LOURTIOZ, J.-M. 2010 Resonance continuum coupling in high-permittivity dielectric metamaterials. Phys. Rev. B 82 (19), 195307.   
LEPETIT, T. & KANTÉ, B. 2014 Controlling multipolar radiation with symmetries for electromagnetic bound states in the continuum. Phys. Rev. B 90 (24), 241103(R).   
LINTON, C. M. & MCIVER, P. 2007 Embedded trapped modes in water waves and acoustics. Wave Motion 45 (1-2), 16–29.   
MAKSIMOV, D. N. & SADREEV, A. F. 2006 Bound states in elastic waveguides. Phys. Rev. E 74 (1), 016201.   
MAKSIMOV, D. N., SADREEV, A. F., LYAPINA, A. A. & PILIPCHUK, A. S. 2015 Coupled mode theory for acoustic resonators. Wave Motion 56, 52–66.   
MARINICA, D., BORISOV, A. & SHABANOV, S. 2008 Bound states in the continuum in photonics. Physical Review Letters 100 (18), 183902.   
MOISEYEV, N. 1998 Quantum theory of resonances: calculating energies, widths and cross-sections by complex scaling. Physics Reports 302, 212–293.   
VON NEUMANN, J. & WIGNER, E. P. 1929 Über merkwürdige diskrete eigenwerte. Z. Physik 50, 291–293.   
NÖCKEL, J. U. 1992 Resonances in quantum-dot transport. Phys. Rev. B 46 (23), 15348–15356.   
OLENSKI, O. & MIKHAILOVSKA, L. 2002 Bound-state evolution in curved waveguides and quantum wires. Phys. Rev. B 66 (3), 035331.   
PARKER, R. 1966 Resonance effects in wake shedding from parallel plates: Some experimental observations. Journal of Sound and Vibration 4 (1), 62–72.   
PARKER, R. 1967 Resonance effects in wake shedding from parallel plates: Calculation of resonant frequencies. Journal of Sound and Vibration 5 (2), 330–343.   
PICHUGIN, K., SCHANZ, H. & SEBA, P. 2001 Effective coupling for open billiards. Phys. Rev. E 64 (5), 056227.   
PLOTNIK, Y., PELEG, O., DREISOW, F., HEINRICH, M., NOLTE, S., SZAMEIT, A. & SEGEV, M. 2011 Experimental observation of optical bound states in the continuum. Physical Review Letters 107 (18), 183901.   
RACEC, P. N., RACEC, E. R. & NEIDHARDT, H. 2009 Evanescent channels and scattering in cylindrical nanowire heterostructures. Phys. Rev. B 79 (15), 155305.   
SADREEV, A., BULGAKOV, E. & ROTTER, I. 2006 Bound states in the continuum in open quantum billiards with a variable shape. Phys. Rev. B 73 (23), 235342.   
SCHULT, R., RAVENHALL, D. & WYLD, H. 1989 Quantum bound states in a classically unbound system of crossed wires. Phys. Rev. B 39 (8), 5476–5479.   
SHAHBAZYAN, T. & RAIKH, M. 1994 Two-channel resonant tunneling. Phys. Rev. B 49 (24), 17123–17129.   
SHIPMAN, S. & VENAKIDES, S. 2005 Resonant transmission near nonrobust periodic slab modes. Phys. Rev. E 71 (2), 026611.   
STILLINGER, F. & HERRICK, D. 1975 Bound states in the continuum. Phys. Rev. A 11 (2), 446-454.   
SUGIMOTO, N. & IMAHORI, H. 2005 Localized mode of sound in a waveguide with helmholtz resonators. J. Fluid Mech. 546 (1), 89–111.   
TONON, D., HIRSCHBERG, A., GOLLIARD, J. & ZIADA, S. 2011 Aeroacoustics of pipe systems with closed branches. Noise Notes 10 (3), 27–88.   
WEIMANN, S., XU, Y., KEIL, R., MIROSHNICHENKO, A., TÜNNERMANN, A., NOLTE, S., SUKHO-RUKOV, A., SZAMEIT, A. & KIVSHAR, Y. 2013 Compact surface fano states embedded in the continuum of waveguide arrays. Physical Review Letters 111 (24), 240403.   
WIERSIG, J. & MAIN, J. 2008 Fractal weyl law for chaotic microcavities: Fresnel's laws imply multifractal scattering. Phys. Rev. E 77 (3), 036205.   
YANG, Y., PENG, C., LIANG, Y., LI, Z. & NODA, S. 2014 Analytical perspective for bound states in the continuum in photonic crystal slabs. Physical Review Letters 113 (3), 037401.   
ZHANG, M. & ZHANG, X. 2015 Ultrasensitive optical absorption in graphene based on bound states in the continuum. Sci. Rep. 5, 8266.

A. A. Lyapina, D. N. Maksimov, A. S. Pilipchuk, and A. F. Sadreev,

ZIADA, S. & BÜHLMANN, E. T. 1992 Self-excited resonances of two side-branches in close proximity. Journal of Fluids and Structures 6 (5), 583–601.

ZIADA, S. & SHINE, S. 1999 Strouhal numbers of flow-excited acoustic resonance of closed side branches. Journal of fluids and structures 13 (1), 127–142.