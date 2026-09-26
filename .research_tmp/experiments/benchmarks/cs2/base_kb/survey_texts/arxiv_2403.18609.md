A survey on learning models of spiking neural membrane systems and spiking neural networks 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2403.18609v1 [cs.NE] 27 Mar 2024 
 
 
 
 MSC Classification]68Q05, 68Q42, 68Q45, 92D20

 

# A survey on learning models of spiking neural membrane systems and spiking neural networks

 
 
 Prithwineel Paul
 
 Email:  prithwineel.paul@iem.edu.in 
 
 Affiliation:  Department of Computer Science Engineering, Institute of Engineering Management, University of Engineering Management, New Town Rd., Kolkata, 700091, India
 
    
 Petr Sosík
 
 Email:  petr.sosik@fpf.slu.cz 
 
 Affiliation:  Institue of Computer Science, Faculty of Philosophy and Science in Opava, Silesian Univerzity in Opava, Bezručovo náměstí 1150/13 , Opava, 746 01, Czech Rebublic
 
    
 Lucie Ciencialová
 
 Email:  lucie.ciencialova@fpf.slu.cz 
 
 Affiliation:  Institue of Computer Science, Faculty of Philosophy and Science in Opava, Silesian Univerzity in Opava, Bezručovo náměstí 1150/13 , Opava, 746 01, Czech Rebublic
 

 Abstract 
 
 Spiking neural networks (SNN) are a biologically inspired model of neural networks with certain brain-like properties. In the past few decades, this model has received increasing attention in computer science community, owing also to the successful phenomenon of deep learning. In SNN, communication between neurons takes place through the spikes and spike trains. This differentiates these models from the “standard” artificial neural networks (ANN) where the frequency of spikes is replaced by real-valued signals. Spiking neural P systems (SNPS) can be considered a branch of SNN based more on the principles of formal automata, with many variants developed within the framework of the membrane computing theory. In this paper, we first briefly compare structure and function, advantages and drawbacks of SNN and SNPS. A key part of the article is a survey of recent results and applications of machine learning and deep learning models of both SNN and SNPS formalisms.

 
 
 keywords Artificial neural network, Spiking neural network, Spiking neural P system, Machine learning, Deep learning
 
 pacs [
 † † equal-contributors: These authors contributed equally to this work. † † equal-contributors: These authors contributed equally to this work. 
 

## 1 Introduction

 
 Spiking neural networks (SNN) are a brain-inspired model of neural communication and computation using individual spikes to transfer information between individual abstract neurons. These models are known as third-generation neural networks Maass (1997) ; Ghosh-Dastida and
Adeli (2009) . Spiking neurons also follow the principle of integrate-and-fire mechanism. The internal state of the spiking neurons changes with time and when it exceeds a threshold, the postsynaptic neurons spike. Moreover, these neurons encode the information using the timing of the spikes. More specifically, in SNN the encoding of spatio-temporal information is carried by the spikes.
Research into the construction of neuromorphic hardware has gained momentum in the past few years, while it has also promoted the research of SNN. Implementations of SNN promises faster information processing capabilities and also lower energy consumption. However, analog deep learning models are mostly more efficient in real-life applications than SNN which still lack comparably efficient machine learning algorithms as well as programming frameworks. Also, the spiking activity in SNN is discrete and non-differentiable which complicates the implementation of backpropagation-like algorithms based on gradient descent. Therefore, constructing efficient supervised machine learning algorithms for single-layer and multi-layer SNN is a recent research challenge.

 
 
 Spiking neural P systems (SNPS) Ionescu et al. (2006) are a variant of spiking neural networks introduced by Ionescu, Păun, Yokomori in 2006. SNPS are also partly inspired by the formal language and automata theory and they belong to the family of computing models called membrane systems (also P systems) Paun et al. (2010) . Up-to-date information on membrane computing can be found, e.g., in the survey Song et al. (2021) . Like SNN, the encoding of information in SNPS is based on the timing of spikes of specified neurons, or also by a number of discrete spikes. In the last decade, SNPS models have gained popularity among the computer science community because of their applicability to many real-life problems Zhang et al. (2017) . Consequently, many variants of SNPS models have been introduced and their computational capability have been investigated extensively Leporati et al. (2022) .

 
 
 The motivation behind preparing this survey is as follows: (1) Though there exist survey works on different aspects of SNN Wang et al. (2020) ; Auge et al. (2021) ; Nunes et al. (2022) ; Eshraghian
et al. (2023) and SNPS Chen et al. (2021) ; Leporati et al. (2022) , no review covering both models has been published yet.
(2) The main issue of both models is the need for efficient learning methods. Therefore, we focused on machine learning and deep learning algorithms for SNN and SNPS.
(3) The research in both areas accelerated in the last few years and especially new deep learning methods emerged. 

 
 
 The structure of the paper provides simultaneously a key to navigate in the cited literature.
Section 2 lists main types and discusses the structure and function together with their comparison. In Section 3 we discuss the machine learning and deep learning algorithms based on SNN, and a similar study is provided for SNPS models in Section 4 . Finally, Section 5 resumes and concludes the paper.

 
 
 

## 2 Structure and function of SNN and SNPS 

 

### 2.1 Architecture and function of SNN

 
 SNN are inspired by the structure and function of biological neurons which communicate using spikes. The neuron spikes only when the membrane potential exceeds a threshold after receiving some spikes from the neighbouring neurons via synaptic connections. Synapses are associated with adjustable scalar weights whose sign makes them excitatory or inhibitory (increase or decrease the membrane potential).
The presynaptic task of an SNN is to encode an analogue input into spike trains. Different types of encoding techniques are used, such as rate-based method, temporal coding Auge et al. (2021) ; Gerstner et al. (2014) etc. Therefore, the membrane potential of the postsynaptic neurons is modulated by the components of the presynaptic neurons. Hodgkin and Huxley first modelled this phenomenon in 1952 Hodgkin and
Huxley (1952) . Later, many abstract variants of SNN models have been introduced such as spike response model (SRM) Kistler et al. (1997) , the Izhikevich neuron model Izhikevich (2003) , leaky
integrated-and-fire (LIF) neuron Hodgkin and
Huxley (1990) ; Koravuna
et al. (2023) , stochastic SNN Morro et al. (2017) ; Rossello et al. (2012) , self-organizing SNN Hazan et al. (2018) , SNN with adaptive structure Wang et al. (2014) , biorealistic SNN Ambroise
et al. (2013) , probabilistic SNN Kasabov (2010) .
Among further well-known SNN models we mention
memristor-based SNN Fouda et al. (2020) , brain-inspired SNN Yu et al. (2014) , including cerebellar-based robotic SNN controller Zahra et al. (2022) , SNN with noise-threshold Zhang et al. (2017) , stereospike models Rançon
et al. (2022) , xeno-genetic SNN Vellappally
et al. (2018) , Izhikevich SNN Garaffa et al. (2021) , NeuCube Kasabov (2014) , SpinalFlow Narayanan
et al. (2020) , polychronous SNN Wang et al. (2013) , Rmp-SNN Han et al. (2020) , Spiking vgg7 Xiang et al. (2022) , BioLCNet Ghaemi et al. (2023) ,
Spiking Capsnet Zhao et al. (2022) , attention SNN Yao et al. (2023) , AutoSNN Na et al. (2022) and more.

 
 
 In the case of implementation on a special purpose hardware, SNN models can be used to construct energy-efficient spiking hardware responding highly to event-based sensors Khan et al. (2008) ; Han and Roy (2020) ; Tang et al. (2019) , on one hand. On the other hand, the construction of hardware based on SNPS models is still at a developing stage.

 
 
 Industrial applications of SNN lags behind the mass spread of traditional deep learning ANN models. This may be resulting from less efficient training algorithms for deep SNNs, but also from other factors as the non-uniformity in encoding procedures of inputs into discrete spike events. However, SNN have still been successfully used in many real-life applications including simulations in neurophysiology, medical diagnostics, pattern recognition (images, faces, handwriting, speech…), robotics control and much more. We refer the reader to recent survey works Lobo et al. (2020) ; Yamazaki et al. (2022) for more information.

 
 
 

### 2.2 Architecture and function of SNPS

 
 The idea of spiking neurons was formalized mathematically by Ionescu, Păun, Yokomori in Ionescu et al. (2006) . The computing model introduced in Ionescu et al. (2006) is based on formal language and automata theory. Briefly, the model operates in discrete time steps and each neuron accumulates a discrete number of spikes sent to it by other neurons. Each neuron contains a set of action rules of two possible types: (i) a spiking rule which consumes a certain number c c of spikes and emits p ≤ c p\leq c spikes to other neurons; (ii) forgetting rule which just removes all spikes from the neuron. Each spiking rule is associated with a regular expression over integers, and it can be applied when the number of spikes present in the neuron matches this expression. Furthermore, its application can be delayed by a specific number of time steps. We refer the reader to the monograph Paun et al. (2010) for a detailed formal definition.

 
 
 In the past decade, many variants of SNPS models have been introduced. Some of these models are rather theoretical such as asynchronous SNPS Cavaliere
et al. (2009) , SNPS with anti-spikes Pan et al. (2009) , SNPS with polarization Wu et al. (2017) , numerical SNPS Wu et al. (2020) , cell-like SNPS Wu et al. (2016) , SNPS with rules on synapses Song et al. (2014) , SNPS with white-hole neurons Song et al. (2016) , SNPS with pre-computed resources Ishdorj
et al. (2010) , SNPS with communication on request Wu et al. (2022) etc. Furthermore, some variants of SNPS models contain features which are related to cell biology of neurons, such as SNPS with neuron division and budding Pan et al. (2011) , SNPS with structural plasticity Cabarle et al. (2015) , weighted SNPS Wang et al. (2010) , SNPS with astrocytes Pan et al. (2012) , SNPS with multiple channels Peng et al. (2017) ; Liu et al. (2022) , echo SNPS Long et al. (2022) , SNPS with self-organization Wang et al. (2016) , SNPS with scheduled synapses Cabarle et al. (2017) , gated nonlinear SNPS with applications in time series forecasting Zhang et al. (2023) , nonlinear SNPS with two outputs used for edge detection Xian et al. (2023) , a fuzzy membrane control SNPS Liu et al. (2023) etc. The variants of the SNPS listed above mostly enrich the basic SNPS model with various additional features, on one hand. On the other hand, normal forms aimed at the study of restricted variants of the SNPS (while preserving their computational power) were studied in Ibarra et al. (2007) ; García Arnau
et al. (2007) .

 
 
 The formal framework of SNPS models has been experimentally verified in many practical applications ranging from control and optimisation to various artificial intelligence problems and finally to biological modelling problems. The applications include areas such as fault diagnosis of power systems, pattern recognition, image processing, classification problems, image processing, time series forecasting and more. For more details we refer the reader to survey papers Rong et al. (2018) ; Fan et al. (2020) .

 
 
 

### 2.3 Comparison of SNN and SNPS models

 
 Both SNN and SNPS belong to the category of third-generation (spiking) neural networks.
The main objective of the SNN and SNPS models is the temporal encoding of information processed by the spikes of neurons. However, along with these similarities, the two models have also distinctive features. We compare some of the properties of SNN and SNPS in Table 1 .

 
 
 Table 1: Comparison of SNN and SNP models 
 
 
 
 Neuron activation condition | 

 
 
 
 SNN | 
 
 
 Membrane potential exceeds a threshold. 
 | 

 
 SNPS | 
 
 
 The number of accumulated spikes is in the set represented by a regular expression. 
 | 

 
 Inputs and outputs | 

 
 SNN | 
 
 
 Spike trains or real numbers encoded into/decoded from spike trains. 
 | 

 
 SNPS | 
 
 
 Mostly integers encoded as a number of spikes or discrete time delays. 
 | 

 
 Computation process | 

 
 SNN | 
 
 
 Continuous rather than the discrete domain. Nuances in firing rules, no forgetting rules. 
 | 

 
 SNPS | 
 
 
 Operating in discrete time, simple spiking and forgetting rules. Colored spikes and anti-spikes to encode additional information. Fuzzy reasoning SNPS Peng et al. (2013) models resemble SNN in operation. 
 | 

 
 Computability and Complexity | 

 
 SNN | 
 
 
 A few studies of Turing completeness Potok et al. (2022) ; Maass (1996) and computational complexity Fonseca Guerra and
Furber (2017) . Neuromorphic completeness of some models has been discussed in Zhang et al. (2020) ; Rhodes (2020) . 
 | 

 
 SNPS | 
 
 
 Most variants are proved to be Turing complete, solving NP-compete or even PSPACE-complete problems Leporati et al. (2007) ; Pan et al. (2009) ; Song et al. (2014) ; Cabarle et al. (2015) ; Sosík (2019) ; Xu and Jeavons (2013) ; Gatti et al. (2022) . 
 | 

 
 Training | 

 
 SNN | 
 
 
 A broad scope of methods - gradient descent-based, spike time dependent plasticity algorithms, substitution methods, approximate derivative methods, deep learning… 
 | 

 
 SNPS | 
 
 
 Due to the formal language / automata background, training even a single-layer model is rather difficult. Not so many results exist. Chen et al. (2018) ; Peng and Wang (2010) ; Wang and Peng (2013) ; Song et al. (2019) ; Chen et al. (2021) ; Zhang et al. (2022) ; Qiu et al. (2022) ; Zhang et al. (2021) ; Xue et al. (2023) ; Liu et al. (2022) ; Liu et al. (2023) ; Yang et al. (2023) ; Huang et al. (2023) . 
 | 

 

 
 
 In summary, SNPS and SNN models are two independent third-generation neural network models based on communication using spikes. Main distinctive features of SNPS with SNN models are:
(1) a formal language-theoretic structure and motivation of SNPS, in contrast with close biological inspiration of SNN;
(2) SNN has different types of encoding schemes allowing for real-numbered inputs/outputs. In contrast, SNPS models encode information using the number of spikes or discrete time delays. There exist also discretized models of SNN with binary weights or even neuron activation Lu and Sengupta (2020) ; Kheradpisheh and
Masquelier (2020) benefiting from temporal encoding of information. Therefore, they are still distinct from SNPS which operate in discrete time steps.
(3) Much effort is devoted to study the computing capability SNPS models, most of which are Turing complete. The computing power of SNN models is less investigated.

 
 
 
 

## 3 Machine learning in SNN

 
 Efficient machine learning of neural networks as SNN with discrete information processing is one of the most challenging problems in their research. As the forward pass of information through the network represents mostly a non-differentiable function, the classical gradient algorithms cannot be used and alternative solutions must be sought. The revolutionary progress of deep learning during the last decade accelerated this research and many types of machine learning and deep learning algorithms for SNN have been introduced.

 
 
 The weight modification process during learning in SNN is associated with timing of spikes between two connected neurons. The learning in SNN can be divided into three categories, i.e., (1) direct supervised learning (backpropagation-based etc.); (2) Unsupervised learning (spike timing-dependent plasticity – STDP); often, weights on the synapses are modified using the information locally available. This learning process is called local learning and it resembles the learning process in the neurons present in a human brain. (3) Indirect supervised learning (conversion of an ANN to SNN). SNN obtained after translation of pre-trained ANN into an equivalent SNN can have higher representation power and capacity Davidson and
Furber (2021) . A brief summary of learning algorithms for SNN is given in Table 2 .

 
 
 Table 2: Machine learning and deep learning algorithms based on SNN 
 
 
 
 
 
 Machine learning and deep 
 | 
 
 
 Supervised/ 
 | 
 
 
 Real-life 
 | 

 
 
 
 learning models based on SNN 
 | 
 
 
 Unsupervised 
 | 
 
 
 Application 
 | 

 
 
 
 SpikeProp with a BP learning rule Bohte et al. (2000) / 
 with Levenberg-Marquardt algorithm Silva and Ruano (2005) / 
 with multi-spike neurons Booij and tat
Nguyen (2005) 
 | 
 
 
 Supervised 
 | 
 
 
 Non-linear classification tasks / 
 classification of Poisson spike trains 
 
 | 

 
 
 
 SpikeProp with QuickProp or Rprop Xin and Embrechts (2001) ; McKennoch et al. (2006) 
 | 
 
 
 Supervised 
 | 
 
 
 XOR and Fisher Iris data sets 
 | 

 
 
 
 Multi-SpikeProp Ghosh-Dastidar and Adeli (2009) 
 | 
 
 
 Supervised 
 | 
 
 
 Epilepsy and seizure detection 
 | 

 
 
 
 Back-propagation with momentum Fu et al. (2017) 
 | 
 
 
 Supervised 
 | 
 
 
 Wisconsin breast cancer classification 
 | 

 
 
 
 Spatio-temporal back propagation Wu et al. (2018) 
 | 
 
 
 Supervised 
 | 
 
 
 MNIST / N-MNIST dataset 
 | 

 
 
 
 SuperSpike + Hebbian three-factor rule Zenke and Ganguli (2018) ; Zenke and Vogels (2021) 
 | 
 
 
 Supervised 
 | 
 
 
 Randman, MNIST, SHD, RawHD 
 | 

 
 
 
 ReSuMe (Remote Supervised Method) Ponulak and
Kasiński (2010) 
 | 
 
 
 Supervised 
 | 
 
 
 Learning spike trains 
 | 

 
 
 
 Chronotron with STDP Florian (2012) 
 | 
 
 
 Supervised 
 | 
 
 
 Precise spike trains 
 | 

 
 
 
 BP-STDP for multi-layer SNNs Tavanaei and Maida (2019) 
 | 
 
 
 Supervised 
 | 
 
 
 XOR, Iris data, MNIST dataset 
 | 

 
 
 
 Supervised STDP (SSTDP) Liu et al. (2021) 
 | 
 
 
 Supervised 
 | 
 
 
 CALTECH 101 / MNIST / CIFAR-10 datasets 
 | 

 
 
 
 Symmetric STDP Hao et al. (2020) 
 | 
 
 
 Supervised 
 | 
 
 
 Fashion-MNIST dataset 
 | 

 
 
 
 SPAN (spike pattern association neuron) Mohemmed et al. (2012) 
 | 
 
 
 Supervised 
 | 
 
 
 Spike pattern classification 
 | 

 
 
 
 Spike train kernel learning (STKLR) LIN et al. (2016) ; Ma et al. (2018) 
 | 
 
 
 Supervised 
 | 
 
 
 LabelMe image dataset 
 | 

 
 
 
 STDP variants Vigneron and
Martinet (2020) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Pattern recognition 
 | 

 
 
 
 Locally-connected SNN with STDP Saunders et al. (2019) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Learning image features 
 | 

 
 
 
 Spiking CNN with STDP Tavanaei and Maida (2017) 
 | 
 
 
 Unsupervised 
 | 
 
 
 MNIST digit dataset 
 | 

 
 
 
 Self-organizing SNN Hazan et al. (2018) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Decision-making system 
 | 

 
 
 
 SpikeDyn Putra and Shafique (2021) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Image classification 
 | 

 
 
 
 SpiCNN (deep spiking CNN) Lee et al. (2018) 
 | 
 
 
 Supervised 
 | 
 
 
 CALTECH / MNIST datasets 
 | 

 
 
 
 Deep spiking CNN using Tensorflow Vaila et al. (2019) 
 | 
 
 
 Supervised 
 | 
 
 
 MNIST and NM-NIST datasets 
 | 

 
 
 
 Deep residual learning SNN Fang et al. (2021) 
 | 
 
 
 Supervised 
 | 
 
 
 ImageNet, DVS Gesture, CIFAR10-DVS 
 | 

 
 
 
 Spiking-Yolo Kim et al. (2020) 
 | 
 
 
 Supervised 
 | 
 
 
 Object detection 
 | 

 
 
 
 Spiking recurrent NN Kim et al. (2019) 
 | 
 
 
 Supervised 
 | 
 
 
 Cognitive tasks 
 | 

 
 
 
 Adaptive spiking RNN Yin et al. (2021) 
 | 
 
 
 Supervised 
 | 
 
 
 Speech and gesture recognition 
 | 

 
 
 
 Spiking convolutional RNN Xing et al. (2020) 
 | 
 
 
 Supervised 
 | 
 
 
 Hand gesture recognition 
 | 

 
 
 
 Spiking deep belief network (DBN) O’Connor et al. (2013) 
 | 
 
 
 Supervised 
 | 
 
 
 MNIST handwritten digits 
 | 

 
 
 
 Spiking DBN on SpiNNaker Stromatias
et al. (2015a) ; Stromatias et al. (2015b) 
 | 
 
 
 Supervised 
 | 
 
 
 MNIST handwritten digits 
 | 

 
 
 
 Spiking DBN with Siegert neuron model Fatahi et al. (2016) 
 | 
 
 
 Supervised 
 | 
 
 
 Face recognition 
 | 

 
 
 
 Spiking DBN Ganapathy and
Swaminathan (2020) 
 | 
 
 
 Supervised 
 | 
 
 
 Emotion analysis from electrodermal signals 
 | 

 

 
 

### 3.1 Supervised learning algorithms for SNN

 
 Supervised learning algorithms for SNN can be classified according to their principle into three basic groups: gradient descent-based, synaptic plasticity-based and spike train convolution-based. We provide a few examples for each group.

 
 

#### 3.1.1 Gradient descent-based algorithms

 
 The SpikeProp Bohte et al. (2000) training algorithm was introduced in 2002 by Bohte et al. It is a feedforward multilayer SNN framework and state-of-the-art error-backpropagation learning rule for SNN. SpikeProp has been a subject to many extensions and improvements. An improved variant of the SpikeProp algorithm was introduced by Silva et al. in Silva and Ruano (2005) using the Levenberg-Marquardt backpropagation method of training. Another extension of the SpikeProp to SNN with neurons that emit multiple spikes was introduced in 2005 by Booji and Nguyen Booij and tat
Nguyen (2005) ;

 
 
 Resilient propagation (RProp) McKennoch et al. (2006) was designed to speed up the training process with a learning rate adaptation technique depending on the sign of the gradient. Also the QuickProp principle from analog ANNs has been used in Xin and Embrechts (2001) ; McKennoch et al. (2006) and others to speed up the training process of SpikeProp, thanks to useful assumptions about the data and error surface. One of the main features of Quickprop algorithms is the use of Newton’s method for the purpose of minimization of one-dimensional function.

 
 
 Multi-SpikeProp was introduced in 2009 by Ghosh-Dastidar and Adeli Ghosh-Dastidar and Adeli (2009) . It can be seen as an extension of the SpikeProp to train MuSpiNNs (Multi-Spiking Neural Networks) which led to a dramatic increase of efficiency of the model (the authors claim the improvement by two orders of magnitude).

 
 
 Back-propagation with momentum for SNN was presented in Fu et al. (2017) . Learning performance of gradient descent algorithms was improved by a combination of back-propagation with momentum, QuickProp and heuristic rules.

 
 
 Spatio-temporal back propagation introduced in Wu et al. (2018) by Wu et al. is a combination of layer-by-layer spatial domain phase and time-dependent temporal domain phase.

 
 
 

#### 3.1.2 Synaptic plasticity-based algorithms

 
 These algorithms are mostly based on the spike time-dependent plasticity (STDP) learning mechanism Caporale and Dan (2008) , applicable both in supervised and unsupervised training schemes. The STDP learning rules are inspired by Hebbian learning. Synaptic strengths are adjusted based on the correlation of the pre-synaptic and post-synaptic firing times.

 
 
 The ReSuMe (Remote Supervised Method) Ponulak and
Kasiński (2010) was introduced in 2010 by Ponulak and Kasiński. The algorithm is based on a combination of STDP and anti-STDP and later it provided a base for several imporoved variants. The SNN trained in ReSuMe are capable of learning and reproducing complex patterns of spikes.

 
 
 Chronotron introduced in Florian (2012) is an algorithm based on information encoding in the form of precise timing of a train of spikes. Neurons are trained to generate a desired output spike train in a response to a specific spike patterns received as input. Two learning rules have been introduced: E-learning with high memory capacity, and more biologically plausible I-learning.

 
 
 More recently, BP-STDP alorithm Tavanaei and Maida (2019) has been introduced in 2019, using the concepts of gradient descent and STDP applicable in multi-layer SNN, potentially also in deep SNN .

 
 
 Among other recent algorithms combining gradient descent and STDP we can name Supervised STDP Liu et al. (2021) . Further variants of supervised learning algorithms based on STDP have been used, for instance, for training of multi-layer photonic SNN Xiang et al. (2020) . Hao et al. Hao et al. (2020) proposed a neuroscience-plausible symmetric STDP rule which provided competitive results in classification tasks.
From many other applications of STDP-based learning we mention, e.g., touch modality classification Dabbous et al. (2021) , or MNIST image dataset learning Chang et al. (2018) ; Hu et al. (2017) .

 
 
 

#### 3.1.3 Spike train convolution algorithms

 
 SPAN (spike pattern association neuron) Mohemmed et al. (2012) stands for a learning algorithm based on the Widrow–Hoff rule and a transformation of spike trains into analog signals via an α \alpha -kernel function.

 
 
 The use of kernel function turned out to be a promising approach in this class of learning methods. A spike train
kernel learning rule (STKLR) with various kernel functions has been proposed in LIN et al. (2016) . The algorithm was later extended from single-layer to multi-layer networks and several applications have been reported, see, e.g., Ma et al. (2018) .

 
 
 

#### 3.1.4 Resume

 
 In 2020, Wang et al. Wang et al. (2020) prepared a comprehensive survey on supervised learning algorithms for SNN, including experimental comparison of a selected group of algorithms. In general it can be concluded that for feedforward SNNs, algorithms based on STKLR LIN et al. (2016) ; Lin et al. (2017) performed very well and they gave the best score in several tests. However, due to the diverse nature of learning approaches and test criteria, it is rather difficult to provide a unified scoreboard Wang et al. (2020) . Further supervised learning methods for multilayer SNN can be found, e.g., in Sporea and
Grüning (2013) ; Zenke and Ganguli (2018) ; Luo et al. (2022) ; Taherkhani
et al. (2018) ; Xie et al. (2016) ; Lin et al. (2021) and many more. A recent study on supervised training algorithms for SNN inspired by deep learning methods has been published in Eshraghian
et al. (2023) .

 
 
 
 

### 3.2 Unsupervised learning algorithms for SNN

 
 Unsupervised learning models deal with unlabelled data and, instead of supervision, they identify hidden patterns inside the data on their own. A vast majority of unsupervised learning algorithms in SNN are based on the concept of synaptic plasticity. Indeed, the concept of spike time-dependent plasticity (STDP) introduced in the previous section forms a base of many SNN training algorithms. STDP rules can have many variants such as C a C_{a} -STDP, C m C_{m} -STDP, M-STDP, P-STDP etc. A summary of these and further STDP variants can be found in Vigneron and
Martinet (2020) . The use of STDP is mostly found in unsupervised learning Vigneron and
Martinet (2020) , although it is also applied in many supervised and reinforcement learning algorithms, often in combination with gradient descent. In this section we provide several examples of successful unsupervised learning algorithms for SNN.

 
 
 In Saunders et al. Saunders et al. (2019) , an unsupervised feature learning for locally connected spiking neural networks was conceptualized. The paper introduced locally connected SNN learning features of images with locally connected layers of SNN. The model uses the STDP learning algorithms providing, thanks to the local connections, faster convergence than original SNN.

 
 
 Tavanaeri and Maida Tavanaei and Maida (2017) proposed a spiking CNN model containing spiking convolutional-pooling layer and feature discovery layer. An unsupervised local learning has been used to train kernels of the convolution layers and the feature discovery layer consists of probabilistic neurons. A probabilistic STDP rule has been used for training convolution layers of the spiking CNN. The network performed competitively on the MNIST digit dataset.

 
 
 Unsupervised learning of self-organizing spiking neural networks has been introduced in Hazan et al. (2018) and applied to train a rapid decision-making system. The architecture is inspired by classical Kohonen’s self-organizing maps with the capability of clustering unlabeled datasets in an unsupervised manner.

 
 
 SpikeDyn Putra and Shafique (2021) is the name of a framework based on SNN with unsupervised learning in dynamic environments was introduced in. Its design focuses on a restriction of neuronal operations and memory and energy consumption, leading to the decrease of training energy consumption by approx. 50%.

 
 
 In Weidel et al. (2021) , a combined framework of unsupervised learning algorithm and clustered connectivity has been introduced, confirming experimentally an enhanced reinforcement learning capability in SNN.

 
 
 

### 3.3 Deep SNN

 
 Deep neural network (DNN) architectures are primarily based on real-valued ANN and they are composed of multiple hidden layers between the input and the output neurons. DNN are capable of learning complex patterns present inside the data. In the last decade, DNN have become a useful tool for solving problems in image processing, natural language processing, pattern recognition, self-driving cars etc. However, training of deep spiking neural networks is still a very challenging problem. As we have already mentioned, the backpropagation algorithm cannot be directly implemented for deep SNN due to their non-differentiable nature. A review of supervised and unsupervised learning methods for training of deep SNN can be found in Tavanaei et al. (2019) . In this section, we briefly discuss main features of several deep SNN models and their real-life applications such as spiking CNN, spiking RNN, spiking DBN etc.
For instance, the performance of the SNN in solving pattern recognition tasks has been very promising Tavanaei et al. (2019) . A recent study Eshraghian
et al. (2023) summarizes deep learning methods and approaches applicable in learning of deep SNNs.

 
 

#### 3.3.1 Spiking convolutional neural networks

 
 Spiking CNNs are mainly used for image processing and two-dimensional grids, object detection etc. They are mostly based on conversion of already trained CNN into a spiking CNN which is more efficient and consume less energy.
In Lee et al. (2018) , Lee et. al introduced a deep spiking neural network with a hierarchy of stacked convolution layers, i.e., SpiCNN (deep Spiking Convolutional Neural Network). The convolution kernels are trained using spike time-dependent plasticity (STDP) rules which is a well-known method in unsupervised learning.

 
 
 A study on the implementation of spiking CNNs using Tensorflow is performed in Vaila et al. (2019) . A spike-element-wise (SEW) ResNet with residual learning in a deep SNN was presented in Fang et al. (2021) . The authors claim primacy in a direct training of a deep SNNs with more than 100 layers. A channel-wise normalization and neurons with imbalanced threshold were used in Kim et al. (2020) to implement a spiking version of the Yolo network, performing on the level comparable with Yolo Tiny, but with energy consumption lower by two orders of magnitude.

 
 
 

#### 3.3.2 Spiking recurrent neural networks

 
 Recurrent neural networks (RNNs) with classical units as the LSTM (Long short-term memory) or GRU (gated recurrent unit) are primarily used to process sequential data. In this section we focus on some recent implementations of RNNS based on spiking neurons.

 
 
 In Kim et. al. Kim et al. (2019) , a biologically realistic framework for spiking RNNs is introduced and shown to be capable of performing many elementary cognitive tasks.
In Yin et al. (2021) , a novel spiking RNN model with adaptive multiple-timescale spiking neurons was introduced by Yin et al. The model reaches the performance of analog RNNs with the energy consumption lower by one to three orders of magnitude.
Xing et al. Xing et al. (2020) conceptualized a new variant of SNN which combines the properties of convolution and recurrent neural networks. The model is called spiking convolutional recurrent neural network (SCRNN) and it uses a supervised training algorithm called SLAYER (Spike Layer Error Reassignment).
A temporal spiking recurrent network, another novel variant of spiking RNN, was introduced in Wang et al. (2019) . The model has been applied for video recognition and its performance is competitive against SOTA CNN-RNN models in the literature.

 
 
 

#### 3.3.3 Spiking deep belief networks

 
 Deep belief network (DBN) Bengio et al. (2009) is a generative neural network model using stacked layers of Restricted Boltzmann Machines (RBMs). It is used mostly to reconstruct or classify input data (e.g., images ).
In O’Connor et al. (2013) , O’Connor et al. proposed a spiking deep belief network. These models can efficiently classify MNSIT handwritten digits. Later studies in Stromatias
et al. (2015a) ; Stromatias et al. (2015b) showed that DBN on SpiNNaker can achieve classification performance of 95 % \% on the same dataset.

 
 
 In Xue et al. (2017) , Xue et al. investigated the object recognition capacity of spiking DBN models.
Face recognition capability of spiking DBN models has been evaluated on ORL dataset in Fatahi et al. (2016) . Spiking DBN models have also been used for emotional analysis using electrodermal signals Ganapathy and
Swaminathan (2020) . The study on spiking convolution deep belief networks has gained interest of researchers due to its capacity to learn high-level features from dynamic vision sensor datasets Kaiser et al. (2017) . However, the accuracy of the spiking DBN on the MNIST dataset was shown lower than that of other types on SNNs Tavanaei et al. (2019) .

 
 
 
 
 

## 4 Learning algorithms for SNPS

 
 Table 3: Machine learning and deep learning algorithms based on SNPS 
 
 
 
 
 
 Machine learning and deep 
 | 
 
 
 Supervised / 
 | 
 
 
 Real-life 
 | 

 
 
 
 learning models based on SNPS 
 | 
 
 
 Unsupervised 
 | 
 
 
 Application 
 | 

 
 
 
 SNPS with Hebbian learning Chen et al. (2018) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Identification of nuclear export signal 
 | 

 
 
 
 Adaptive SNPS with Widrow-Hoff learning algorithm Peng and Wang (2010) 
 | 
 
 
 Supervised 
 | 
 
 
 Fault diagnosis 
 | 

 
 
 
 Adaptive fuzzy SNPS with Widrow-Hoff learning algorithm Wang and Peng (2013) 
 | 
 
 
 Supervised 
 | 
 
 
 Fault diagnosis 
 | 

 
 
 
 SNPS with learning function Song et al. (2019) (Hebbian) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Recognize digital English letters 
 | 

 
 
 
 Associative memory network based on SNPS with white holes
+ Hebbian learning Chen et al. (2021) 
 | 
 
 
 Unsupervised 
 | 
 
 
 Identification of digits 
 | 

 
 
 
 Layered SNPS Zhang et al. (2022) 
 | 
 
 
 Supervised 
 | 
 
 
 Classification problem 
 | 

 
 
 
 Deep dynamic SNPS (SNPS + CNN) Qiu et al. (2022) 
 | 
 
 
 Supervised (ensemble learning) 
 | 
 
 
 Organ segmentation 
 | 

 
 
 
 Hypergraph-based SNPS Xue et al. (2023) 
 | 
 
 
 Supervised (ensemble learning) 
 | 
 
 
 predicting the survival time of glioblastoma patients 
 | 

 
 
 
 LSTM-SNP Liu et al. (2022) 
 | 
 
 
 Supervised 
 | 
 
 
 Time series forecasting 
 | 

 
 
 
 Nonlinear SNPS Liu et al. (2023) 
 | 
 
 
 Supervised (linear machine learning) 
 | 
 
 
 multivariate time series forecasting 
 | 

 
 
 
 Spiking neural P-type Dual-channel dilated convolutional network (SDDC-Net) Yang et al. (2023) 
 | 
 
 
 Supervised 
 | 
 
 
 Retinal vessel segmentation 
 | 

 
 
 
 (bidirectional) LSTM-SNP Huang et al. (2023) ; Liu et al. (2023) 
 | 
 
 
 Supervised 
 | 
 
 
 Sentiment classification 
 | 

 
 
 
 Load forecasting non-linear SNPS (LF-NSNPS) Long et al. (2022) 
 | 
 
 
 Supervised 
 | 
 
 
 Electricity load forecasting 
 | 

 
 
 
 Learning SNPS with belief AdaBoost Zhang et al. (2021) 
 | 
 
 
 Supervised learning 
 | 
 
 
 Fault diagnosis 
 | 

 

 
 
 In recent years, machine learning and deep learning frameworks have been introduced also for SNPS models. In Gutiérrez-Naranjo and
Pérez-Jiménez (2009) , Gutiérrez-Naranjo et al. introduced the first study of introducing Hebbian learning in the SNPS framework. This study was further continued by many authors and the SNPS model with Hebbian learning method has been applied to various tasks. Among them we mention, e.g., the analysis of nuclear export signals Chen et al. (2018) , recognition of digital English letters Song et al. (2019) , identification of digits Chen et al. (2021) . Hebbian learning belongs to the category of unsupervised machine learning.
The model introduced in Chen et al. (2018) has two modules, i.e., (i) input and (ii) predict module. The rules used in this model are E / a c → a p E/a^{c}\rightarrow a^{p} and a s → λ a^{s}\rightarrow\lambda . The Hebbian learning strategy is used in the predict module. It is important to note that the topology of the model remains unchanged in the input module. However, it changes in the predict module. To perform the task of identification of nuclear export signal (NES), at first, it is encoded into a binary string. Moreover, the model in Chen et al. (2018) has a better precision rate than NES-REBS, Wregex, ELM and NetNES. In Song et al. (2019) , Song et. al. introduced an SNPS-based machine-learning model for recognition of digital English letters. The spiking rule used in Song et al. (2019) is E / a c → a E/a^{c}\rightarrow a . Similarly, as in Chen et al. (2018) , synaptic connections change dynamically using learning function. Also, this model is divided into two modules, i.e., in (i) the input module; (ii) the recognize module. The recognize module is divided into innermost, middle and outermost layers. This model performs better than SNNs in recognizing letters with noise. In Chen et al. (2021) , Hebbian learning is used for the identification of digits. The working of the model is inspired by the Hopfield network. It uses only spiking rules of the form a ∗ / a a ​ l ​ l → a a^{*}/a^{all}\rightarrow a and the set of input and output neurons are the same. Finally, the synaptic weights are updated using the energy function.

 
 
 Over the years, researchers from the membrane computing community have developed several supervised machine learning algorithms based on spiking neural P systems. The first of them, based on the Widrow-Hoff rule, was proposed in 2010 by Peng and Wang Peng and Wang (2010) . In this model, a neuron σ i = ( α i , ω i , R i ) ; \sigma_{i}=(\alpha_{i},\omega_{i},R_{i}); has rules of the form E / a α → a α E/a^{\alpha}\rightarrow a^{\alpha} where α \alpha represents the potential of the neuron σ i \sigma_{i} instead of number of spikes.
This study was further extended in Wang and Peng (2013) . In Wang and Peng (2013) , Wang et. al. introduced adaptive fuzzy spiking neural P systems. This model is capable of performing dynamic fuzzy reasoning using the weighted fuzzy reasoning rules, where the fuzzy reasoning algorithm and the training mechanism are based on the firing mechanism of the neurons.
Moreover, this model has two types of neurons, i.e., proposition neuron and rule neuron. A proposition neuron is a 4 4 -tuple σ i = ( α i , ω i , λ i , r i CLOSE \sigma_{i}=(\alpha_{i},\omega_{i},\lambda_{i},r_{i} ), where α i ∈ [ 0 , 1 ] , ω i \alpha_{i}\in[0,1],\omega_{i} (weight vector), λ i ∈ [ 0 , 1 ) \lambda_{i}\in[0,1) (threshold) and has spiking rule E / a α → a α E/a^{\alpha}\rightarrow a^{\alpha} . Similarly, rule neuron is represented by a 4 4 -tuple σ r ​ i = ( α i , γ i , τ i , r i ) \sigma_{ri}=(\alpha_{i},\gamma_{i},\tau_{i},r_{i}) , where α i , γ i ∈ [ 0 , 1 ] \alpha_{i},\gamma_{i}\in[0,1] and has spiking rule E / a α → a β , α , β ∈ [ 0 , 1 ] E/a^{\alpha}\rightarrow a^{\beta},\alpha,\beta\in[0,1] . Moreover, synaptic connections exist only between proposition and rule neurons.
In Zhang et al. (2022) , a new variant of SNPS model, i.e., layered spiking neural P system was introduced by Zhang et. al to solve classification problems. This model contains multiple weighted SNPS and weight-adjusting rules. Furthermore, the performance of layered SNPS in solving classification problems has been verified on the MNIST dataset.
The model in Zhang et al. (2022) is highly-flexible, simple and has a
multi-layer weighted SNPS with adaptive weight adjustment rule. Similarly, as in Wang and Peng (2013) ; Peng and Wang (2010) , the model contains proposition neurons and rule neurons with similar spiking rules. However unlike in Wang and Peng (2013) ; Peng and Wang (2010) , it contains a high dimensional encoder (which exists between the pre-processing and input layer) and a weighted fuzzy SNPS classifier. It has input, hidden, comparison, selection and output layers. The main difference between WFSNPS and LSNPS is that the first one has one layer and the latter has more than one layer. Furthermore, it has excellent learning ability and convergence.

 
 
 In Zhang et al. (2021) , Zhang et. al introduced a novel learning spiking neural P system with the belief AdaBoost for identification of faults in oil-immersed power transformers. Similarly, as in Wang and Peng (2013) ; Peng and Wang (2010) ; Zhang et al. (2022) , it has proposition and rule neurons.
Moreover, the experimental implementation of the models shows that it has higher accuracy in the identification of faults in comparison with back-propagation neural networks, support vector machine etc.

 
 
 In Xue et al. (2023) , a new variant of hypergraph-based neural network, i.e., hypergraph-based numerical P system was introduced. In recent years, hypergraph neural networks have become an attractive direction for research due to their superior performance in many domains. The P system variant of the hypergraph neural network in Xue et al. (2023) can represent the high-order neural connections and can be implemented for the segmentation of tumours/organs in medical images. This model also outperforms state-of-the-art methods which are based on hippocampal datasets and multiple brain metastases datasets.
It consists of three new classes of neurons with Plane, Transmembrane and Hierarchical rules to describe higher-order relationships among neurons. The time consumed for the segmentation of images was also reduced.
In Qiu et al. (2022) , Qiu et. al. proposed a novel organ segmentation method based on a deep dynamic SNPS model. It is a combination of features from SNPS and CNN. It has two subsystems (i) locating pancreas (ii) accurate segmentation of the pancreas. Accuracy of this model is lower than some state-of-the-art methods. Again, the system is represented as Π = ( O , C , σ i ​ L , s ​ y ​ n , M , σ i ​ n , σ o ​ u ​ t ) \Pi=(O,C,\sigma_{iL},syn,M,\sigma_{in},\sigma_{out}) where C C represents coefficients of the spike and the neuron σ i = ( s i , r i ) \sigma_{i}=(s_{i},r_{i}) has rules of the form [ a s → a p ] i [a^{s}\rightarrow a^{p}]_{i} , [ a s → λ ] i , [ a s ] i → [ λ ] j [ λ ] k ( i ≠ j ≠ k ) [a^{s}\rightarrow\lambda]_{i},[a^{s}]_{i}\rightarrow[\lambda]_{j}[\lambda]_{k}(i\neq j\neq k) .

 
 
 Long short-term memory is a novel variant of neural networks. Based on the properties of LSTM, a new variant, i.e., LSTM-SNP model was introduced in Liu et al. (2022) . LSTM-SNP also has been implemented for time series forecasting.
It is a recurrent-type model and can process sequential data. Again, it has
nonlinear spiking mechanisms (nonlinear spike consumption and
generation) and nonlinear gate functions (reset, consumption,
and generation gates). The neurons in this model have spiking rule of the form T / g ⁡ ( x i ) → f ⁡ ( x i ) T/g(x_{i})\rightarrow f(x_{i}) where T T represents a threshold, g ⁡ ( x i ) g(x_{i}) is a linear or non-linear function, f ⁡ ( x i ) f(x_{i}) is a non-linear function. LSTM-SNP model has better predictive performance than some state-of-the-art prediction models.
Also, it has a lower predictive performance for some time series.

 
 
 In Liu et al. (2023) , Liu et. al proposed a new variant of SNPS model as well as a recurrent neural network, i.e., SNP systems with autapses. These models contain three non-linear gate functions and a recurrent-type prediction model is constructed based on this model.
In summary, it has four mechanisms, i.e., (1) Non-linear spiking; (2) recurrent mechanism; (3) Non-linear gate mechanism; (4) autapses mechanism.
Furthermore, this model has been used in chaotic time series and experimental results demonstrate the effectiveness of this model in time series forecasting.
The neurons in Liu et al. (2023) are of the form σ i = ( u i , r i , ( ρ 1 i , ρ 2 i , ρ 3 i ) CLOSE \sigma_{i}=(u_{i},r_{i},(\rho_{1}^{i},\rho_{2}^{i},\rho_{3}^{i}) where u i u_{i} represents internal state and the gate function is represented by ρ i \rho^{i} . The spiking rule used in this model is of the form OPEN T / a g ⁡ ( u ) → a f ⁡ ( u ) ) T/a^{g(u)}\rightarrow a^{f(u)}) where T T is a threshold and f , g f,g are non-linear functions. This model also can Predict Lorenz
chaotic time series efficiently.
In Long et al. (2022) , Long et. al. used non-linear spiking neural P systems for MST-based time series forecasting. In this method, wavelet transformations are used for the transformation of time series data into frequency domain data during the training phase. Next, the non-linear SNPS model is trained using the frequency domain data. This model is also capable of generating sequential data automatically as prediction results. More specifically, the proposed method in Long et al. (2022) is capable of processing non-stationary time series problems. The spiking rules used in Long et al. (2022) is similar to the rules in Liu et al. (2023) . However, the model has some extra properties such as (i) synaptic weights are associated with the neurons; (ii) neurons do not have any forgetting rules and have only one spiking rule. Moreover, the constraint g ⁡ ( x ) ≥ f ⁡ ( x ) ≥ 0 g(x)\geq f(x)\geq 0 has been removed from the spiking rules.

 
 
 In Huang et al. (2023) , a modified LSTM-SNP model, i.e., bidirectional LSTM-SNP (BiLSTM-SNP) was introduced by Huang et. al. Moreover, aspect-level sentiment classification model based on BiLSTM-SNP has been introduced. The semantic correlation between the aspect word and content word has been investigated using this model and to understand the effectiveness of the model, it has been implemented on English and Chinese datasets.
This model is a variant of non-linear SNPS and it is a combination of a forward LSTM-SNP (which extracts the contextual semantic information towards aspect word) and a backward LSTM-SNP (which obtains the semantic correlation between aspect word and content words). It also outperforms some variants of LSTM models.

 
 
 Recently, convolution neural networks have gained prominence due to their image and video processing capacity and perform tasks such as facial recognition, object detection, image classification etc very efficiently.
In Yang et al. (2023) ,
an U-shaped deep CNN model, i.e., spiking neural P-type Dual-channel dilated convolution network (SDDC-Net) is proposed by Yang et. al.
It has the classic encoder-decoder architecture containing convolutional neurons which makes it different from the standard SNPS model.
This model has excellent segmentation, and generalization ability and SNP U-Net increases the accuracy.

 
 
 

## 5 Conclusion

 
 This survey focuses on the relation of spiking neural networks (SNN) and their variant popular in the field of membrane computing, i.e, spiking neural P systems (SNPS). We compared these models based on their properties, architectures and real-life applications. Most importantly, we discussed a range of machine learning algorithms for SNN and SNPS. In order to keep this study to a reasonable length, we could not cover all related topics, such as the issue of real-time recurrent learning (RTRL), for which we refer to the recent source Eshraghian
et al. (2023) .

 
 
 Although the research in both SNN and SNPS provides a series of theoretical results and interesting applications, it is fair to note that their efficient learning remains a challenging problem and their applicability in deep learning tasks cannot recently compete with that of real-valued deep ANN models. Therefore, among open and challenging research problems in SNN and SNPS fields we mention first an efficient training of multi-layer SNN and SNPS. An interesting topic is their possible hardware implementation which promises a lower energy consumption than in the case of classical ANN. Thanks to their closer relation to biological brain structure, a possible use of SNN / SNPS models in sentiment classification seems interesting. Another promising application topic is the analysis of time series where the potential of spiking neuronal model is high due to their dynamical representation of data patterns Ghosh-Dastida and
Adeli (2009) . Finally, an efficient implementation of the spike timing-dependent plasticity learning Vigneron and
Martinet (2020) in SNPS could greatly improve their learning potential.

 
 

#### Acknowledgements

 
 Supported by the Silesian University in Opava under the Student Funding Plan, project SGS/11/2023.

 
 
 
 

## References

 
 
 Maass (1997) 
 
Maass, W.:
Networks of spiking neurons: the third generation of neural network
models.
Neural Networks
 10 (9),
1659–1671
(1997)

 

 
 Ghosh-Dastida and
Adeli (2009) 
 
Ghosh-Dastida, S.,
Adeli, H.:
Spiking neural networks.
International Journal of Neural Systems
 19 (04),
295–308
(2009)

 

 
 Ionescu et al. (2006) 
 
Ionescu, M.,
Păun, G.,
Yokomori, T.:
Spiking neural P systems.
Fundamenta informaticae
 71 (2-3),
279–308
(2006)

 

 
 Paun et al. (2010) 
 
Paun, G.,
Rozenberg, G.,
Salomaa, A.:
The Oxford Handbook of Membrane Computing.
Oxford University Press, Inc., ???
(2010)

 

 
 Song et al. (2021) 
 
Song, B.,
Li, K.,
Orellana-Martín, D.,
Pérez-Jiménez, M.J.,
Pérez-Hurtado, I.:
A survey of nature-inspired computing: Membrane computing.
ACM Computing Surveys (CSUR)
 54 (1),
1–31
(2021)

 

 
 Zhang et al. (2017) 
 
Zhang, G.,
Pérez-Jiménez, M.J.,
Gheorghe, M.:
Real-life Applications with Membrane Computing
vol. 25.
Springer, ???
(2017)

 

 
 Leporati et al. (2022) 
 
Leporati, A.,
Mauri, G.,
Zandron, C.:
Spiking neural P systems: main ideas and results.
Natural Computing
 21 (4),
629–649
(2022)

 

 
 Wang et al. (2020) 
 
Wang, X.,
Lin, X.,
Dang, X.:
Supervised learning in spiking neural networks: A review of algorithms
and evaluations.
Neural Networks
 125 ,
258–280
(2020)

 

 
 Auge et al. (2021) 
 
Auge, D.,
Hille, J.,
Mueller, E.,
Knoll, A.:
A survey of encoding techniques for signal processing in spiking
neural networks.
Neural Processing Letters
 53 (6),
4693–4710
(2021)

 

 
 Nunes et al. (2022) 
 
Nunes, J.D.,
Carvalho, M.,
Carneiro, D.,
Cardoso, J.S.:
Spiking neural networks: A survey.
IEEE Access
 10 ,
60738–60764
(2022)

 

 
 Eshraghian
et al. (2023) 
 
Eshraghian, J.K.,
Ward, M.,
Neftci, E.O.,
Wang, X.,
Lenz, G.,
Dwivedi, G.,
Bennamoun, M.,
Jeong, D.S.,
Lu, W.D.:
Training spiking neural networks using lessons from deep learning.
Proceedings of the IEEE
(2023)

 

 
 Chen et al. (2021) 
 
Chen, Y.,
Chen, Y.,
Zhang, G.,
Paul, P.,
Wu, T.,
Zhang, X.,
Rong, H.,
Ma, X.:
A survey of learning spiking neural P systems and a novel instance.
International Journal of Unconventional Computing
 16 
(2021)

 

 
 Gerstner et al. (2014) 
 
Gerstner, W.,
Kistler, W.M.,
Naud, R.,
Paninski, L.:
Neuronal Dynamics: From Single Neurons to Networks and Models of
Cognition.
Cambridge University Press, ???
(2014)

 

 
 Hodgkin and
Huxley (1952) 
 
Hodgkin, A.L.,
Huxley, A.F.:
A quantitative description of membrane current and its application to
conduction and excitation in nerve.
The Journal of physiology
 117 (4),
500
(1952)

 

 
 Kistler et al. (1997) 
 
Kistler, W.M.,
Gerstner, W.,
Hemmen, J.L.v.:
Reduction of the hodgkin-huxley equations to a single-variable
threshold model.
Neural computation
 9 (5),
1015–1045
(1997)

 

 
 Izhikevich (2003) 
 
Izhikevich, E.M.:
Simple model of spiking neurons.
IEEE Transactions on neural networks
 14 (6),
1569–1572
(2003)

 

 
 Hodgkin and
Huxley (1990) 
 
Hodgkin, A.,
Huxley, A.:
A quantitative description of membrane current and its application to
conduction and excitation in nerve.
Bulletin of mathematical biology
 52 ,
25–71
(1990)

 

 
 Koravuna
et al. (2023) 
 
Koravuna, S.,
Rückert, U.,
Jungeblut, T., et al.:
Evaluation of spiking neural nets-based image classification using the runtime
simulator ravsim.
International Journal of Neural Systems,
2350044–2350044
(2023)

 

 
 Morro et al. (2017) 
 
Morro, A.,
Canals, V.,
Oliver, A.,
Alomar, M.L.,
Galán-Prado, F.,
Ballester, P.J.,
Rosselló, J.L.:
A stochastic spiking neural network for virtual screening.
IEEE transactions on neural networks and learning systems
 29 (4),
1371–1375
(2017)

 

 
 Rossello et al. (2012) 
 
Rossello, J.L.,
Canals, V.,
Morro, A.,
Oliver, A.:
Hardware implementation of stochastic spiking neural networks.
International Journal of Neural Systems
 22 (04),
1250014
(2012)

 

 
 Hazan et al. (2018) 
 
Hazan, H.,
Saunders, D.,
Sanghavi, D.T.,
Siegelmann, H.,
Kozma, R.:
Unsupervised learning with self-organizing spiking neural networks.
In: 2018 International Joint Conference on Neural Networks (IJCNN),
Rio de Janeiro, Brazil,
pp. 1–6
(2018).
IEEE

 

 
 Wang et al. (2014) 
 
Wang, J.,
Belatreche, A.,
Maguire, L.,
McGinnity, T.M.:
An online supervised learning method for spiking neural networks with
adaptive structure.
Neurocomputing
 144 ,
526–536
(2014)

 

 
 Ambroise
et al. (2013) 
 
Ambroise, M.,
Levi, T.,
Bornat, Y.,
Saighi, S.:
Biorealistic spiking neural network on fpga.
In: 2013 47th Annual Conference on Information Sciences and Systems
(CISS),
Princeton, USA,
pp. 1–6
(2013).
IEEE

 

 
 Kasabov (2010) 
 
Kasabov, N.:
To spike or not to spike: A probabilistic spiking neuron model.
Neural Networks
 23 (1),
16–19
(2010)

 

 
 Fouda et al. (2020) 
 
Fouda, M.E.,
Kurdahi, F.,
Eltawil, A.,
Neftci, E.:
Chapter 19 - spiking neural networks for inference and learning: a
memristor-based design perspective.
In: Spiga, S.,
Sebastian, A.,
Querlioz, D.,
Rajendran, B. (eds.)
Memristive Devices for Brain-Inspired Computing.
Woodhead Publishing Series in Electronic and Optical Materials,
pp. 499–530.
Woodhead Publishing, ???
(2020)

 

 
 Yu et al. (2014) 
 
Yu, Q.,
Tang, H.,
Tan, K.C.,
Yu, H.:
A brain-inspired spiking neural network model with temporal encoding
and learning.
Neurocomputing
 138 ,
3–13
(2014)

 

 
 Zahra et al. (2022) 
 
Zahra, O.,
Navarro-Alarcon, D.,
Tolu, S.:
A neurorobotic embodiment for exploring the dynamical interactions of
a spiking cerebellar model and a robot arm during vision-based manipulation
tasks.
International Journal of Neural Systems
 32 (08),
2150028
(2022)

 

 
 Zhang et al. (2017) 
 
Zhang, M.,
Qu, H.,
Xie, X.,
Kurths, J.:
Supervised learning in spiking neural networks with noise-threshold.
Neurocomputing
 219 ,
333–349
(2017)

 

 
 Rançon
et al. (2022) 
 
Rançon, U.,
Cuadrado-Anibarro, J.,
Cottereau, B.R.,
Masquelier, T.:
Stereospike: Depth learning with a spiking neural network.
IEEE Access
 10 ,
127428–127439
(2022)

 

 
 Vellappally
et al. (2018) 
 
Vellappally, S.,
Abdullah Al-Kheraif, A.,
Anil, S.,
Basavarajappa, S.,
Hassanein, A.S.:
Maintaining patient oral health by using a xeno-genetic spiking neural
network.
Journal of Ambient Intelligence and Humanized Computing
 0 ,
1–9
(2018)

 

 
 Garaffa et al. (2021) 
 
Garaffa, L.C.,
Aljuffri, A.,
Reinbrecht, C.,
Hamdioui, S.,
Taouil, M.,
Sepulveda, J.:
Revealing the secrets of spiking neural networks: The case of
izhikevich neuron.
In: 2021 24th Euromicro Conference on Digital System Design (DSD),
Palermo, Sicily, Italy,
pp. 514–518
(2021).
IEEE

 

 
 Kasabov (2014) 
 
Kasabov, N.K.:
Neucube: A spiking neural network architecture for mapping, learning
and understanding of spatio-temporal brain data.
Neural Networks
 52 ,
62–76
(2014)

 

 
 Narayanan
et al. (2020) 
 
Narayanan, S.,
Taht, K.,
Balasubramonian, R.,
Giacomin, E.,
Gaillardon, P.-E.:
Spinalflow: An architecture and dataflow tailored for spiking neural
networks.
In: 2020 ACM/IEEE 47th Annual International Symposium on Computer
Architecture (ISCA),
Virtual,
pp. 349–362
(2020).
IEEE

 

 
 Wang et al. (2013) 
 
Wang, R.,
Cohen, G.,
Stiefel, K.M.,
Hamilton, T.J.,
Tapson, J.,
Schaik, A.:
An fpga implementation of a polychronous spiking neural network with
delay adaptation.
Frontiers in neuroscience
 7 ,
14
(2013)

 

 
 Han et al. (2020) 
 
Han, B.,
Srinivasan, G.,
Roy, K.:
Rmp-snn: Residual membrane potential neuron for enabling deeper
high-accuracy and low-latency spiking neural network.
In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition,
Virtual,
pp. 13558–13567
(2020)

 

 
 Xiang et al. (2022) 
 
Xiang, S.,
Jiang, S.,
Liu, X.,
Zhang, T.,
Yu, L.:
Spiking vgg7: Deep convolutional spiking neural network with direct
training for object recognition.
Electronics
 11 (13),
2097
(2022)

 

 
 Ghaemi et al. (2023) 
 
Ghaemi, H.,
Mirzaei, E.,
Nouri, M.,
Kheradpisheh, S.R.:
Biolcnet: Reward-modulated locally connected spiking neural networks.
In: Machine Learning, Optimization, and Data Science: 8th
International Conference, Part II,
Certosa di Pontignano, Italy,
pp. 564–578
(2023).
Springer

 

 
 Zhao et al. (2022) 
 
Zhao, D.,
Li, Y.,
Zeng, Y.,
Wang, J.,
Zhang, Q.:
Spiking capsnet: A spiking neural network with a biologically
plausible routing rule between capsules.
Information Sciences
 610 ,
1–13
(2022)

 

 
 Yao et al. (2023) 
 
Yao, M.,
Zhao, G.,
Zhang, H.,
Hu, Y.,
Deng, L.,
Tian, Y.,
Xu, B.,
Li, G.:
Attention spiking neural networks.
IEEE Transactions on Pattern Analysis and Machine Intelligence
 45 (8),
9393–9410
(2023)

 

 
 Na et al. (2022) 
 
Na, B.,
Mok, J.,
Park, S.,
Lee, D.,
Choe, H.,
Yoon, S.:
Autosnn: towards energy-efficient spiking neural networks.
In: International Conference on Machine Learning,
Baltimore MD, USA,
pp. 16253–16269
(2022).
PMLR

 

 
 Khan et al. (2008) 
 
Khan, M.M.,
Lester, D.R.,
Plana, L.A.,
Rast, A.,
Jin, X.,
Painkras, E.,
Furber, S.B.:
SpiNNaker: mapping neural networks onto a massively-parallel chip
multiprocessor.
In: 2008 IEEE International Joint Conference on Neural Networks,
Hong Kong, China,
pp. 2849–2856
(2008).
Ieee

 

 
 Han and Roy (2020) 
 
Han, B.,
Roy, K.:
Deep spiking neural network: Energy efficiency through time based
coding.
In: Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK,
pp. 388–404
(2020).
Springer

 

 
 Tang et al. (2019) 
 
Tang, G.,
Shah, A.,
Michmizos, K.P.:
Spiking neural network on neuromorphic hardware for energy-efficient
unidimensional slam.
In: 2019 IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS),
The Venetian Macao, Macau, China,
pp. 4176–4181
(2019).
IEEE

 

 
 Lobo et al. (2020) 
 
Lobo, J.L.,
Del Ser, J.,
Bifet, A.,
Kasabov, N.:
Spiking neural networks and online learning: An overview and
perspectives.
Neural Networks
 121 ,
88–100
(2020)

 

 
 Yamazaki et al. (2022) 
 
Yamazaki, K.,
Vo-Ho, V.-K.,
Bulsara, D.,
Le, N.:
Spiking neural networks and their applications: A review.
Brain Sciences
 12 (7),
863
(2022)

 

 
 Cavaliere
et al. (2009) 
 
Cavaliere, M.,
Ibarra, O.H.,
Păun, G.,
Egecioglu, O.,
Ionescu, M.,
Woodworth, S.:
Asynchronous spiking neural P systems.
Theoretical Computer Science
 410 (24-25),
2352–2364
(2009)

 

 
 Pan et al. (2009) 
 
Pan, L.,
Paun, G.,
Pérez Jiménez, M.J.:
Spiking neural P systems with neuron division and budding.
In: Proceedings of the Seventh Brainstorming Week on Membrane
Computing, Vol. II, 151-167. Sevilla, ETS de Ingeniería Informática,
2-6 de Febrero, 2009.
Fénix Editora,
Sevilla, Spain
(2009)

 

 
 Wu et al. (2017) 
 
Wu, T.,
Păun, A.,
Zhang, Z.,
Pan, L.:
Spiking neural P systems with polarizations.
IEEE transactions on neural networks and learning systems
 29 (8),
3349–3360
(2017)

 

 
 Wu et al. (2020) 
 
Wu, T.,
Pan, L.,
Yu, Q.,
Tan, K.C.:
Numerical spiking neural P systems.
IEEE Transactions on Neural Networks and Learning Systems
 32 (6),
2443–2457
(2020)

 

 
 Wu et al. (2016) 
 
Wu, T.,
Zhang, Z.,
Păun, G.,
Pan, L.:
Cell-like spiking neural P systems.
Theoretical Computer Science
 623 ,
180–189
(2016)

 

 
 Song et al. (2014) 
 
Song, T.,
Pan, L.,
Păun, G.:
Spiking neural P systems with rules on synapses.
Theoretical Computer Science
 529 ,
82–95
(2014)

 

 
 Song et al. (2016) 
 
Song, T.,
Gong, F.,
Liu, X.,
Zhao, Y.,
Zhang, X.:
Spiking neural P systems with white hole neurons.
IEEE transactions on nanobioscience
 15 (7),
666–673
(2016)

 

 
 Ishdorj
et al. (2010) 
 
Ishdorj, T.-O.,
Leporati, A.,
Pan, L.,
Zeng, X.,
Zhang, X.:
Deterministic solutions to qsat and q3sat by spiking neural P
systems with pre-computed resources.
Theoretical Computer Science
 411 (25),
2345–2358
(2010)

 

 
 Wu et al. (2022) 
 
Wu, T.,
Neri, F.,
Pan, L.:
On the tuning of the computation capability of spiking neural membrane
systems with communication on request.
International Journal of Neural Systems
 32 (08),
2250037
(2022)

 

 
 Pan et al. (2011) 
 
Pan, L.,
Păun, G.,
Pérez-Jiménez, M.J.:
Spiking neural P systems with neuron division and budding.
Science China Information Sciences
 54 ,
1596–1607
(2011)

 

 
 Cabarle et al. (2015) 
 
Cabarle, F.G.C.,
Adorna, H.N.,
Pérez-Jiménez, M.J.,
Song, T.:
Spiking neural P systems with structural plasticity.
Neural Computing and Applications
 26 ,
1905–1917
(2015)

 

 
 Wang et al. (2010) 
 
Wang, J.,
Hoogeboom, H.J.,
Pan, L.,
Păun, G.,
Pérez-Jiménez, M.J.:
Spiking neural P systems with weights.
Neural computation
 22 (10),
2615–2646
(2010)

 

 
 Pan et al. (2012) 
 
Pan, L.,
Zeng, X.,
Zhang, X.,
Jiang, Y.:
Spiking neural P systems with weighted synapses.
Neural Processing Letters
 35 ,
13–27
(2012)

 

 
 Peng et al. (2017) 
 
Peng, H.,
Yang, J.,
Wang, J.,
Wang, T.,
Sun, Z.,
Song, X.,
Luo, X.,
Huang, X.:
Spiking neural P systems with multiple channels.
Neural Networks
 95 ,
66–71
(2017)

 

 
 Liu et al. (2022) 
 
Liu, M.,
Zhao, F.,
Jiang, X.,
Zhang, H.,
Zhou, H.:
Parallel binary image cryptosystem via spiking neural networks
variants.
International Journal of Neural Systems
 32 (08),
2150014
(2022)

 

 
 Long et al. (2022) 
 
Long, L.,
Lugu, R.,
Xiong, X.,
Liu, Q.,
Peng, H.,
Wang, J.,
Orellana-Martín, D.,
Pérez-Jiménez, M.J.:
Echo spiking neural P systems.
Knowledge-Based Systems
 253 ,
109568
(2022)

 

 
 Wang et al. (2016) 
 
Wang, X.,
Song, T.,
Gong, F.,
Zheng, P.:
On the computational power of spiking neural P systems with
self-organization.
Scientific reports
 6 (1),
1–16
(2016)

 

 
 Cabarle et al. (2017) 
 
Cabarle, F.G.C.,
Adorna, H.N.,
Jiang, M.,
Zeng, X.:
Spiking neural P systems with scheduled synapses.
IEEE Transactions on Nanobioscience
 16 (8),
792–801
(2017)

 

 
 Zhang et al. (2023) 
 
Zhang, Y.,
Yang, Q.,
Liu, Z.,
Peng, H.,
Wang, J.:
A prediction model based on gated nonlinear spiking neural systems.
International Journal of Neural Systems
 34 (9),
2350029
(2023)

 

 
 Xian et al. (2023) 
 
Xian, R.,
Lugu, R.,
Peng, H.,
Yang, Q.,
Luo, X.,
Wang, J.:
Edge detection method based on nonlinear spiking neural systems.
International Journal of Neural Systems
 33 (06),
2250060–1225006014
(2023)

 

 
 Liu et al. (2023) 
 
Liu, X.,
Zhang, G.,
Mastoi, M.S.,
Neri, F.,
Pu, Y.:
A human-simulated fuzzy membrane approach for the joint controller of walking
biped robots.
Integrated Computer-Aided Engineering
(Preprint),
1–16
(2023)

 

 
 Ibarra et al. (2007) 
 
Ibarra, O.H.,
Păun, A.,
Păun, G.,
Rodríguez-Patón, A.,
Sosík, P.,
Woodworth, S.:
Normal forms for spiking neural P systems.
Theoretical Computer Science
 372 (2-3),
196–217
(2007)

 

 
 García Arnau
et al. (2007) 
 
García Arnau, M.,
Peréz, D.,
Rodríguez Patón, A.,
Sosík, P.:
Spiking neural P systems: Stronger normal forms.
In: Proceedings of the Fifth Brainstorming Week on Membrane
Computing,
pp. 157–178.
Fénix Editora,
Sevilla
(2007).
Universidad de Sevilla

 

 
 Rong et al. (2018) 
 
Rong, H.,
Wu, T.,
Pan, L.,
Zhang, G.:
Spiking neural P systems: Theoretical results and applications.
Enjoying Natural Computing: Essays Dedicated to Mario de Jesús
Pérez-Jiménez on the Occasion of His 70th Birthday
 11270 ,
256–268
(2018)

 

 
 Fan et al. (2020) 
 
Fan, S.,
Paul, P.,
Wu, T.,
Rong, H.,
Zhang, G.:
On applications of spiking neural P systems.
Applied Sciences
 10 (20),
7011
(2020)

 

 
 Peng et al. (2013) 
 
Peng, H.,
Wang, J.,
Pérez-Jiménez, M.J.,
Wang, H.,
Shao, J.,
Wang, T.:
Fuzzy reasoning spiking neural P system for fault diagnosis.
Information Sciences
 235 ,
106–116
(2013)

 

 
 Potok et al. (2022) 
 
Potok, T.,
Schuman, C.,
Kay, B., et al.:
Neuromorphic computing is turing-complete.
Technical report,
Oak Ridge National Lab.(ORNL), Oak Ridge, TN (United States)
(2022)

 

 
 Maass (1996) 
 
Maass, W.:
Lower bounds for the computational power of networks of spiking
neurons.
Neural computation
 8 (1),
1–40
(1996)

 

 
 Fonseca Guerra and
Furber (2017) 
 
Fonseca Guerra, G.A.,
Furber, S.B.:
Using stochastic spiking neural networks on SpiNNaker to solve
constraint satisfaction problems.
Frontiers in neuroscience
 11 ,
714
(2017)

 

 
 Zhang et al. (2020) 
 
Zhang, Y.,
Qu, P.,
Ji, Y.,
Zhang, W.,
Gao, G.,
Wang, G.,
Song, S.,
Li, G.,
Chen, W.,
Zheng, W., et al. :
A system hierarchy for brain-inspired computing.
Nature
 586 (7829),
378–384
(2020)

 

 
 Rhodes (2020) 
 
Rhodes, O.:
Brain-inspired computing boosted by new concept of completeness.
Nature Publishing Group UK London
(2020)

 

 
 Leporati et al. (2007) 
 
Leporati, A.,
Zandron, C.,
Ferretti, C.,
Mauri, G.:
Solving numerical NP-complete problems with spiking neural P
systems.
In: Membrane Computing: 8th International Workshop, WMC 2007,
Thessaloniki, Greece,
pp. 336–352
(2007).
Springer

 

 
 Song et al. (2014) 
 
Song, T.,
Luo, L.,
He, J.,
Chen, Z.,
Zhang, K.:
Solving subset sum problems by time-free spiking neural P systems.
Applied Mathematics Information Sciences
 8 (1),
327
(2014)

 

 
 Sosík (2019) 
 
Sosík, P.:
P systems attacking hard problems beyond NP: a survey.
Journal of Membrane Computing
 1 ,
198–208
(2019)

 

 
 Xu and Jeavons (2013) 
 
Xu, L.,
Jeavons, P.:
Simple neural-like P systems for maximal independent set selection.
Neural Computation
 25 (6),
1642–1659
(2013)

 

 
 Gatti et al. (2022) 
 
Gatti, M.,
Leporati, A.,
Zandron, C.:
On spiking neural membrane systems with neuron and synapse creation.
International Journal of Neural Systems
 32 (08),
2250036
(2022)

 

 
 Chen et al. (2018) 
 
Chen, Z.,
Zhang, P.,
Wang, X.,
Shi, X.,
Wu, T.,
Zheng, P.:
A computational approach for nuclear export signals identification
using spiking neural P systems.
Neural Computing and Applications
 29 ,
695–705
(2018)

 

 
 Peng and Wang (2010) 
 
Peng, H.,
Wang, J.:
Adaptive spiking neural P systems.
In: 2010 Sixth International Conference on Natural Computation,
vol. 6.
Yantai, Shandong, China,
pp. 3008–3011
(2010).
IEEE

 

 
 Wang and Peng (2013) 
 
Wang, J.,
Peng, H.:
Adaptive fuzzy spiking neural P systems for fuzzy inference and
learning.
International Journal of Computer Mathematics
 90 (4),
857–868
(2013)

 

 
 Song et al. (2019) 
 
Song, T.,
Pan, L.,
Wu, T.,
Zheng, P.,
Wong, M.D.,
Rodríguez-Patón, A.:
Spiking neural P systems with learning functions.
IEEE transactions on nanobioscience
 18 (2),
176–190
(2019)

 

 
 Zhang et al. (2022) 
 
Zhang, G.,
Zhang, X.,
Rong, H.,
Paul, P.,
Zhu, M.,
Neri, F.,
Ong, Y.-S.:
A layered spiking neural system for classification problems.
International Journal of Neural Systems
 32 (08),
2250023
(2022)

 

 
 Qiu et al. (2022) 
 
Qiu, C.,
Xue, J.,
Liu, X.,
Li, Q.:
Deep dynamic spiking neural P systems with applications in organ
segmentation.
Journal of Membrane Computing
 4 ,
1–12
(2022)

 

 
 Zhang et al. (2021) 
 
Zhang, X.,
Zhang, G.,
Paul, P.,
Zhang, J.,
Wu, T.,
Fan, S.,
Xiong, X.:
Dissolved gas analysis for transformer fault based on learning spiking neural
P system with belief adaboost.
International Journal of Unconventional Computing
 16 
(2021)

 

 
 Xue et al. (2023) 
 
Xue, J.,
Ren, L.,
Song, B.,
Guo, Y.,
Lu, J.,
Liu, X.,
Gong, G.,
Li, D.:
Hypergraph-based numerical neural-like P systems for medical image
segmentation.
IEEE Transactions on Parallel and Distributed Systems
 34 (4),
1202–1214
(2023)

 

 
 Liu et al. (2022) 
 
Liu, Q.,
Long, L.,
Yang, Q.,
Peng, H.,
Wang, J.,
Luo, X.:
Lstm-snp: A long short-term memory model inspired from spiking neural
P systems.
Knowledge-Based Systems
 235 ,
107656
(2022)

 

 
 Liu et al. (2023) 
 
Liu, Q.,
Peng, H.,
Long, L.,
Wang, J.,
Yang, Q.,
Pérez-Jiménez, M.J.,
Orellana-Martín, D.:
Nonlinear spiking neural systems with autapses for predicting chaotic time
series.
IEEE Transactions on Cybernetics
 Early Access 
(2023)

 

 
 Yang et al. (2023) 
 
Yang, B.,
Qin, L.,
Peng, H.,
Guo, C.,
Luo, X.,
Wang, J.:
Sddc-net: A u-shaped deep spiking neural P convolutional network for
retinal vessel segmentation.
Digital Signal Processing
 136 ,
104002
(2023)

 

 
 Huang et al. (2023) 
 
Huang, Y.,
Liu, Q.,
Peng, H.,
Wang, J.,
Yang, Q.,
Orellana-Martín, D.:
Sentiment classification using bidirectional lstm-snp model and
attention mechanism.
Expert Systems with Applications
 221 ,
119730
(2023)

 

 
 Lu and Sengupta (2020) 
 
Lu, S.,
Sengupta, A.:
Exploring the connection between binary and spiking neural networks.
Frontiers in neuroscience
 14 ,
535
(2020)

 

 
 Kheradpisheh and
Masquelier (2020) 
 
Kheradpisheh, S.R.,
Masquelier, T.:
Temporal backpropagation for spiking neural networks with one spike
per neuron.
International Journal of Neural Systems
 30 (06),
2050027
(2020)

 

 
 Davidson and
Furber (2021) 
 
Davidson, S.,
Furber, S.B.:
Comparison of artificial and spiking neural networks on digital
hardware.
Frontiers in Neuroscience
 15 ,
651141
(2021)

 

 
 Bohte et al. (2000) 
 
Bohte, S.M.,
Kok, J.N.,
La Poutré, J.A.:
SpikeProp: backpropagation for networks of spiking neurons.
In: 8th European Symposium on Artificial Neural Networks, ESANN 2000,
vol. 48.
Bruges, Belgium,
pp. 419–424
(2000)

 

 
 Silva and Ruano (2005) 
 
Silva, S.M.,
Ruano, A.E.:
Application of Levenberg-Marquardt method to the training of spiking
neural networks.
In: 2005 International Conference on Neural Networks and Brain,
vol. 3.
Beijing, China,
pp. 1354–1358
(2005).
IEEE

 

 
 Booij and tat
Nguyen (2005) 
 
Booij, O.,
Nguyen, H.:
A gradient descent rule for spiking neurons emitting multiple spikes.
Information Processing Letters
 95 (6),
552–558
(2005)

 

 
 Xin and Embrechts (2001) 
 
Xin, J.,
Embrechts, M.J.:
Supervised learning with spiking neural networks.
In: IJCNN’01. International Joint Conference on Neural Networks.
Proceedings (Cat. No. 01CH37222),
vol. 3.
Washington, DC, USA,
pp. 1772–1777
(2001).
IEEE

 

 
 McKennoch et al. (2006) 
 
McKennoch, S.,
Liu, D.,
Bushnell, L.G.:
Fast modifications of the SpikeProp algorithm.
In: The 2006 IEEE International Joint Conference on Neural Network
Proceedings,
Vancouver, BC, Canada,
pp. 3970–3977
(2006).
IEEE

 

 
 Ghosh-Dastidar and Adeli (2009) 
 
Ghosh-Dastidar, S.,
Adeli, H.:
A new supervised learning algorithm for multiple spiking neural
networks with application in epilepsy and seizure detection.
Neural Networks
 22 (10),
1419–1431
(2009)

 

 
 Fu et al. (2017) 
 
Fu, Q.,
Luo, Y.,
Liu, J.,
Bi, J.,
Qiu, S.,
Cao, Y.,
Ding, X.:
Improving learning algorithm performance for spiking neural networks.
In: 2017 IEEE 17th International Conference on Communication
Technology (ICCT),
Chengdu, China,
pp. 1916–1919
(2017).
IEEE

 

 
 Wu et al. (2018) 
 
Wu, Y.,
Deng, L.,
Li, G.,
Zhu, J.,
Shi, L.:
Spatio-temporal backpropagation for training high-performance spiking
neural networks.
Frontiers in neuroscience
 12 ,
331
(2018)

 

 
 Zenke and Ganguli (2018) 
 
Zenke, F.,
Ganguli, S.:
Superspike: Supervised learning in multilayer spiking neural
networks.
Neural computation
 30 (6),
1514–1541
(2018)

 

 
 Zenke and Vogels (2021) 
 
Zenke, F.,
Vogels, T.P.:
The remarkable robustness of surrogate gradient learning for
instilling complex function in spiking neural networks.
Neural computation
 33 (4),
899–925
(2021)

 

 
 Ponulak and
Kasiński (2010) 
 
Ponulak, F.,
Kasiński, A.:
Supervised learning in spiking neural networks with resume: sequence
learning, classification, and spike shifting.
Neural computation
 22 (2),
467–510
(2010)

 

 
 Florian (2012) 
 
Florian, R.V.:
The chronotron: A neuron that learns to fire temporally precise spike
patterns.
PloS one
 7 (8),
40233
(2012)

 

 
 Tavanaei and Maida (2019) 
 
Tavanaei, A.,
Maida, A.:
BP-STDP: Approximating backpropagation using spike timing dependent
plasticity.
Neurocomputing
 330 ,
39–47
(2019)

 

 
 Liu et al. (2021) 
 
Liu, F.,
Zhao, W.,
Chen, Y.,
Wang, Z.,
Yang, T.,
Jiang, L.:
SSTDP: Supervised spike timing dependent plasticity for efficient
spiking neural network training.
Frontiers in Neuroscience
 15 ,
756876
(2021)

 

 
 Hao et al. (2020) 
 
Hao, Y.,
Huang, X.,
Dong, M.,
Xu, B.:
A biologically plausible supervised learning method for spiking neural
networks using the symmetric stdp rule.
Neural Networks
 121 ,
387–395
(2020)

 

 
 Mohemmed et al. (2012) 
 
Mohemmed, A.,
Schliebs, S.,
Matsuda, S.,
Kasabov, N.:
Span: Spike pattern association neuron for learning spatio-temporal
spike patterns.
International Journal of Neural Systems
 22 (04),
1250012
(2012)

 

 
 LIN et al. (2016) 
 
LIN, X.,
WANG, X.,
DANG, X.:
A new supervised learning algorithm for spiking neurons based on spike
train kernels.
Acta Electronica Sinica
 44 (12),
2877
(2016)

 

 
 Ma et al. (2018) 
 
Ma, Q.,
Lin, X.,
Wang, X.:
Supervised learning of single-layer spiking neural networks for image
classification.
IOP conference series: materials science and engineering
 435 (1),
012049
(2018).
IOP Publishing

 

 
 Vigneron and
Martinet (2020) 
 
Vigneron, A.,
Martinet, J.:
A critical survey of STDP in spiking neural networks for pattern
recognition.
In: 2020 International Joint Conference on Neural Networks (IJCNN),
Glasgow, UK,
pp. 1–9
(2020).
IEEE

 

 
 Saunders et al. (2019) 
 
Saunders, D.J.,
Patel, D.,
Hazan, H.,
Siegelmann, H.T.,
Kozma, R.:
Locally connected spiking neural networks for unsupervised feature
learning.
Neural Networks
 119 ,
332–340
(2019)

 

 
 Tavanaei and Maida (2017) 
 
Tavanaei, A.,
Maida, A.S.:
Multi-layer unsupervised learning in a spiking convolutional neural
network.
In: 2017 International Joint Conference on Neural Networks (IJCNN),
Anchorage, AK, USA,
pp. 2023–2030
(2017).
IEEE

 

 
 Putra and Shafique (2021) 
 
Putra, R.V.W.,
Shafique, M.:
Spikedyn: A framework for energy-efficient spiking neural networks
with continual and unsupervised learning capabilities in dynamic
environments.
In: 2021 58th ACM/IEEE Design Automation Conference (DAC),
San Francisco, CA, USA,
pp. 1057–1062
(2021).
IEEE

 

 
 Lee et al. (2018) 
 
Lee, C.,
Srinivasan, G.,
Panda, P.,
Roy, K.:
Deep spiking convolutional neural network trained with unsupervised
spike-timing-dependent plasticity.
IEEE Transactions on Cognitive and Developmental Systems
 11 (3),
384–394
(2018)

 

 
 Vaila et al. (2019) 
 
Vaila, R.,
Chiasson, J.,
Saxena, V.:
Feature extraction using spiking convolutional neural networks.
In: Proceedings of the International Conference on Neuromorphic
Systems,
Knoxville, TN, USA,
pp. 1–8
(2019)

 

 
 Fang et al. (2021) 
 
Fang, W.,
Yu, Z.,
Chen, Y.,
Huang, T.,
Masquelier, T.,
Tian, Y.:
Deep residual learning in spiking neural networks.
Advances in Neural Information Processing Systems
 34 ,
21056–21069
(2021)

 

 
 Kim et al. (2020) 
 
Kim, S.,
Park, S.,
Na, B.,
Yoon, S.:
Spiking-Yolo: spiking neural network for energy-efficient object
detection.
In: Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 34.
New York, USA,
pp. 11270–11277
(2020)

 

 
 Kim et al. (2019) 
 
Kim, R.,
Li, Y.,
Sejnowski, T.J.:
Simple framework for constructing functional spiking recurrent neural
networks.
Proceedings of the national academy of sciences
 116 (45),
22811–22820
(2019)

 

 
 Yin et al. (2021) 
 
Yin, B.,
Corradi, F.,
Bohté, S.M.:
Accurate and efficient time-domain classification with adaptive
spiking recurrent neural networks.
Nature Machine Intelligence
 3 (10),
905–913
(2021)

 

 
 Xing et al. (2020) 
 
Xing, Y.,
Di Caterina, G.,
Soraghan, J.:
A new spiking convolutional recurrent neural network (SCRNN) with
applications to event-based hand gesture recognition.
Frontiers in neuroscience
 14 ,
590164
(2020)

 

 
 O’Connor et al. (2013) 
 
O’Connor, P.,
Neil, D.,
Liu, S.-C.,
Delbruck, T.,
Pfeiffer, M.:
Real-time classification and sensor fusion with a spiking deep belief
network.
Frontiers in neuroscience
 7 ,
178
(2013)

 

 
 Stromatias
et al. (2015a) 
 
Stromatias, E.,
Neil, D.,
Galluppi, F.,
Pfeiffer, M.,
Liu, S.-C.,
Furber, S.:
Scalable energy-efficient, low-latency implementations of trained
spiking deep belief networks on SpiNNaker.
In: 2015 International Joint Conference on Neural Networks (IJCNN),
San Francisco, CA, USA,
pp. 1–8
(2015).
IEEE

 

 
 Stromatias et al. (2015b) 
 
Stromatias, E.,
Neil, D.,
Galluppi, F.,
Pfeiffer, M.,
Liu, S.-C.,
Furber, S.:
Live demonstration: handwritten digit recognition using spiking deep
belief networks on SpiNNaker.
In: 2015 IEEE International Symposium on Circuits and Systems
(ISCAS),
San Francisco, CA, USA,
pp. 1901–1901
(2015).
IEEE

 

 
 Fatahi et al. (2016) 
 
Fatahi, M.,
Ahmadi, M.,
Ahmadi, A.,
Shahsavari, M.,
Devienne, P.:
Towards an spiking deep belief network for face recognition
application.
In: 2016 6th International Conference on Computer and Knowledge
Engineering (ICCKE),
San Francisco, CA, USA,
pp. 153–158
(2016).
IEEE

 

 
 Ganapathy and
Swaminathan (2020) 
 
Ganapathy, N.,
Swaminathan, R.:
Emotion analysis using electrodermal signals and spiking deep belief
network.
In: al., L.B.P.-H. (ed.)
Digital Personalized Health and Medicine,
pp. 1269–1270.
IOS Press, ???
(2020)

 

 
 Caporale and Dan (2008) 
 
Caporale, N.,
Dan, Y.:
Spike timing–dependent plasticity: a hebbian learning rule.
Annu. Rev. Neurosci.
 31 ,
25–46
(2008)

 

 
 Xiang et al. (2020) 
 
Xiang, S.,
Ren, Z.,
Zhang, Y.,
Song, Z.,
Guo, X.,
Han, G.,
Hao, Y.:
Training a multi-layer photonic spiking neural network with modified
supervised learning algorithm based on photonic STDP.
IEEE Journal of Selected Topics in Quantum Electronics
 27 (2),
1–9
(2020)

 

 
 Dabbous et al. (2021) 
 
Dabbous, A.,
Ibrahim, A.,
Valle, M.,
Bartolozzi, C.:
Touch modality classification using spiking neural networks and
supervised-STDP learning.
In: 2021 28th IEEE International Conference on Electronics, Circuits,
and Systems (ICECS),
Dubai, United Arab Emirates,
pp. 1–4
(2021).
IEEE

 

 
 Chang et al. (2018) 
 
Chang, C.-C.,
Chen, P.-C.,
Hudec, B.,
Liu, P.-T.,
Hou, T.-H.:
Interchangeable hebbian and anti-hebbian STDP applied to supervised
learning in spiking neural network.
In: 2018 IEEE International Electron Devices Meeting (IEDM),
San Francisco, CA, USA,
pp. 15–5
(2018).
IEEE

 

 
 Hu et al. (2017) 
 
Hu, Z.,
Wang, T.,
Hu, X.:
An STDP-based supervised learning algorithm for spiking neural
networks.
In: Neural Information Processing: 24th International Conference,
ICONIP 2017, Part II,
Guangzhou, China,
pp. 92–100
(2017).
Springer

 

 
 Lin et al. (2017) 
 
Lin, X.,
Wang, X.,
Hao, Z.:
Supervised learning in multilayer spiking neural networks with inner
products of spike trains.
Neurocomputing
 237 ,
59–70
(2017)

 

 
 Sporea and
Grüning (2013) 
 
Sporea, I.,
Grüning, A.:
Supervised learning in multilayer spiking neural networks.
Neural computation
 25 (2),
473–509
(2013)

 

 
 Luo et al. (2022) 
 
Luo, X.,
Qu, H.,
Wang, Y.,
Yi, Z.,
Zhang, J.,
Zhang, M.:
Supervised learning in multilayer spiking neural networks with spike temporal
error backpropagation.
IEEE Transactions on Neural Networks and Learning Systems
 Early access 
(2022)

 

 
 Taherkhani
et al. (2018) 
 
Taherkhani, A.,
Belatreche, A.,
Li, Y.,
Maguire, L.P.:
A supervised learning algorithm for learning precise timing of
multiple spikes in multilayer spiking neural networks.
IEEE transactions on neural networks and learning systems
 29 (11),
5394–5407
(2018)

 

 
 Xie et al. (2016) 
 
Xie, X.,
Qu, H.,
Liu, G.,
Zhang, M.,
Kurths, J.:
An efficient supervised training algorithm for multilayer spiking
neural networks.
PloS one
 11 (4),
0150329
(2016)

 

 
 Lin et al. (2021) 
 
Lin, X.,
Zhang, M.,
Wang, X., et al.:
Supervised learning algorithm for multilayer spiking neural networks with
long-term memory spike response model.
Computational Intelligence and Neuroscience
 2021 
(2021)

 

 
 Weidel et al. (2021) 
 
Weidel, P.,
Duarte, R.,
Morrison, A.:
Unsupervised learning and clustered connectivity enhance reinforcement
learning in spiking neural networks.
Frontiers in computational neuroscience
 15 ,
543872
(2021)

 

 
 Tavanaei et al. (2019) 
 
Tavanaei, A.,
Ghodrati, M.,
Kheradpisheh, S.R.,
Masquelier, T.,
Maida, A.:
Deep learning in spiking neural networks.
Neural Networks
 111 ,
47–63
(2019)

 

 
 Wang et al. (2019) 
 
Wang, W.,
Hao, S.,
Wei, Y.,
Xiao, S.,
Feng, J.,
Sebe, N.:
Temporal spiking recurrent neural network for action recognition.
IEEE Access
 7 ,
117165–117175
(2019)

 

 
 Bengio et al. (2009) 
 
Bengio, Y., et al. :
Learning deep architectures for AI.
Foundations and trends in Machine Learning
 2 (1),
1–127
(2009)

 

 
 Xue et al. (2017) 
 
Xue, F.,
Chen, X.,
Li, X.:
Real-time classification through a spiking deep belief network with
intrinsic plasticity.
In: Advances in Neural Networks-ISNN 2017: 14th International
Symposium, ISNN 2017, Part I 14,
New York, USA,
pp. 188–196
(2017).
Springer

 

 
 Kaiser et al. (2017) 
 
Kaiser, J.,
Zimmerer, D.,
Tieck, J.C.V.,
Ulbrich, S.,
Roennau, A.,
Dillmann, R.:
Spiking convolutional deep belief networks.
In: Artificial Neural Networks and Machine Learning–ICANN 2017, Part
II,
New York, USA,
pp. 3–11
(2017).
Springer

 

 
 Liu et al. (2023) 
 
Liu, Q.,
Huang, Y.,
Yang, Q.,
Peng, H.,
Wang, J.:
An attention-aware long short-term memory-like spiking neural model for
sentiment analysis.
International Journal of Neural Systems,
2350037–2350037
(2023)

 

 
 Long et al. (2022) 
 
Long, L.,
Liu, Q.,
Peng, H.,
Yang, Q.,
Luo, X.,
Wang, J.,
Song, X.:
A time series forecasting approach based on nonlinear spiking neural
systems.
International Journal of Neural Systems
 32 (08),
2250020
(2022)

 

 
 Gutiérrez-Naranjo and
Pérez-Jiménez (2009) 
 
Gutiérrez-Naranjo, M.A.,
Pérez-Jiménez, M.J.:
Hebbian learning from spiking neural P systems view.
In: Membrane Computing: 9th International Workshop, WMC 2008,
Edinburgh, UK,
pp. 217–230
(2009).
Springer