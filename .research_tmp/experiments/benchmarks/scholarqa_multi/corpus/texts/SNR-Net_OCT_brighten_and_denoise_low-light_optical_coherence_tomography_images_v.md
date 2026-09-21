# SNR-Net OCT: brighten and denoise low-light optical coherence tomography images via deep learning

SHAOYAN HUANG, $^{1,\dagger}$ RONG WANG, $^{2,\dagger}$ RENXIONG WU, $^{1}$ JUNMING ZHONG, $^{1}$ XIN GE, $^{3}$ YONG LIU, $^{1}$ AND GUANGMING NI $^{1,\star}$

*guangmingni@uestc.edu.cn

Abstract: Low-light optical coherence tomography (OCT) images generated when using low input power, low-quantum-efficiency detection units, low exposure time, or facing high-reflective surfaces, have low bright and signal-to-noise rates (SNR), and restrict OCT technique and clinical applications. While low input power, low quantum efficiency, and low exposure time can help reduce the hardware requirements and accelerate imaging speed; high-reflective surfaces are unavoidable sometimes.

Here we propose a deep-learning-based technique to brighten and denoise low-light OCT images, termed SNR-Net OCT. The proposed SNR-Net OCT deeply integrated a conventional OCT setup and a residual-dense-block U-Net generative adversarial network with channel-wise attention connections trained using a customized large speckle-free SNR-enhanced brighter OCT dataset. Results demonstrated that the proposed SNR-Net OCT can brighten low-light OCT images and remove the speckle noise effectively, with enhancing SNR and maintaining the tissue microstructures well.

Moreover, compared to the hardware-based techniques, the proposed SNR-Net OCT can be of lower cost and better performance.

© 2023 Optica Publishing Group under the terms of the Optica Open Access Publishing Agreement

# 1. Introduction

Optical coherence tomography (OCT), as a non-invasive and radiation-free optical imaging device [1], has been widely used in clinical applications including ophthalmology [2], cardiology [3], gastroenterology [4], and endoscopy [5], due to its micron-scale imaging resolution, depth-slicing, and three-dimensional structure resolving capabilities.

However, when facing low input power, low-quantum-efficiency detection unit [6], low exposure time for high imaging speed, or facing high-reflective surfaces, low-light OCT images having low signal-to-noise rates (SNR) occur, which are highly degraded [7] and restrict OCT clinical applications. Nonetheless, low-power input light, low-quantum-efficiency detection unit, and low exposure time can help reduce the hardware requirements and accelerate imaging speed, respectively, and high-reflective surfaces are unavoidable sometimes.

Besides, OCT is consequently susceptible to speckle noise caused by the mutual interference of all the backscattered waves, which also deteriorates OCT SNR and imposes significant limitations on its diagnostic capabilities [8].

When facing low-light imaging with low SNR, one conventional hardware-based strategy to bright low-light OCT images and enhance their SNR is increasing imaging exposure time or input light power [6] when the quantum efficiency of the detection unit is fixed, which can increase the number of photons entering the detection unit and enhance the imaging light. However, increasing imaging exposure time inevitably reduces imaging speed, and using higher light power is always limited by the illuminant technique and safety considerations. In addition,

Check for updates

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20696

Optics EXPRESS

491391

Journal © 2023

https://doi.org/10.1364/OE.491391

Received 23 Mar 2023; revised 22 May 2023; accepted 23 May 2023; published 5 Jun 2023

some other methods are based on hardware such as using a dual-axis architecture [9]. By detecting multiple forward scattered photons [10], OCT brightness and SNR can be extended to a certain extent, but this method also seriously reduces the imaging speed and increases the data size processed, and has limited ability to remove speckle noise.

Another proposed hardware-based approach is spectral-domain optical coherence tomography with dual-balanced detection [11], which utilizes a multiline single-camera spectrometer to acquire a combination of two phase-opposed interferometric spectra for the OCT SNR enhancement. However, all the methods mentioned above can't fix the low-light problem when facing cases of low-power input, low quantum efficiency, etc., and they also could not remove the speckle.

Some software-based image processing methods are used to improve OCT SNR, such as the 2D-filter combination algorithm [12], BM3D [13], and sparse representation [14]. However, these methods cannot effectively work for low-light OCT images and suppress speckle noise, and are somewhat prone to over-smoothing, leading to limited OCT SNR enhancement.

In recent years, with the rise of deep learning, the inspiring capabilities of neural networks in image processing, especially in the medical field, have attracted widespread attention [15,16]. For example, with the deep learning techniques, speckle noise in ophthalmological OCT imaging has been addressed well, such as the results shown in the work of Wang et al. [17], Huang et al. [18], Zhou et al. [19], etc.

However, speckle noise in non-ophthalmological OCT images has still been a difficult challenge to overcome, which restricts the SNR enhancement via deep learning, because of the missing proper ground truths which can't be obtained by averaging operation. Further, those works haven't shown how to brighten low-light OCT images and enhance the image SNR now. Recently, the advent of speckle-modulating OCT has made great progress in removing speckle noise but it badly reduces OCT sample-arm input-light power and temporal resolution [20].

Our recent works [8,21] have proposed a deep-learning-based speckle-modulating OCT, which deeply integrated the conventional OCT setup and deep learning networks trained with a customized large speckle-modulating OCT dataset. The proposed deep-learning-based speckle-modulating OCT has been proven that it can effectively remove OCT speckle noise without reducing OCT sample-arm input-light power and temporal resolution, paying the way for OCT SNR enhancement via deep learning methods.

Here we proposed a deep-learning-based technique to brighten and denoise low-light OCT images, termed SNR-Net OCT, which can brighten low-light OCT images, remove speckle noise, and preserve the microstructures in OCT images better, with effectively enhancing SNR. The proposed SNR-Net OCT was deeply integrated with conventional OCT setup by using deep learning networks and a large speckle-free SNR-enhanced brighter dataset.

Here we conducted experiments on Scotch tape, meat, and medical OCT images under low-light imaging conditions, and demonstrated the valuable imaging performance and the clinical application prospects of the proposed SNR-Net OCT. Meanwhile, the proposed technique SNR-Net OCT can also help use low-cost hardware units, such as low-power light sources or low-performance detectors, to achieve a high-performance OCT system using high-cost hardware units.

# 2. Method

# 2.1. Dataset preparation

Here a customized-design spectral-domain optical coherence tomography (SD-OCT) [22] system was built to collect the deep-learning training and testing datasets by configuring different imaging exposure times, as shown in Fig. 1(a). Specifically, a broadband light source (cBLMD-T-850-HP, Superlum) with a center wavelength of $\lambda c = 850\mathrm{nm}$ and a full width at half maximum bandwidth of $\Delta \lambda = 165\mathrm{nm}$ was used for our customized SD-OCT system. The corresponding spectrometer used was a 2048-pixel spectrometer (Cobra-S 800, Wasatch Photonics). In addition, the other main components used were a $2\times 2$ 850-nm wideband fiber optic coupler (TW850R5A2, Thorlabs), a 2-axis scanning galvanometer scanner (GVSM002-EC/M, Thorlabs), two polarization

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20697

Optics EXPRESS

controllers (FPC030, Thorlabs), and two identical collimators (TC25APC-850, Thorlabs). A round continuously
variable metallic neutral density filter (NDC-100C-2M-B, Thorlabs) was inset into the reference arm to adjust the reference-arm optical power.

![](dt=2026-03-20/ht=08/2b054707f00b24a527205973ed0624bc0dd5439af78b80035e3be15ebe6ace19.jpg)

![](dt=2026-03-20/ht=08/9e0bcb6947ca12c8d221d68a42369202a5ceec392d212cd41d0328536ad63667.jpg)

Although deep learning is playing a dominant role in the field of image processing over the past few years, demonstrating its great power to address OCT image processing tasks of recognition and segmentation, speckle noise is still the key challenge in many OCT imaging performance-improved fields when using deep-learning-based methods, because it is difficult to obtain the proper training ground truths that are not affected by speckle. Therefore, it is quite difficult to directly use OCT images collected using the above method to perform OCT brightening and SNR enhancement.

Our previous works have demonstrated the deep-learning neural network trained from a large OCT image dataset customized by speckle-modulating OCT can significantly suppress OCT speckle noise [8]. Here we proposed a new strategy to achieve brightening OCT image and SNR enhancement based on deep-learning networks and a customized training dataset. As Fig.

1(a) shows, we collected the low-light OCT image dataset having low SNR and speckle noise as the input training data and the high-light OCT image data having high SNR, which was further processed using the speckle-removal algorithm from our previous work to become the final training SNR-enhanced data without speckle noise. In detail, we collected the high-light (hL) OCT images and low-light (lL) OCT images at the same imaging position, by using the same image-mapping gray value and configuring the imaging exposure time as $50.0~\mu \mathrm{s}$ and $12.5~\mu \mathrm{s}$ , respectively.

As shown in Fig. 1(a), we collected low-light $l\mathrm{L}$ OCT images and $h\mathrm{L}$ OCT images affected by speckle noise at the same imaging positions by adjusting the imaging exposure time. Then, the $h\mathrm{L}$ OCT images were preprocessed by the speckle-free-integrated algorithm to remove the speckle noise for obtaining the corresponding brighter SNR-enhanced OCT images ( $h$ -SNR

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20698

Optics EXPRESS

OCT images). Then, we used pixel-level paired $lL$ OCT images and $h$ -SNR OCT images as our training dataset.

# 2.2. Pipeline

Compared with conventional methods, the proposed deep-learning-based OCT image-brightening and denoising algorithm, based on a residual-dense-block U-Net generative adversarial network with channel-wise attention connections, can achieve brightening low-light OCT images and speckle-free SNR enhancement. For more details on our proposed SNR-Net structure, see Section 3 below. A key part of the proposed SNR-Net OCT was a deep-learning network trained on a large speckle-free SNR-enhanced brighter dataset, which simultaneously achieved OCT image brightening and speckle noise removal. It mainly contained three main steps to achieve the SNR-Net OCT based on deep learning: training dataset collection and preprocessing, network model training, and OCT SNR enhancement.

In Fig. 1(a), low-light $lL$ OCT images and high-light $hL$ OCT images were collected by using the same mapping gray value and adjusting the imaging exposure time, respectively. Then, the $hL$ OCT images were further preprocessed by the speckle-free-integrated algorithm [8,21] to remove the speckle noise to obtain the brighter $h$ -SNR OCT images. Then we used pixel-level pairing of $lL$ OCT images and $h$ -SNR OCT images as the dataset. Details can be found in Section 2.1.

In the network model training phase, the SNR-Net was constructed to achieve OCT image brightening and remove speckle noise, as shown in Fig. 1(b). The training data was input into the deep learning network respectively, and the forward propagation was started to obtain the predicted OCT image. By calculating the loss between the predicted image and the ground truth image, backward propagation was used to calculate the gradient and update the model parameters.

After the training phase, the trained network model was integrated with a conventional OCT setup, which was temporarily reconstructed for SNR-Net OCT. After acquiring original images, the trained neural network can directly output relatively speckle-free brighter OCT images with SNR enhanced, as Fig. 1(c) shows.

# 3. Proposed SNR-Net

# 3.1. U-Net with composite structure

U-Net is a convolutional neural network structure proposed by Ronneberger et al. [23] and has achieved great achievements in the fields of medical segmentation, image denoising, etc. In recent years, U-Net with composite structures has received extensive attention and has been shown to achieve good results on many vision tasks [24]. Here we used the composite-structured U-Net as our generator network structure, shown in Fig. 2(b).

First, the original low-light $lL$ OCT images were input into the encoder of the network to gradually extract features. Here the residual dense block structure (RDB) [25] was used to replace the original convolutional layer of U-Net. As shown in Fig. 2(a), the RDB contained densely connected layers, local feature fusion, and local residual learning. Meanwhile, the densely connected layer contained three $3 \times 3$ convolutional layers followed by a Leaky ReLU $(\alpha = 0.2)$ layer, where the number of filters used was 32.

In the encoding and decoding paths, a $1 \times 1$ convolutional layer was used to compress or expand the number of feature channels. In the encoding path, the number of filters was gradually increased, from 16 to 32, 64, 128, 256, and vice versa. In the encoding path, features extracted by RDB were down-sampled using a $2 \times 2$ max-pooling with a stride of 2, and the down-sampling layer is performed four times in total. Meanwhile, $2 \times 2$ up-sampling layers and $1 \times 1$ convolutions were used for up-sampling in the decoding path.

Then, to enable the network to efficiently capture the global information in each feature, the channel-wise attention (CA) [26] blocks were further used to connect the encoding layers

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20699

Optics EXPRESS

![](dt=2026-03-20/ht=08/93736af4c9034d8711cf62e56148d946b1ae32e948c515361351e91cf75af6e5.jpg)

![](dt=2026-03-20/ht=08/fbbfae718e465f2cdb3c37b51a513c7f0c07f522f700435e085a4c770937c076.jpg)

![](dt=2026-03-20/ht=08/0590986cfe8a108f053bc44968d04961ecd8d4ce34d82dfbe3dc0e080c9fa6c9.jpg)

with the decoding layers. As shown in Fig. 2(b), CA blocks squeezed spatial information into channels through global average pooling and then calculated channel attention through multi-layer perceptual, which was used to allocate the feature map weight. The last layer was a $1 \times 1$ convolutional layer that reconstructs the decoded image of the same size as the input image. Meanwhile, to make the neural network focus on learning the difference between input and output, residual connections were applied between input and output. In this way, the network was able to learn how to brighten low-light OCT images and obtain clear SNR-enhanced images.

# 3.2. Generative adversarial network structure

Generative adversarial networks (GANs) [27,28] have strong learning abilities and can effectively recover detailed information in images, which is widely used for ophthalmic OCT image domains. Here we built a GAN to guard the brightening
and denoising performance of SNR-Net OCT, as Fig. 3 shows.

During the training, the discriminator distinguished between ground-truth images and images predicted by the generator. Oppositely, the generator tried to generate high-quality brightened and denoised OCT images from original images so that the discriminator cannot recognize them. For the discriminator network structure, the convolutional layers had $3 \times 3$ kernels, followed by Leaky ReLU ( $\alpha = 0.2$ ) layers, and except for the first convolutional layer with stride 2, the rest of the layers had stride 1, as shown in Fig. 3. The classification probabilities were output using two dense layers and Leaky ReLU and sigmoid activation functions. Note that, we employed the same composite-structured U-Net in Fig. 2 as the generator of GAN.

# 3.3. Objective function

In order to obtain high-performance SNR-Net OCT, here a hybrid loss function was used in our deep learning network, and the training objective function is shown in Eq. (1):

$$
L o s s = \alpha_ {1} L _ {A} + \alpha_ {2} L _ {2} + L _ {V G G}, \tag {1}
$$

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20700

Optics EXPRESS

![](dt=2026-03-20/ht=08/98a3cb7639ec774464f7aefdca132049c0395fed160172adfcac95e57003e755.jpg)

where $\alpha_{1}$ and $\alpha_{2}$ are weighting coefficients used to balance the contributions of different loss functions. Among them, the $L_{\mathrm{A}}$ represents the adversarial loss, which is defined as Eq. (2):

$$
L _ {A} = - \frac {1}{n} \sum_ {n = 1} ^ {n} \log D [ G (x _ {i}) ], \tag {2}
$$

where $D[G(x_i)]$ denotes the probability predicted by the discriminator that the image is an SNR-enhanced image.

The pixel-level loss $L_{2}$ is a commonly used model optimization loss function in image processing tasks as shown in Eq. (3):

$$
L _ {2} = \frac {1}{n} \frac {1}{W \cdot H} \sum_ {i = 1} ^ {n} \| G (x _ {i}) - y _ {i} \| _ {2} ^ {2}, \tag {3}
$$

where $W$ and $H$ represent the width and height of the image, respectively, $x_{i}$ and $y_{i}$ represent the input original image and the ground truth, respectively. However, $L_{2}$ loss function is not conducive to the recovery of high-frequency details (such as textures), which may cause the image to be too smooth [29]. Therefore, perceptual loss ( $L_{\mathrm{VGG}}$ ) was used to recover detailed texture information. Here, the VGG-19 network [30] was used to extract the feature images of predicted images and real images. $L_{\mathrm{VGG}}$ is the Euclidean distance between two feature images, defined as Eq. (4):

$$
L _ {V G G} = \frac {1}{n} \frac {1}{w h d} \sum_ {i = 1} ^ {n} \| V G G _ {1 9} (G (x _ {i})) - V G G _ {1 9} (y _ {i}) \| _ {2} ^ {2}, \tag {4}
$$

where $VGG_{19}$ denotes the VGG-19 network pre-trained on ImageNet. Here, we extracted high-level features using the 4th convolution before the 5th max-pooling layer. Moreover, $w$ , $h$ , and $d$ are the width, height, and depth of the feature image, respectively. Furthermore, we replicated the 1-channel OCT images to the second and third channels in order to match the pre-trained VGG-19 network.

Discriminator loss $(L_{\mathrm{D}})$ defined as Eq. (5) was used to optimize the discriminator:

$$
L _ {D} = \frac {1}{n} \sum_ {n = 1} ^ {n} \left\{\log D \left(y _ {i}\right) - \log \left[ 1 - D \left(G \left(x _ {i}\right)\right) \right] \right\}. \tag {5}
$$

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20701

Optics EXPRESS

# 3.4. Evaluation metrics

Here we used structural similarity index (SSIM), contrast-to-noise ratio (CNR) and image signal-to-noise ratio as evaluation metrics, in order to objectively quantify the detection SNR enhancement and denoise performance of our proposed approach. Image SSIM was a metric to measure the perceptual visual difference between the predicted image and the ground truth, as Eq. (6) expressed.

$$
\operatorname {S S I M} (\mathrm {y}, \hat {\mathrm {y}}) = \frac {\left(2 \mu_ {\mathrm {y}} \mu_ {\hat {\mathrm {y}}} + C _ {1}\right) \cdot \left(\sigma_ {\mathrm {y} \hat {\mathrm {y}}} + C _ {2}\right)}{\left(\mu_ {\mathrm {y}} ^ {2} + \mu_ {\hat {\mathrm {y}}} ^ {2} + C _ {1}\right) \cdot \left(\sigma_ {\mathrm {y}} ^ {2} + \sigma_ {\hat {\mathrm {y}}} ^ {2} + C _ {2}\right)}, \tag {6}
$$

where $\mu_{\mathrm{y}}, \mu_{\mathrm{oy}}, \sigma_{\mathrm{y}}, \sigma_{\mathrm{oy}}, \sigma_{\mathrm{yoy}}$ represent the local mean, standard deviation, and cross-covariance for predicted image $\hat{y}$ and ground truth $y$ , respectively. To avoid situations where the denominator is zero in (6), $\mathbf{C}_1$ and $\mathbf{C}_2$ as the regularization coefficients are added to the denominator.

As an evaluation index of image quality, CNR measures the contrast between the selected region of interest (ROI) and the background ROI. CNR can be calculated according to Eq. (7), where $\mu_r$ and $\mu_b$ represent the average value of the ROI and background ROI respectively, $\sigma_{\mathrm{r}}$ and $\sigma_b$ are the standard deviation (SD) of the ROI and background ROI respectively.

$$
C N R = 1 0 \times \log_ {1 0} \left(\frac {\left| \mu_ {r} - \mu_ {b} \right|}{\sqrt {\sigma_ {r} ^ {2} + \sigma_ {b} ^ {2}}}\right). \tag {7}
$$

Meanwhile, we chose the image metric signal-to-noise ratio as a quantitative indicator, which can well measure the signal intensity at the ROI. where the SNR is shown in Eq. (8), where $\mathrm{I_r}$ represents the value of the ROI; $m$ and $n$ are the height and width, respectively; $\sigma_{b}$ represents SD of the background ROI.

$$
S N R = 1 0 \times \log_ {1 0} \left(\frac {\sum_ {i = 1} ^ {m} \sum_ {j = 1} ^ {n} I _ {r} ^ {2} (i , j)}{\sigma_ {b} ^ {2}}\right). \tag {8}
$$

# 4. Experiment and results

# 4.1. Dataset and training details

Here we collected a customized speckle-free-integrated and image-brightened OCT dataset to train our SNR-Net model. The dataset was first obtained by adjusting the imaging exposure times of conventional SD-OCT setup, and the same image-mapping gray value was used to obtain the corresponding $lL$ and $hL$ OCT images. Then $hL$ OCT images was further pre-processed through the speckle-free-integrated algorithm to acquire the training ground-truth speckle-free brighter OCT images.

As previously mentioned, speckle noise removal in non-ophthalmological OCT images has always been a difficult challenge. Speckle-modulating OCT (SM-OCT) can obtain speckle-free OCT images, but it seriously reduces the imaging sensitivity and time resolution. The speckle-free-integrated algorithm proposed by our previous work can achieve the same effect as SM-OCT, and ensure the imaging sensitivity and time resolution.

Figure 4 shows the original OCT images, OCT images obtained from our speckle-free-integrated algorithm, and SM-OCT images, respectively, which demonstrated that our speckle-free-integrated algorithm can effectively remove the speckle noise and recover the detailed texture almost consistent with ground truth. In detail, in Fig. 4(b) and 4(e), we can see that the speckle noise in the tape and meat has been effectively removed and more microstructures lost in Fig. 4(a) and 4(d) have been resolved.

To obtain mass speckle-free-integrated image-brightened OCT images for training a deep learning network, we imaged Scotch tape and pork meat (foodstuff). For each sample, we

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20702

Optics EXPRESS

![](dt=2026-03-20/ht=08/137ad8d48b3e3295d9a253981a04814d9525be0621835215ca8a0e45bbf224c3.jpg)

![](dt=2026-03-20/ht=08/bc96086a957a5ff69e264a498f03c15a19ba1145eb1ed06e333f1988ce406834.jpg)

![](image)
20/ht=08//cbe6b8ad33ab766bedf74920801b2c0352c94ea8ac2a8b8dedfef595ca565ed8.jpg)

![](dt=2026-03-20/ht=08/b561ce95a4025710763a0a05f0390c9a005d73a1f5861bf8f6cf4ec4af2be653.jpg)

![](dt=2026-03-20/ht=08/906ace124215ee1c4b7b8831924532f13bada233a1707e94d2042759a01b63f3.jpg)

![](dt=2026-03-20/ht=08/c97a4eb3b32b37532b554b7eba707f9bbb8c1196904fc8af50881617a219ddab.jpg)

![](dt=2026-03-20/ht=08/b8873d6026fa181115e8c6a1902c97c8e08f07e4f17e3b5dce9be0f15db2e626.jpg)

![](dt=2026-03-20/ht=08/ea209f667fe607b3b000103907f5261295ef8a969e5dedb817e6b571238514f1.jpg)

![](dt=2026-03-20/ht=08/6153c4f56e9eb673d585acb57f3f0e6c4577868ccb1999715057915a91895069.jpg)

![](dt=2026-03-20/ht=08/3afc4b80955ba623b966833952c10988404437248fc5274072cbf08fcdfca889.jpg)

![](dt=2026-03-20/ht=08/18801b46f6981b03fd73fce0f55cc7a570c5b21a83f46b87714fa400420db67e.jpg)

![](dt=2026-03-20/ht=08/7bfbaac3b10d0464b00eeca47bafafb44234cb34e44bb90f76e03762620549f0.jpg)

![](dt=2026-03-20/ht=08/8aa1d06e1ecfbc5c3f862c61388ed74e0f4dfc245d718a855ffee19598772572.jpg)

![](dt=2026-03-20/ht=08/56732c476b58543d72adb28a998941d5b8031636996d35a430f49c90802e7685.jpg)

![](dt=2026-03-20/ht=08/70ccc9558db1c89781b1021a05cb1e9143c0c66b440f6b4deba6d80a98d4b887.jpg)

acquired images at different locations and preprocessed them with the speckle-free-integrated algorithm. We collected 800 frames consecutively as a sub-dataset and the sub-datasets were of 20 Scotch tape and 20 pork meat, respectively. After selection and preprocessing, a total of 16,000 pairs of B-scan frames were collected, and we randomly selected 5,000 pairs of B-scan frames as validation data. Then, we performed augmentation operations on other 11,000 pairs of images, such as image rotation, flipping, and random cropping, to increase the dataset size. After augmentation, a total of 880,000 pairs of OCT images were used for training.

The training network was implemented using Python (v3.8.12) based on a Tensorflow (v2.5.0) backend using a customized speckle-free-integrated image-brightened OCT dataset. To enable parallel computing and speed up the training process, training was performed on an NVIDIA Geforce RTX 3080 GPU. We trained the model using 200,000 iterations with a batch size of 1 for training. In the training phase, we trained the generator using the hybrid loss function in (1), where $\alpha_{1} = 5\times 10^{-3}$ and $\alpha_{2} = 1\times 10^{-2}$ .

For optimization, we chose the adaptive learning rate optimization algorithm (Adam) as the training optimizer, with learning rate $\alpha = 1\times 10^{-4}$ , decay parameters $\beta_{1} = 0.9$ , $\beta_{2} = 0.9$ . The parameters of the generator and discriminator were updated until the model converged.

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20703

Optics EXPRESS

# 4.2. Performance of the proposed SNR-Net

To demonstrate the ability of the proposed U-Net GAN to brighten and denoise low-light OCT images, we tested the trained U-Net GAN on OCT images of Scotch tape and pork meat, which were not used to train the deep learning network. Figure 5 and Fig. 6 contain low-light $lL$ OCT images of Scotch tape and pork meat, $hL$ OCT images, and $h$ -SNR OCT images preprocessed by the speckle-free-integrated algorithm acting as the ground truth, and the predicted results of the proposed SNR-Net.

![](dt=2026-03-20/ht=08/38f8b92c9353d5c9bc03c64999613559201c3aca7b819d4a8ae29cee232a2362.jpg)

![](dt=2026-03-20/ht=08/fe76ca484b9e57f89603fa3d1a09ee64528945b929fae26a885c77433efbc91d.jpg)

![](dt=2026-03-20/ht=08/174330024dd7cc5735a5cd8bd3d32b8f2c9362fb65ce56d00445120f6a520e1c.jpg)

![](dt=2026-03-20/ht=08/fcbea5a2650a313d38857449c93cc115b1b0870efe9d65bf9bbae0b0e5fe7543.jpg)

![](dt=2026-03-20/ht=08/01ed54bbb5db224dab8c16f549a76847d4d37168c9a578e2f768b8f7cf617c9d.jpg)

![](dt=2026-03-20/ht=08/611b218287018fe3a802f2aa6082595e4197cf63393722d321ccb0155e6298f2.jpg)

![](dt=2026-03-20/ht=08/b87434b3e390556c18a4927a64cdf91bccab8bf852273778b2fead67090ccb77.jpg)

![](dt=2026-03-20/ht=08/9711952a12f81957ddc62b1b3b6b182a615161d92692adb143e3df781d87ee18.jpg)

In Fig. 5(a), (b) and Fig. 6(a), (b), we can see that image brightness has been obviously improved when using larger imaging exposure time, but due to the influence of noise including the speckle, it still had a low image quality and the microstructures in the image were obscured. As shown in Fig. 5(c) and Fig. 6(c), our previous speckle-free algorithm can effectively remove the speckle noise and electronic noise, and more detailed microstructures can be observed well, such as the blemish gap in Scotch tape and the membrane gap of pork meat.

Meanwhile, the predicted results of our proposed SNR-Net have almost the same performance in brightening OCT images and show a strong ability to remove noise compared with the ground truth images, as shown in Fig. 5(c) and Fig. 6(c). Comparing the zoomed-in regions of interest (ROI) in Fig. 5(a) and 5(c), and Fig. 6(a) and 6(c), deep microstructures that cannot be observed in low-light OCT images can be observed well, such as the stain gap in the scotch tape and membrane gap in the pork meat, demonstrating the ability of the proposed SNR-Net to brighten and denoise low-light OCT images.

Table 1 shows the quantitative performance of the proposed SNR-Net, including SSIM, CNR and SNR metrics. Specifically, the original low-light OCT images in Fig. 5 and Fig. 6 have the lowest SSIM, CNR and SNR values due to speckle noise. Compared to the original low-light OCT images, the SSIM metrics of the proposed SNR-Net were improved and the corresponding CNR and SNR metrics are close to the ground truth images. This means that the image quality of brightened and denoised images is close to ground truth, proving the strong performance of the proposed SNR-Net on low-light OCT image brightening and denoising tasks.

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20704

Optics EXPRESS

![](dt=2026-03-20/ht=08/25f936f3ffd821c51b2cf284bccf8268bcc1f073eb0c15bef28cce5d4c6d7b32.jpg)

![](dt=2026-03-20/ht=08/dce8baea3722619d28f98c5f833308bf7933e45cea31af32d3821e787b8526de.jpg)

![](image)
0/result=success/type=image/dt=2026-03-20/ht=08//4d77073655b78858b0439945b158bcd31b6abb11fa627ee2602c1cef036d1240.jpg)

![](dt=2026-03-20/ht=08/b8444c34ab454d16c173bdeae64518803900a431784f001861f759c4bdcd3165.jpg)

![](dt=2026-03-20/ht=08/72819663522cccb4592b0d6eccabeea988a80ff6afa0abcb599576b3f3678300.jpg)

![](dt=2026-03-20/ht=08/5c7ef13449a75a77cc19f4c5e9d9a4440b39e9efbdc044e4f19008756fa8a68c.jpg)

![](dt=2026-03-20/ht=08/4fe3f2e2ffdc651655cf4000b5d75b7807d1a27dde05e9ecc60720f5bc672571.jpg)

![](dt=2026-03-20/ht=08/df306166cca17df8608b3d66b7ecf680e7fad913bfb2e460f0b63d8b7ab45822.jpg)

Table 1. SSIM, CNR, SNR of Scotch tape and pork meat images shown in Fig. 5 and Fig. 6

![](dt=2026-03-20/ht=08/d3db33028086aac7fcc9913f268f076e0d7e9232293e208109835687eeb46b16.jpg)

<table><tr><td></td><td colspan="3">Scotch tape</td><td colspan="3">Pork meat</td></tr><tr><td>Evaluation metrics</td><td>SSIM</td><td>CNR</td><td>SNR</td><td>SSIM</td><td>CNR</td><td>SNR</td></tr><tr><td>lL OCT image</td><td>0.1079</td><td>0.3605</td><td>16.3371</td><td>0.0859</td><td>0.0607</td><td>10.8618</td></tr><tr><td>hL OCT image</td><td>0.2122</td><td>3.4778</td><td>19.6752</td><td>0.1779</td><td>1.5746</td><td>16.7919</td></tr><tr><td>SNR-Net OCT image</td><td>0.7184</td><td>6.2961</td><td>35.9173</td><td>0.7091</td><td>5.8441</td><td>33.8326</td></tr><tr><td>Ground truth</td><td>1.0000</td><td>6.4403</td><td>36.1036</td><td>1.0000</td><td>6.0261</td><td>34.1628</td></tr></table>

# 4.3. Results of SNR-Net OCT

Figure 7 shows a comparison between low-light OCT images and corresponding SNR-Net OCT images. Figure 7(a) and Fig. 7(d) are the low-light OCT images of Scotch tape and pork meat, respectively. Figure 7(b) and Fig. 7(e) are the corresponding images of the proposed SNR-Net OCT. Figure 7(c) and Fig. 7(f) are the corresponding ground truths which were acquired to test the performance of SNR-Net and never used to train the SNR-Net model. Figure 8 shows the normalized intensity distribution charts of the corresponding zoomed-in ROIs.

As shown in Fig. 7, deep microstructures that cannot be revealed in the low-light OCT images, were effectively revealed in brightened SNR-Net OCT images with removing speckle. For example, fringe details of the tissue were seriously affected by noise in Fig. 7(a) and Fig. 7(d), while our SNR-Net OCT removes speckle noise and electronic noise and brightens images, preserving the fringe details well and with similar performance to ground truth. In details, as shown in Fig.

8, the SNR of Scotch tape OCT image was improved from $8.15\mathrm{dB}$ to $32.66\mathrm{dB}$ , and SNR of pork OCT image was improved from $7.14\mathrm{dB}$ to $30.68\mathrm{dB}$ , which indicates that SNR-Net OCT can reveal microstructure well on the selected ROI. Meanwhile, the normalized curves of SNR-Net OCT image and ground truth exhibit a remarkable level of agreement, which serves as a testament to the exceptional reconstruction accuracy of SNR-Net OCT.

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20705

Optics EXPRESS

![](dt=2026-03-20/ht=08/769ca23935dfa7b18c1e0ede4f8caaad6f0297bb1bc06ba224c6a7aa096c3748.jpg)

![](dt=2026-03-20/ht=08/bc46fc713713d605a23166cf00d77f3e761965656fae26237607e7c82da85fe1.jpg)

![](dt=2026-03-20/ht=08/7433ff07717c85cecbd8dd239fd2c393f617c8eca1da9e57884a8bd7f4832db2.jpg)

![](dt=2026-03-20/ht=08/8973bed095825f85cfdeac665c82449992347dc28c5be5b4cb1d96252ebe73a2.jpg)

![](dt=2026-03-20/ht=08/bb310de79a9cf5b3944170f6585fabac97129bf0acfb6c23dcd2b60736461fc1.jpg)

![](dt=2026-03-20/ht=08/077e53d9b17d2454be83d04fc03f7da246bba6638e1d0f68b682db58c00cb733.jpg)

![](dt=2026-03-20/ht=08/750f671659a47a374a5d3c52b242b4add5c493924139eec90fca3b7b8e2735cc.jpg)

![](dt=2026-03-20/ht=08/461a5850a01197b05785e163d4b9b1896a0f8b1d7ae24db085bb96d69e95e225.jpg)

![](dt=2026-03-20/ht=08/4fd0a862f435300aa692ab2c8a170c3b7704c9e60b10fd154a91f68c250c37ae.jpg)

![](dt=2026-03-20/ht=08/0e7b28652797ef3261add796c074b20c558b938b89aa5b3d17b8063a92466c29.jpg)

![](dt=2026-03-20/ht=08/9ff048631b6502b622d0e03e771de3f864d9e4a0ee5a10a70dc5c2ac3fcf34ed.jpg)

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20706

Optics EXPRESS

To demonstrate the generalization ability of our SNR-Net OCT, we further tested the low-light OCT images of foodstuff duck meat and fish meat which were not used to train the model. Figure 9(a) and Fig. 9(d) are the low-light OCT images of duck meat and fish meat, respectively. Figure 9(b) and Fig. 9(e) are the corresponding SNR-Net OCT images. Figure 9(c) and Fig. 9(f) are the corresponding ground truths. Figure 10(a) shows the normalized intensity distributions of the zoomed-in ROIs in Fig. 9(a), (b) and (c).

Figure 10(b) shows the normalized intensity distribution of the zoomed-in ROIs in Fig. 9(d), (e) and (f). From the Fig. 9(a) and brightened Fig. 9(b), it can be observed that the blemish gap obscured by the scattered noise is effectively revealed. Meanwhile, the deep texture details can be well observed in the green-color-marked region and orange-color-marked region in Fig. 9(b) and brightened Fig. 9(e). In terms of evaluation metrics, as shown in Fig.

10, the SNR of the indicated position in the duck images increases from $11.02\mathrm{dB}$ to $31.93\mathrm{dB}$ , and the SNR of the corresponding position in the fish images increases from $10.57\mathrm{dB}$ to $31.53\mathrm{dB}$ . Moreover, the normalized curves of SNR-Net OCT images are in good agreement with the ground truth.

![](dt=2026-03-20/ht=08/4c711ff50652fec4df49e7463fa9af42d24e59a8afbc383207111139f6b85629.jpg)

![](dt=2026-03-20/ht=08/112600f9458e734189f1d44649fcb515ecf9a02c800381d24b8db2ecb46d3c65.jpg)

![](dt=2026-03-20/ht=08/a21e54b435c102ddad90f49f96481ba89d11700ada64921b4acf0a068adba4a6.jpg)

![](dt=2026-03-20/ht=08/569c9177856b2b3d3410c3bf95d786ea672d07608bf27c838b8851d4419ff7d5.jpg)

![](dt=2026-03-20/ht=08/b23386b7b65acc746491ecd469537b2382ff2f49081f6a2729f733a576ddbfd1.jpg)

![](dt=2026-03-20/ht=08/ba1cdd82893e5fd2fc0c7cf600f79748bb674d5263f7a1b5bc266feaf3916b9e.jpg)

![](image)
produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-03-20/ht=08//2343466054eb3fba738648033919246626960cf8d6cf0248c3e022dd52058a65.jpg)

![](dt=2026-03-20/ht=08/297aec69fc4079e747e6da1f2ca7959c4cc5c95ca06a7637fa9841739902ccab.jpg)

![](dt=2026-03-20/ht=08/11d723f5d7a68704f44915692882587f4ae252b77714416024bbe1e7333e1aac.jpg)

![](dt=2026-03-20/ht=08/e1a10bb1e5817ea504d6816ac07dc36a1c0133c3d4f0d6e677eb98b8adb06bb4.jpg)

![](dt=2026-03-20/ht=08/44a0e2f6e7714486ebb172d2d2efe56444714e4f2a18e15b674774b8fb5bab28.jpg)

![](dt=2026-03-20/ht=08/537bee228cfd0a97db678920cf8fa5df302322ad4925706cd8fa5cc039bd1c2b.jpg)

Further, we also tested and compared the performance of SNR-Net OCT when dealing with different conditions of low-light OCT images, to demonstrate that our SNR-Net OCT also has favorable performances, as shown in Fig. 11. Figure 11(a) and Fig. 11(d) were brighter OCT images collected by a conventional SD-OCT setup using larger imaging exposure time. Figure 11(b) and Fig. 11(e) are the corresponding SNR-Net OCT images. Figure 11(c) and (f) are ground-truth OCT images obtained from Fig. 11(a) and Fig. 11(d) by being processed using our speckle-free algorithm, respectively. Figure 12(a) and Fig. 12(b) show the normalized intensity

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20707

Optics EXPRESS

![](dt=2026-03-20/ht=08/693700e7dbe38cc1f16dfdccbaa5a024d955e46cc5c624529de08fde01628e94.jpg)

![](dt=2026-03-20/ht=08/865b3ab5bb9d5d1d430da0115aae0b66835e9a230ee3fe6834c16ae41dbee3bc.jpg)

distributions of the corresponding ROIs. From Fig. 11(b) and Fig. 11(e), it can be observed that SNR-Net OCT reveals the deep fringe details well with higher SNR and speckle removal, such as the blemish gap in Scotch tape and the membrane gap of pork meat can be seen clearly. In details, it can be observed that SNR at the indicated fringes of Scotch tape increases from 14.31 dB to 34.18 dB, and that of pork meat increases from 8.78 dB to 30.54 dB.

# 4.4. Network comparisons

We further performed the ablation experiments to prove the capability of our proposed network structures in the SNR-Net OCT. Here we trained U-Net GAN with Resblocks (baseline) [27,31], U-Net GAN with RDB, and proposed U-Net GAN combined with RDB and CA blocks, respectively, using the same training dataset and strategy mentioned in Section 4.1, as shown in Fig. 13. We further randomly selected 50 low-light OCT images from the validation dataset (Section 4.1) for testing, and compared the averaged SSIM, CNR and SNR metrics of different network.

As Table 2 shows, the single RDB provides favorable gains of $0.2299\mathrm{dB}$ for CNR and $1.2475\mathrm{dB}$ for SNR over the baseline, and our $\mathrm{RDB} + \mathrm{CA}$ block connections further produces $0.1727\mathrm{dB}$ CNR gain and $0.7920\mathrm{dB}$ SNR gain over the single RDB. In our proposed network structure, we combined the RDB to capture the local information well and introduced CA mechanism to maintain the global information in each feature, which led to obvious CNR gain of $0.4026\mathrm{dB}$ , SNR gain of $2.1395\mathrm{dB}$ , and higher SSIM over the baseline.

Table 2. Ablation experiment results for proposed network structure

![](dt=2026-03-20/ht=08/4d63bf3d18619b8ea69d58123ebadb0396fd6e40b38cf8ff7e7bd8f94cf94558.jpg)

<table><tr><td>Network</td><td>SSIM</td><td>CNR</td><td>SNR</td></tr><tr><td>Baseline</td><td>0.6957</td><td>5.7130</td><td>32.5851</td></tr><tr><td>RDB</td><td>0.7039</td><td>5.9429</td><td>33.8326</td></tr><tr><td>RDB + CA</td><td>0.7104</td><td>6.1156</td><td>34.7246</td></tr></table>

# 4.5. SNR-Net OCT in clinical applications

In addition to OCT images of tape and pork meat samples, we performed OCT image brightening and denoising on our OCT imaging dataset of human placenta villi and human skin obtained used the same conventional OCT setup mentioned above, which were not used in training the model. Here human placenta villi sample acquisition was approved by the Ethics Committee of the University of Electronic Science and Technology of China (ID: 1061420211102003).

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20708

Optics EXPRESS

![](dt=2026-03-20/ht=08/b907ce2ffa2ecbc6d067c21d881ea1938d3c2296b19827a8ffe6b0ddf784ba10.jpg)

![](dt=2026-03-20/ht=08/b77b6dd13ab46750c429a791f0d697badd7e2e1e7e9a8e93532654869cc91a49.jpg)

![](dt=2026-03-20/ht=08/5dd59acc605d251fdcec9fb6fa3dfe667d18852ea5be58cc774179ed75a3d0ec.jpg)

![](dt=2026-03-20/ht=08/c8458c034f180237768ffb88e9f9f73e74fbef87d6a19c83a15c5c47eef3b685.jpg)

![](dt=2026-03-20/ht=08/61b6cb7defee43e7d52c31ec8b8f55a537a793c5eba0d20d5e8a802446b74d5c.jpg)

![](dt=2026-03-20/ht=08/bd21083b07a721127c03cfa00af07330746c0bee06788c2388b40c044d3afbfd.jpg)

![](dt=2026-03-20/ht=08/669bbc09298947f3aef711dcd749ab1f2762d1b8230e4652fd3de0a6f6783b0d.jpg)

![](dt=2026-03-20/ht=08/f622bb0c5941cb5e0659960505cab39565edcf6d023a40b7d3023049e07eb55a.jpg)

![](dt=2026-03-20/ht=08/44bc8c9521d8e047eac891cb4c318d0bee30751e12c911b1ef5407a73caefdd6.jpg)

![](dt=2026-03-20/ht=08/7e83cc69e1af164d4ec4a6bf2b2793ddc00fe0f2130ee0f3e2898b539d06abbe.jpg)

![](dt=2026-03-20/ht=08/ef5677be8f3498a1f0935f45ff46fc9f994391869bdb32d46bf19623b78cddba.jpg)

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20709

Optics EXPRESS

![](dt=2026-03-20/ht=08/98bcf3f6f1d3853bb4375ca5fa7b1e6f34980fca05b4ab88ff66ab28646054ce.jpg)

![](dt=2026-03-20/ht=08/c7594a4e5e4a4e52a8d18c2dd7fb2b50794c16b2b35dc79de1b36f75512b940c.jpg)

![](dt=2026-03-20/ht=08/2be3116835a591c099edc4ea80007345ae195aac8b5a9955b7f7a62bd9ab9fd9.jpg)

![](dt=2026-03-20/ht=08/976372293c86a5095b6405f9220719e22139a09a8f370dabef9bd8426ac8792d.jpg)

![](dt=2026-03-20/ht=08/0d24e03e257824a79f34de353292ebbb1e053b4994a10a15c69f3544f8e56ab8.jpg)

![](dt=2026-03-20/ht=08/e6c6693dc29a4db79bdb777cfcab8df67b7a24972b34504e986049769a2e8ea8.jpg)

![](image)
xt/v0/result=success/type=image/dt=2026-03-20/ht=08//9013a6930bee4a92bdfd55cd6b3468ce3aafbfe78a1e1703c580fac36ea8e7c1.jpg)

![](dt=2026-03-20/ht=08/1c257f62804683a91834abbe8f91a7db0b4c38e2e5b72f41a9ba482a36e06777.jpg)

Figure 14(a) and Fig. 14(b) are the low-light OCT images of human skin and corresponding SNR-Net OCT images. For the brightened Fig. 14(b), it can be observed that the noticeable speckle noise is reduced and the capillary vessel structure in the dermis becomes clearer, as shown in the blue-color-marked region. Meanwhile, SNR of OCT image in deep tissue region is also effectively enhanced, which facilitates the observation of microstructures after brightening. For example, the capillary microstructure, which is difficult to be distinguished in the original low-light OCT image, can be effectively observed after image brightening and denoising, as shown in the orange-color-marked region.

The proposed SNR-Net OCT can also be used to brighten and denoise conventional OCT images, as Fig. 15 shows. Figure 15 shows conventional OCT and SNR-Net OCT images of human placental villi. Figure 15(a) is the OCT image of human placental villi acquired by the conventional OCT setup, and brightened Fig. 15(b) is the corresponding SNR-Net OCT result. As shown in Fig. 15, deep-region microstructures can be observed better with the image brightening, SNR enhancement and speckle removal of the proposed SNR-Net OCT, and the human placental villus membrane can be observed well, as shown in the blue-color-marked region, which demonstrated the valuable ability of SNR-Net OCT.

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20710

Optics EXPRESS

![](dt=2026-03-20/ht=08/76bfda3ef8e08ad96bc0bb30e4b4a6ebc4b7ed090cc3f42d0bf341fb69bbac91.jpg)

![](dt=2026-03-20/ht=08/f482b686660c3b01d14cb32f6ad8dac7825ccf03c2b3148cccf71dce362231d7.jpg)

![](dt=2026-03-20/ht=08/ecf02812de920a309af056cd201a4a4ad7d75fcd6e4d5ef53bb474edd3ed167d.jpg)

![](dt=2026-03-20/ht=08/60a827247de1076fb47958e5b510e152fa36ccbd96cfdaf8b01d1c27a5d6ad7c.jpg)

![](dt=2026-03-20/ht=08/5b25b850ffd46be91f7e2f9f5bd529a7f3ae3d1889a8f0bdcabe12ccb60cf1de.jpg)

![](dt=2026-03-20/ht=08/6638ef2fabf4d16c5f3be68701dd8fe7f68ea71bf4974fed72f73d252cd027ee.jpg)

![](dt=2026-03-20/ht=08/c1db5e59e744a6e1ec7f2ad5cbbb3334d51ed6e000d86cfbcb55519539b66b59.jpg)

![](dt=2026-03-20/ht=08/c24e26007b4ea8d55b524e669b06b00476795f7e5739638af8ee84956572853d.jpg)

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20711

Optics EXPRESS

# 5. Discussion and conclusion

In this study, we analyzed the shortcomings of previous work of OCT SNR enhancement, especially for low-light OCT images, including hardware and software-based methods. Here, combining the characteristics of deep learning and our previous work in speckle noise removal, we further proposed a deep-learning-based brightening and denoising technique for low-light OCT images, termed SNR-Net OCT, to acquire higher SNR OCT images combined with speckle removal and image brightening.

A customized large speckle-free-integrated SNR-enhanced training dataset was trained by deeply integrating the conventional OCT setup with the proposed SNR-Net. This method can directly generate speckle-free SNR-enhanced brighter images from original low-light images and preserve the texture information of the images well.

Further, to prove the denoising plays a dominant role in improving the image quality in the above experiments, comparative experiments were conducted. As shown in Fig. 16, we compared the proposed SNR-Net with the results of low light images obtained after being successively processed by the speckle-free algorithm and the hist-equalization method. As shown in Fig. 16(b) and (e), the application of the speckle-free algorithm and subsequent hist-equalization method to low light images leads to the generation of artifacts and over-enhancement in low-light regions, which can result in loss of important details. In contrast, our proposed SNR-Net can effectively suppress speckle noise and enhance SNR of low-light OCT images while retaining detailed information.

![](dt=2026-03-20/ht=08/fe15a5c0e2273c5c91d534468daaeff3a26b0d366ad47bcfab5790548a30161a.jpg)

![](dt=2026-03-20/ht=08/d032538fcb5024c9e0bc3c08f9503c99434186e3bfee26d4d92b220ca4246b99.jpg)

![](dt=2026-03-20/ht=08/f347f9a19aaeee883f00509751804a21bb4d15159ec9a32d4b66797fbf6d42c7.jpg)

![](dt=2026-03-20/ht=08/dc468d5718f01bf818e9118ed6de5ca3a1318dea75d0dd38ca2a6775442174c4.jpg)

![](dt=2026-03-20/ht=08/730c987b84fd0d66ebd3ef9523920ddd476683d2dc741fefcca2590534050e98.jpg)

![](dt=2026-03-20/ht=08/2dc335e5276525f354e0975a1e7f46ed053ad5ab054914cbb62c3e4a1dba31d8.jpg)

Although our proposed SNR-Net OCT has achieved good results in low-light OCT image quality improvement, its performance still can be improved. By now, only Scotch tape and pork meat datasets were used during training, which limits the robustness of the network. Also, different OCT setups having variations in light central wavelength, bandwidth, and scanning lens, have substantial impacts on imaging resolution and speckle characteristics. As a result, to get better results, collecting data from different OCT systems when using different light central wavelength, bandwidth, or scanning lens and retraining are needed. Meanwhile, Our SNR-Net

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20712

Optics EXPRESS

OCT may exhibit limitations in certain scenarios. Specifically, when the input image is already bright, the grayscale value of which is close to 255, the SNR-Net OCT may result in saturation, leading to an overly smooth OCT image. In the future, a wider range of sample datasets with varying OCT setups can be added to improve the robustness of SNR-Net OCT. Additionally, increasing the mapping of high-light OCT images to their corresponding ground truth can further enhance the generalizability and increase its potential for practical applications.

In summary, here we applied the deep learning method to the task of low-light OCT image brightening and denoising by fixing the OCT speckle challenge for low-light OCT imaging. We proposed a deep-learning-based OCT combined with brightening and denoising features, and demonstrated its capabilities of image brightening, speckle noise removal, and microstructure preservation, which can effectively reveal deep tissue information and have good prospects in clinical applications. Moreover, compared to the hardware-based techniques, the proposed SNR-Net OCT can be of lower cost and better performance.

Funding. National Natural Science Foundation of China (61905036); China Postdoctoral Science Foundation (2019M663465, 2021T140090); Fundamental Research Funds for the Central Univ
ersities (ZYGX2021J012); Medico-Engineering Cooperation Funds from University of Electronic Science and Technology of China (ZYGX2021YG CX019); Chengdu Medical Research Project (2022569).

Disclosures. The authors declare no conflicts of interest.

Data availability. Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the authors upon reasonable request.

# References

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20713

Optics EXPRESS

Research Article

Vol. 31, No. 13/19 Jun 2023 / Optics Express 20714

Optics EXPRESS