# Real-time noise reduction based on ground truth free deep learning for optical coherence tomography

YONG HUANG, $^{*}$ NAN ZHANG, AND QUN HAO

School of Optics and Photonics, Beijing Institute of Technology, No. 5 South Zhongguancun Street, Haidian, Beijing, 100081, China

\*huangyong2015@bit.edu.cn

Abstract: Optical coherence tomography (OCT) is a high-resolution non-invasive 3D imaging modality, which has been widely used for biomedical research and clinical studies. The presence of noise on OCT images is inevitable which will cause problems for post-image processing and diagnosis. The frame-averaging technique that acquires multiple OCT images at the same or adjacent locations can enhance the image quality significantly. Both conventional frame averaging methods and deep learning-based methods using averaged frames as ground truth have been reported. However, conventional averaging methods suffer from the limitation of long image acquisition time, while deep learning-based methods require complicated and tedious ground truth label preparation. In this work, we report a deep learning-based noise reduction method that does not require clean images as ground truth for model training. Three network structures, including Unet, super-resolution residual network (SRResNet), and our modified asymmetric convolution-SRResNet (AC-SRResNet), were trained and evaluated using signal-to-noise ratio (SNR), contrast-to-noise ratio (CNR), edge preservation index (EPI) and computation time (CT). The effectiveness of these three trained models on OCT images of different samples and different systems was also investigated and confirmed. The SNR improvement for different sample images for $L_{2}$ -loss-trained Unet, SRResNet, and AC-SRResNet are 20.83 dB, 24.88 dB, and 22.19 dB, respectively. The SNR improvement for public images from different system for $L_{1}$ -loss-trained Unet, SRResNet, and AC-SRResNet are 19.36 dB, 20.11 dB, and 22.15 dB, respectively. AC-SRResNet and SRResNet demonstrate better denoising effect than Unet with longer computation time. AC-SRResNet demonstrates better edge preservation capability than SRResNet while Unet is close to AC-SRResNet. Eventually, we incorporated Unet, SRResNet, and AC-SRResNet into our graphic processing unit accelerated OCT imaging system for online noise reduction evaluation. Real-time noise reduction for OCT images with size of $512 \times 512$ pixels for Unet, SRResNet, and AC-SRResNet at 64 fps, 19 fps, and 17 fps were achieved respectively.

© 2021 Optical Society of America under the terms of the OSA Open Access Publishing Agreement

# 1. Introduction

Optical coherence tomography (OCT) imaging has been widely used in the field of medical diagnosis due to its advantages of non-invasiveness, high sensitivity and high resolution $[1-4]$ . However, noise is inevitably generated such as thermal noise, shot noise of the detectors and inherently speckle noise during the imaging process. The presence of noise will decrease the contrast and resolution of OCT image, resulting in the degeneration of the image quality, which can cause issues for diagnosis. At the same time, noise could also affect post-processing of OCT images, such as image segmentation $[5-7]$ . Therefore, in the field of OCT imaging, noise reduction has always been an urgent problem to be solved and one of the hot research topics.

Traditional OCT image noise reduction methods can mainly be divided into hardware and software categories. The hardware methods are mainly divided into frequency compounding

# Biomedical Optics EXPRESS

and spatial compounding (multi-frame averaging technology) [8–15]. Although multi-frame averaging has been proven to be effective in reducing noise, acquiring B-scan images at the same position multiple times requires both a long scan time and a long time for the patient to remain stationary. In addition, it depends on the accuracy of registration algorithms [14]. It is quite a challenge for patients, especially the elderly and children, and it could also cause a certain degree of discomfort [14]. The software methods rely on post image processing algorithms such as non-local mean (NLM) filtering and block-matching and 3D (BM3D) filtering algorithm [16,17]. However, these conventional noise reduction methods could inevitably cause the destruction of image details, reduce the contrast at the edge of OCT images, and result in a degeneration of image quality. Some of them also have the problem of long processing time [18,19], making them difficult to meet the clinical real-time noise reduction requirement.

Recently, deep learning has been widely used in the field of OCT image denoising. Development of convolutional neural network (CNN) has also shown great potential in recent years $[20–25]$ . The CNN can effectively extract image information from a large number of training samples, so it is widely used in the field of noise reduction. Ma et al. used a conditional generative adversarial network (cGAN) to reduce noise of retinal OCT images, and this method is better than other traditional method in performance and generalization ability $[21]$ . Qiu et al. used a convolutional network with perceptually-sensitive loss function to denoise OCT images, and this method was proved to be superior to NLM and BM3D in preserving image details $[24]$ .

Compared with traditional algorithms, deep learning noise reduction method has shown promising improvement in image quality, especially in preserving the details of the image edges. To train these deep learning methods, it is necessary to prepare the clean ground truth corresponding to the noisy images as labels $[26,27]$ . However, it is very difficult to obtain a noisy image and the corresponding clean ground truth. In the field of OCT imaging, it is often necessary to acquire multiple frames of B-scan OCT images at the same location and then register and average the images to get the ground truth, which is often a complicated process.

Based on a deep learning method named Noise2Noise, we propose a deep learning method in noise reduction for OCT images without obtaining noise free ground truth as labels $[28,29]$ . With this method, we only need to obtain any two B-scan OCT images at the same sample location, taking one noisy image as input and the other noisy image as the label. The underlying principle is that noise from two different OCT images should be different while the true sample structure should be the same. Three network structures including Unet and super-resolution residual network (SRResNet) as used in previous work $[28,29]$ and our modified asymmetric convolution super-resolution residual network (AC-SRResNet) were trained and evaluated using signal-to-noise ratio (SNR), contrast-to-noise ratio (CNR) and edge preservation index (EPI). Effectiveness of these three trained models on OCT images of different samples and different systems were investigated and compared with traditional BM3D methods. Eventually, we incorporated these three models into our graphics processing units accelerated OCT imaging system for online noise reduction evaluation. To the best of our knowledge, currently no implementation and evaluation of deep learning noise reduction methods for online real-time OCT imaging have been reported.

The remainder of this paper is organized as follows. Details of our method are illustrated in Section 2. Experimental results and comparison with traditional methods on different samples and systems in addition to real-time performance evaluation are presented and discussed in detail in Section 3. Finally, the main conclusions are presented in Section 4.

# 2. Methods

Detailed information of Noise2Noise method can be found in $[28,29]$ . With deep learning when using a convolutional neural network to deal with the image noise reduction problem, if the input image is x and the target image is y, then the noise reduction problem can be regarded as the

following parameter optimization problem:

$$
\underset {\theta} {\arg \min} E _ {(x, y)} \{L (f _ {\theta} (x), y) \} \tag {1}
$$

where L is loss function, E is the expectation of the observations, $f_{\theta}(x)$ is network function, and $\theta$ is network parameter.

If the entire training task is decomposed into same minimization problem at every training sample, according to Bayes theorem, Eq. (1) is equivalent to:

$$
\underset {\theta} {\arg \min} E _ {x} \{E _ {y | x} \{L (f _ {\theta} (x), y) \} \} \tag {2}
$$

If both the input and the label are corrupted with noises, the objective function of the network can be seen as:

$$
\underset {\theta} {\arg \min} \sum_ {i} L (f _ {\theta} (\hat {x} _ {i}), \hat {y} _ {i}) \tag {3}
$$

where $\hat{x}_{i}=x_{i}+\sigma^{\prime},\hat{y}_{i}=y_{i}+\sigma^{\prime\prime}$ are one of the i-th noisy image pairs at the same position, $x_{i}$ is the noise-reduced image and $y_{i}$ is the unobserved clean label. Both $\sigma^{\prime}$ and $\sigma^{\prime\prime}$ represent different noise contents following the same underlying distributions respectively. As demonstrated in [29], the solution of minimizing Eq. (3) will result in $f(\hat{x}_{i})=x_{i}$ given sufficient data as $E\{\hat{y}_{i}|\hat{x}_{i}\}=y_{i}$ . Therefore, as long as the number of samples is large enough, even both the input and label images have noise, the output image will be a noise-reduced image. It worth mentioning that speckle noises inherent with OCT images caused by microstructures of samples should be considered inherent structural signals instead of random noises. The reason is that speckle patterns of static samples don’t change between frames.

# 2.1. Data sources

We obtained OCT images of various samples including finger nails, hand palms, tomato, sample tooth, plastic tubes and thin films to form the data sources. The OCT system we used is a home-built spectral domain OCT imaging system, and the system parameters are as follows: the central wavelength of the light source is 1300 nm, the scanning interval frequency is 70 kHz, the axial resolution is 14 $\mu$ m, the imaging depth is 6.7 mm, the system sensitivity is 92 dB.

We collected 15 sets of OCT images of different samples, where 10 sets of images were 250 B-mode images of the sample used as the training data set, and the other 5 sets were C-mode images of the sample with 250 B-frames used as the test data set. The training data set consists of B-mode images of static samples to meet the requirement of images at the same position. In addition to that, no registration or match operation is necessary since the samples are static. Original size of the acquired image is $1000 \times 1024$ pixels. Considering the training speed-up and the limitation of GPU memory size, we adjusted the size of input image and label to $256 \times 256$ pixels. Data augmentation was performed by randomly cropping each image to get 12 sub-images and then resized, which left us with 120 training data sets.

Unlike other deep learning training methods, time-consuming pairing of input images and labels on the training data set is not necessary here. We randomly select one data set from 120 training data sets, and then randomly select two noisy images from the 250 B-scan OCT images of this data set. One of the two noisy images is used as the input image, while the other is used as the label. Any one of the 250 noisy images in this data set can be either an input image or a label.

# 2.2. Network architectures

In this work, we chose Unet and SRResNet and AC-SRResNet as our feature extraction network for comparison since they have demonstrated powerful capabilities in image feature extraction and have been widely used in image segmentation $[30]$ and noise reduction $[31]$ . Unet is a typical

# Biomedical Optics EXPRESS

lightweight CNN network while SRResNet and AC-SRResNet are relatively deeper and more powerful at feature extraction.

Architecture of our proposed AC-SRResNet is shown in Fig. 1. The $3 \times 3$ convolution kernel was replaced by asymmetric convolution block in SRResNet. Asymmetric convolutional network architecture, which replaces the convolution kernel in the CNN architecture with an asymmetric structure, can improve the accuracy of deep learning [32]. We replaced the $3 \times 3$ convolution kernel with parallel layers with $3 \times 3$ , $1 \times 3$ , and $3 \times 1$ convolution kernel respectively to improve the accuracy. If the data belongs to small batches, that is, data with a batch size less than 32, there could be a problem of reduced effectiveness during batch normalization processing. Limited by GPU memory, our batch size was set to 4, which belongs to small batches of data. Therefore, we used batch renormalization instead of batch normalization to ensure the effectiveness of normalization [33].

![](images/829f91612659d12874628b38ff6bc4506c6a91b2e9c30c67703ea9b513674dca.jpg)

<details>
<summary>flowchart</summary>

Neural network architecture diagram showing input, 8 AC-ResBlocks, and output stages with skip connections and batch normalization components.
</details>

Fig. 1. Architecture of our proposed network with corresponding kernel size(k), number of feature maps(n) and stride(s) indicated for each convolutional layer.

# 2.3. Training procedure

Training parameters were tuned empirically. We chose Adam optimizer and the learning rate $I_r$ was 0.005, and the learning decay rate was set as $1.67 \times 10^{-5}$ . The maximal iteration numbers were set to be 300. The model with minimum loss value over 300 iterations was saved for further evaluation. Both $L_1$ and $L_2$ loss functions were adopted for testing in our study, which are defined as:

$$
L _ {2} = \sum_ {i = 0} ^ {m - 1} \sum_ {j = 0} ^ {n - 1} \left[ I _ {n} (i, j) - I _ {d} (i, j) \right] ^ {2} \tag {4}
$$

$$
L _ {1} = \sum_ {i = 0} ^ {m - 1} \sum_ {j = 0} ^ {n - 1} \left| I _ {n} (i, j) - I _ {d} (i, j) \right| \tag {5}
$$

m and n are the size of the image, and in our study, the values of m and n are both $256.I_{n}(i,j)$ and $I_{d}(i,j)$ are the gray values of the output image and the label, respectively. Generally, $L_{2}$ loss function is used for noises with zero-mean, such as additive Gaussian and Poisson noises while $L_{1}$ loss function seeks to recover the median of targets [29].

# 2.4. Quantitative evaluation

We adopted signal-to-noise ratio (SNR), contrast-to-noise ratio (CNR) and edge preservation index (EPI) as the objective comparison parameters. SNR is the most common and widely used image evaluation index in the field of noise reduction. In general, the larger the value of SNR, the better the noise reduction effect of the image. It is defined as follows:

$$
S N R = 1 0 \cdot \log_ {1 0} (\frac {\sum_ {i = 0} ^ {M - 1} \sum_ {j = 0} ^ {N - 1} I _ {o} ^ {2} (i , j)}{\sigma_ {b} ^ {2}}) \tag {6}
$$

where $I_{o}$ is the value of object region, $\sigma_{b}$ is the standard deviation of the noisy background region, M and N are the ROI height and width respectively.

CNR can effectively evaluate the contrast between the object region and the noise background region. The definition of CNR is shown as below:

$$
C N R = 1 0 \cdot \log_ {1 0} (\frac {\left| \mu_ {o} - \mu_ {b} \right|}{\sigma_ {b}}) \tag {7}
$$

Where $\mu_{o}$ and $\mu_{b}$ are the average values of the object region and noisy background region respectively, $\sigma_{b}$ are the standard deviation of the noisy background region respectively.

EPI can effectively evaluate the degree of preservation of edge details of noise reduction images. It can be defined as:

$$
E P I = \frac {\sum_ {i = 0} ^ {M - 1} \sum_ {j = 0} ^ {N - 1} \left| I _ {d} (i + 1 , j) - I _ {d} (i , j) \right|}{\sum_ {i = 0} ^ {M - 1} \sum_ {j = 0} ^ {N - 1} \left| I _ {n} (i + 1 , j) - I _ {n} (i , j) \right|} \tag {8}
$$

where $I_{d}$ and $I_{n}$ are the pixel value of noise reduction image and noisy image, M and N are the ROI height and width respectively. It worth mentioning that definition of EPI captures edges in the horizontal direction as images we will test are clear and abundant in horizontal edge features.

# 2.5. Real-time online imaging system software architecture

The deep learning-based noise reduction models were integrated into our GPU accelerated customized OCT imaging software platform to evaluate their real-time online performance. The architecture of the customized software platform is shown in Fig. 2. It was developed using Visual Studio 2015 with Qt as the graphics user interface, which contains four separate threads including acquisition thread, CUDA processing thread, image plotting thread and Tensorflow C-API image denoising thread. Threads communication and synchronization were achieved using Qt signal and slots mechanism.

When the imaging process starts, the acquisition thread will acquire raw B-mode data from the spectrum data pool. Here we control the whole system imaging speed by modifying how fast we acquire the raw B-mode data. Once the raw B-mode data is ready, the acquisition thread will emit a signal to CUDA process thread to transfer that data into the GPU memory for OCT signal processing including wavelength-to-wavenumber interpolation, reference subtraction, FFT and post image magnitude and log mapping to form the raw B-mode image data. Once the raw B-mode image data is ready, the CUDA processing thread will emit a signal to the plotting thread to display the processed image and a second signal to the Tensorflow C-API image denoising thread to transfer the B-mode image data into pre-allocated deep learning processing memory to perform noise reduction and denoised image display. It needs to be pointed out that since both Tensorflow C-API image denoising and CUDA OCT signal processing share the same

# Biomedical Optics EXPRESS

![](images/c82029d1df21de226770d4f011cedde2369b282134cf95323e7ff00ed67856b4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph "Thread #2 CUDA Processing Thread"
        A["Reference Subtraction"] --> B["FFT"]
        C["λ-to-k Interpolation"] --> D["Magnitude and Log"]
        E["(≤60%)OCT Signal Processing Memory"] --> D
        F["Raw B-mode Data"] --> G["Spectrum Data Pool"]
    end

    subgraph "Thread #3 Plotting Thread"
        H["B-mode Image Data"] --> I["Image Display"]
        J["(40%) Deep Learning Processing Memory"] --> K["Noise Reduction"]
        L["Trained Model Saved"] --> M["Model Initialization"]
        N["Data Storage"] --> O["Data Storage"]
    end

    subgraph "Thread #4 Tensorflow C-API Image Denoising Thread"
        P["Denoised Image Display"] --> Q["Noise Reduction"]
        R["Trained Model Saved"] --> S["Data Storage"]
        T["Data Storage"] --> U["Image Display"]
    end

    style "Thread #2" fill:#f9f,stroke:#333
    style "Thread #3" fill:#f9f,stroke:#333
    style "Thread #4" fill:#f9f,stroke:#333
```
</details>

Fig. 2. Architecture of GPU accelerated OCT imaging software platform with integrated Tensorflow C-API image denoising thread. Red boxes show the memory allocated in both GPU and HOST side; gray boxes indicate the data processing operations. Black arrows show the data flow between threads managing both GPU and HOST memories and operations. Threads communication and synchronization were achieved using Qt signal and slots mechanism.

amount of GPU memory, we specifically allocated 40% of the GPU memory for deep learning processing and the left was for CUDA processing whenever needed. Since Tensorflow C-API image processing using tensor for image denoising, it was necessary to convert the unsigned char image data processed by CUDA to tensor specifically.

# 2.6. System implementation

Our training process is performed on a Sitonholy (Beijing, China) IW4200-4G workstation (Xeon CPU E5-2650 v4, 2.2GHz-2.9 GHz) with one graphics processing units (GPU) which is NVIDIA (Santa Clara, California, USA) Tesla P100 and 128GB of RAM, and our models were implemented with Python (v3.5.2) based on Keras (v2.1.6) with NVIDIA CUDA (v8.0) and cu-DNN (v6.1) libraries.

Our test process is performed on a personal computer with Intel Core i7-9700k CPU (3.6 GHz), Windows 10 64-bit operation system, and all methods were implemented with Python (v3.6.9) based on Keras (v2.2.0) and Tensorflow (v1.10.0).

A workstation with Intel Xeon CPU E5-2620 (2.4 GHz), 48GB RAM, Windows 10 64-bit operation system is used as the host computer for online noise reduction performance evaluation that contains one NVIDIA Geforce RTX 2080Ti GPU with 4352 stream processors, 1.4 GHz processor clock and 11GB global memory. Customized OCT imaging software was developed combing Qt (v5.6.3) and Microsoft Visual Studio 2015. Deep learning model based noise reduction was implemented with Tensorflow C-API (v1.12.0) as the engine. CUDA (v10.1) was used for GPU accelerated data processing.

# 3. Results and discussions

# 3.1. Effect of resizing in data augmentation

During the training process, for data augmentation 12 regions of interest with random sizes were first cropped from one original OCT image of size $1000 \times 1024$ and then resized into a fixed size of $256 \times 256$ using the bi-linear interpolation operation. To show the effect of the resizing in data augmentation, we trained all the three models both with and without the resizing operation using

$L_{2}$ loss function. From the loss curves shown in Fig. 3, we can see that with resizing, all three models reached lower convergence levels.

![](images/818986f39eeb6548d5ccd3e7a47394d87b0ff4633e52e1ff01c6eae1e9e6cc81.jpg)

<details>
<summary>line</summary>

| Epoch | AC-SRResNet-L2-with resizing | SRResNet-L2-with resizing | Unet-L2-with resizing | AC-SRResNet-L2-with resizing | SRResNet-L2-with resizing | Unet-L2-with resizing |
|-------|------------------------------|---------------------------|------------------------|------------------------------|---------------------------|------------------------|
| 0     | 500                          | 400                       | 900                    | 500                          | 400                       | 900                    |
| 50    | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
| 100   | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
| 150   | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
| 200   | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
| 250   | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
| 300   | 400                          | 350                       | 450                    | 400                          | 350                       | 450                    |
</details>

Fig. 3. Loss value comparison of all network models for training data sets with and without resizing operation.

One exemplary transparent gel board sample image denoising comparison is shown in Fig. 4. We can see that resizing makes the output images cleaner and contributes to better denoising effect. This agrees with the fact that resizing increases the noise spatial variety in the training dataset for the model to learn.

![](images/63487da4953b399b6df0f7ca551e4f1b07a6894ac882cf0d2df1934780b49eb6.jpg)

<details>
<summary>text_image</summary>

Input
Unet	SRResNet	AC-SRResNet
Data Augmentation Without Resizing
Data Augmentation With Resizing
</details>

Fig. 4. Transparent gel board image denoising comparison between models trained with and without resizing operation.

# 3.2. Comparison between networks

To compare the noise reduction performance of Unet, SRResNet, AC-SRResNet and traditional BM3D method, we test the results with five different samples including human finger nail, tooth sample, onion, human hand palm and transparent gel board. Figure 5 shows the original noisy image and images processed with BM3D and three models using $L_{2}$ loss function. Figure 6 shows the images processed with three models using $L_{1}$ loss function. Visual inspection of Fig. 5 can tell all three models using $L_{2}$ loss outputted noise-reduced images with better preserved details compared to BM3D as images denoised by BM3D methods show no grainy speckle pattern.

# Biomedical Optics EXPRESS

![](images/e80c87be7f4d6ffa1be37d7d3a21c511ca3c42f9e9caa696c10a435e129ade23.jpg)

<details>
<summary>text_image</summary>

(a)
Noisy image
BM3D
Unet-L2
SRResNet-L2
AC-SRResNet-L2
(b)
(c)
(d)
(e)
</details>

Fig. 5. Noise reduction results of BM3D, Unet, SRResNet and AC-SRResNet using $L_{2}$ loss function: (a) human finger nail, (b) tooth sample, (c) onion, (d) human hand palm and (e) transparent gel board. The red boxes show the object regions, the yellow box shows the background region, and the blue boxes show the edge region of image. (Scale bar: 500 $\mu$ m)

![](images/22afee7e69271c6211a247bda7ae8980140217d6bcd9549c6fd110eed4b91c98.jpg)

<details>
<summary>text_image</summary>

Unet-L₁
SRResNet-L₁
AC-SRResNet-L₁
</details>

Fig. 6. Noise reduction results of Unet, SRResNet and AC-SRResNet using $L_{1}$ loss function.

# Biomedical Optics EXPRESS

Visual inspection of Fig. 6 shows that SRResNet and AC-SRResNet using $L_{1}$ loss achieved similar performance while the Unet model generated images with noticeable nonuniform background feature compared to Fig. 5. Figure 7 shows an exemplary hand palm OCT image denoising effect comparison between all methods. The red box region marked on the original image of each denoised result was enlarged and shown on the right. From Fig. 7 we can see that while reducing the noise Unet model introduced small uniform patches on the image. Phenomenon like this is not noticeable with other two models. The reason might be that capability of Unet as a lightweight network is inferior to more sophisticated SRResNet and AC-SRResNet.

![](images/def0de0ba0c70edf73a590beb62b6a1246e14029434a54901a5afbaafbdf175f.jpg)  
Fig. 7. Denoising effect comparison of hand palm image between BM3D, Unet, SRResNet and AC-SRResNet trained with $L_{1}$ and $L_{2}$ loss. Red box shows the zoomed region on the right.

To evaluate the performance of all these methods quantitatively, we calculated the SNR, CNR, EPI and computation time (CT) shown in Table 1. Three regions of object and edge details were chosen on each image for analysis. We selected 20 noise images from above five test sets. A total of 100 images were denoised. Average SNR, CNR, EPI, and CT were calculated with these 100 images. Overall the noise in original images have been reduced. BM3D demonstrates highest SNR of 44.31 dB and CNR of 43.49 dB with lowest EPI of 0.54 and longest processing time of 21.96 s. Compared to BM3D method, deep learning-based noise reduction methods showed advantages in both detail preservation and computation time.

For Unet model, $L_{2}$ loss shows an average of 6.01 dB SNR and 5.04 dB CNR advantage over $L_{1}$ loss, which agrees with the visual inspection. In terms of EPI and CT, there is trivial difference. For SRResNet, $L_{2}$ loss shows an average 1.94 dB SNR and 1.39 dB CNR advantage while $L_{1}$ loss shows 0.04 EPI improvement. CT difference is small. For AC-SRResNet, $L_{1}$ loss shows an average 2.06 dB SNR, 1.16 dB CNR and 0.06 EPI improvement over $L_{2}$ loss. CT difference is still small. On average $L_{2}$ loss performs better than $L_{1}$ loss in removing noise while the edge preservation capability of $L_{1}$ loss is better.

Among all three models, SRResNet and AC-SRResNet perform better than Unet for both SNR and CNR while Unet holds consistent high EPI and fastest processing time due to its lightweight among three. Compared to SRResNet, introduction of asymmetric convolution in AC-SRResNet

Table 1. The comparison of the SNR (dB), CNR (dB), EPI and CT (computation time, s) of the noise-reduced images processed by BM3D, Unet, SRResNet and AC-SRResNet 

<table><tr><td>Method</td><td>SNR (dB)</td><td>CNR (dB)</td><td>EPI</td><td>CT (s)</td></tr><tr><td>Original</td><td> $19.10 \pm 1.82$ </td><td> $17.22 \pm 3.15$ </td><td>1</td><td>/</td></tr><tr><td>BM3D</td><td> $44.31 \pm 4.27$ </td><td> $43.49 \pm 6.22$ </td><td> $0.54 \pm 0.11$ </td><td>21.96</td></tr><tr><td>Unet- $L_2$ </td><td> $39.93 \pm 3.11$ </td><td> $39.62 \pm 4.76$ </td><td> $0.67 \pm 0.12$ </td><td>0.20</td></tr><tr><td>SRResNet- $L_2$ </td><td> $43.98 \pm 3.81$ </td><td> $43.66 \pm 5.09$ </td><td> $0.60 \pm 0.14$ </td><td>0.82</td></tr><tr><td>AC-SRResNet- $L_2$ </td><td> $41.29 \pm 3.59$ </td><td> $40.84 \pm 4.70$ </td><td> $0.62 \pm 0.11$ </td><td>1.00</td></tr><tr><td>Unet- $L_1$ </td><td> $33.92 \pm 2.56$ </td><td> $34.58 \pm 3.16$ </td><td> $0.67 \pm 0.15$ </td><td>0.21</td></tr><tr><td>SRResNet- $L_1$ </td><td> $42.04 \pm 3.95$ </td><td> $42.27 \pm 4.89$ </td><td> $0.64 \pm 0.12$ </td><td>0.81</td></tr><tr><td>AC-SRResNet- $L_1$ </td><td> $43.35 \pm 3.83$ </td><td> $42.00 \pm 5.66$ </td><td> $0.68 \pm 0.12$ </td><td>1.01</td></tr></table>

did improve the edge preservation capability. It is difficult to vote for the best model here as there is always trade-off between parameters.

# 3.3. Generalization ability test

To evaluate the generalization ability for different OCT system images, we tested the OCT2017 dataset (public dataset) [34]. We selected the noisy images from choroidal neovascularization (CNV), diabetic macular edema (DME), drusen and normal datasets respectively.

An exemplary image denoising comparison result of the retina fovea region is shown in Fig. 8. Visual comparison of OCT2017 dataset and our system images show that they are of different contrasts. Visual inspection finds that there are noticeable cloudy artifacts for deep learning models with $L_{2}$ loss at the weak boundary pointed out by the arrows on the image. Unet creates the most serious artifacts while AC-SRResNet shows a barely noticeable artifact. On the other hand, all three models trained with $L_{1}$ loss output no such artifacts. The reason for artifacts generation might be that $L_{2}$ loss seeks average recovery while L1 seeks the medium recovery in principle. Because the image contrast is different for our system and OCT2017 dataset, noises in OCT2017 are not zero-mean distributed that are more suitable to process with $L_{2}$ loss. For this reason, only models trained with $L_{1}$ loss were tested and compared with BM3D quantitatively on OCT2017 dataset. We can see a better generalization ability of models trained with $L_{1}$ loss compared to $L_{2}$ loss. The noise reduction results are shown in Fig. 9. All of the images have been clearly denoised.

![](images/42a23f15ab8eee36c0c0877438040ea9c2b5a4e255f166f1138f952906500966.jpg)

<details>
<summary>text_image</summary>

Origin
(a)
</details>

![](images/13938a37004cb153d26734b94159c4c5633429517df742a1462e8c4d6eb1ed92.jpg)

<details>
<summary>text_image</summary>

Unet-L₁
</details>

![](images/e08d47d50ea65c38aa1c53dc266aed551846f70e7cae21a16847f5d0ba99376a.jpg)

<details>
<summary>text_image</summary>

SRResNet-L₁
</details>

![](images/5b406bffc362bc2deaccebb60679f38028cfdd79b83a84d546df0dec74af75ec.jpg)

<details>
<summary>text_image</summary>

AC-SRResNet-L₁
</details>

![](images/22615666d9db1e1cf711fa4a87ff076b6d57dc240859bb2b0bd28d8af3acad6b.jpg)

<details>
<summary>natural_image</summary>

Grayscale medical scan image showing layered tissue structure with no visible text or symbols
</details>

![](images/7cc2e9c7e04f446c4b4494933d936b8922c23e57ba1cf036fa75387990ea0ef2.jpg)

<details>
<summary>text_image</summary>

Unet-L₂
</details>

![](images/ef793d0bf808f5fd9b82db4ccb2b37ca74ef0bd56d7a4b5522b23720021ec3ff.jpg)

<details>
<summary>text_image</summary>

SRResNet-L₂
</details>

![](images/21a13ea10aa60031881069c875a82221d2af57b5ac7b489e427d513256cb755c.jpg)

<details>
<summary>text_image</summary>

AC-SRResNet-L₂
</details>

Fig. 8. Denoising comparison for images from public dataset between BM3D, Unet, SRResNet and AC-SRResNet trained with $L_{1}$ and $L_{2}$ loss.

# Biomedical Optics EXPRESS

![](images/76ab5fad6bce9d5fd78ba0987e3d567d6ee4058ab3dd2d7a574993135216a754.jpg)

<details>
<summary>text_image</summary>

Noisy image
BM3D
Unct-L₁
SRResNet-L₁
AC-SRResNet-L₁
(a)
(b)
(c)
(d)
</details>

Fig. 9. The noise reduction results of the OCT2017 dataset with BM3D and three networks with $L_{1}$ loss: (a) CNV (b) DME (c) Drusen and (d) normal images. The red boxes are the object regions, the yellow box shows the background region, and the blue boxes are the edge regions of image.

Parameter comparison results are shown in Table 2. BM3D shows lowest SNR and CNR improvement here. Among deep learning models, we can see AC-SRResNet achieves the highest SNR of 41.80 dB and CNR of 44.64 dB at the cost of longest CT of 0.98 s, while U-net achieves the lowest SNR of 39.01 dB and CNR of 41.06 dB with the advantage of shortest CT of 0.2 s. SRResNet gets a moderate SNR of 39.76 dB and CNR of 42.72 dB. For the EPI values, Unet model tends to achieve a higher EPI value than other two models. The reason might be that the denoising effect of Unet is lowest since the original image with no noise removed will get an EPI value of 1. AC-SRResNet shows consistent detail preservation advantage over SRResNet again.

Table 2. The comparison of the SNR (dB), CNR (dB), EPI and CT (computation time, s) of OCT2017 dataset processed by BM3D, Unet, SRResNet and AC-SRResNet 

<table><tr><td>Method</td><td>SNR (dB)</td><td>CNR (dB)</td><td>EPI</td><td>CT (s)</td></tr><tr><td>Original</td><td> $19.65 \pm 1.51$ </td><td> $19.79 \pm 2.05$ </td><td>1</td><td>/</td></tr><tr><td>BM3D</td><td> $36.00 \pm 6.01$ </td><td> $38.71 \pm 7.09$ </td><td> $0.68 \pm 0.03$ </td><td>23.26</td></tr><tr><td>Unet-L $_1$ </td><td> $39.01 \pm 3.91$ </td><td> $41.06 \pm 4.64$ </td><td> $\mathbf{0.73} \pm \mathbf{0.03}$ </td><td> $\mathbf{0.20}$ </td></tr><tr><td>SRResNet-L $_1$ </td><td> $39.76 \pm 1.79$ </td><td> $42.72 \pm 2.20$ </td><td> $0.67 \pm 0.03$ </td><td>0.78</td></tr><tr><td>AC-SRResNet-L $_1$ </td><td> $\mathbf{41.80} \pm \mathbf{1.63}$ </td><td> $\mathbf{44.64} \pm \mathbf{2.09}$ </td><td> $0.72 \pm 0.03$ </td><td>0.98</td></tr></table>

Based on the analysis results of our system images and public dataset, we can see that deep learning models trained based on Noise2Noise principle all achieved good denoising performance. Although speckle pattern is still remaining with the image as expected, certain smoothing effect can be observed, which will reduce the speckle contrast. This effect on future applications such as speckle-based OCT angiography (OCTA) analysis for images in the logarithm domain requires further study in the future. Please note that method proposed in this manuscript denoises OCT

images in the logarithm domain. For intensity or amplitude based OCTA analysis might not be able to use this method directly.

# 3.4. Real-time online imaging test results

We tested the online performance of the noise reduction models of Unet, SRResNet and AC-SRResNet trained with $L_{2}$ loss for the whole B-mode image with size of $512 \times 512$ pixels as $L_{2}$ loss offers better performance for our system images. The results are shown in Table 3. When tested off-line without considering the image data transfer in memory and format change, Unet, SRResNet and AC-SRResNet based noise reduction model can reach 75 fps, 21 fps and 19 fps respectively. On the real-time online imaging system, Unet, SRResNet and AC-SRResNet based noise reduction model can achieve a real-time maximal 64 fps, 19 fps and 17 fps image denoising without causing program interface freezing respectively. Screen capture of real-time finger nail image noise reduction are shown in Fig. 10 with zoomed area of interest on the right. Visualization 1, Visualization 2, and Visualization 3 are the videos showing image denoising results of Unet, SRResNet and AC-SRResNet for in vivo human hand palm images respectively. The processing speed based on Unet model is the fastest due to the simplicity and relative shallow network depth of the structure.

![](images/4a6d783fbf84b385fac76a6d22ddefe45d8b06e7a1b6a458f09cc41d38b8d57b.jpg)

Fig. 10. Real-time denosing results of three methods:(a) Unet, (b) SRResNet, and (c) AC-SRResNet.   
Table 3. The comparison of time consumption in model processing and real-time online imaging system by Unet, SRResNet and AC-SRResNet 

<table><tr><td>Method</td><td>Off-line (fps)</td><td>On-line (fps)</td></tr><tr><td>Unet</td><td>75</td><td>64</td></tr><tr><td>SRResNet</td><td>21</td><td>19</td></tr><tr><td>AC-SRResNet</td><td>19</td><td>17</td></tr></table>

# Biomedical Optics EXPRESS

From the real-time test, we can see that there is a tradeoff between image quality improvement and processing time. As more layers are added to the network feature extraction capability can be increased thus better noise discrimination and reduction can be achieved. However, the side-effect of more time consumption is becoming obvious. Nevertheless, for application scenarios where time consumption might not be a crucial factor, sophisticated network model can be adopted. Meanwhile, there is still improvement for the processing time reduction. Currently there is only one GPU configured for both OCT signal processing and deep learning image denoising. We tested installation of a second GPU and tried to implement the multithread-controlled data parallelism technique to further reduce the processing time. However, current Tensorflow C-API library doesn't support independent configuration of each GPU for noise reduction. Once the library support is ready, we believe further processing time reduction is probable.

# 4. Conclusions

We proposed a deep learning-based noise reduction method for OCT images that requires no noise free ground truth images as labels. Comparison study with conventional noise reduction method BM3D and different network structures including Unet, SRResNet and modified AC-SRResNet trained with $L_{2}$ and $L_{1}$ loss respectively were performed. Further incorporation into online OCT imaging system for real-time noise reduction was demonstrated for images with size of $512 \times 512$ at 64 fps for Unet, 19 fps for SRResNet and 17 fps for AC-SRResNet. We believe proposed methods will benefit future responsive deep learning-based OCT signal processing and analysis platform.

Funding. National Natural Science Foundation of China (61505006); Beijing Institute of Technology (2018CX01018); Overseas Expertise Introduction Project for Discipline Innovation (B18005); CAST Innovation Foundation (2018QNRC001).

Disclosures. The authors declare no conflicts of interest.

# References

1. D. Huang, E. A. Swanson, C. P. Lin, J. S. Schuman, W. G. Stinson, W. Chang, M. R. Hee, T. Flotte, K. Gregory, and C. A. Puliafito, "Optical coherence tomography," Science 254(5035), 1178-1181 (1991).   
2. W. Drexler, U. Morgner, R. K. Ghanta, F. X. Kärtner, J. S. Schuman, and J. G. Fujimoto, “Ultrahigh-resolution ophthalmic optical coherence tomography,” Nat. Med. 7(4), 502–507 (2001).   
3. T. Gambichler, G. Moussa, M. Sand, D. Sand, P. Altmeyer, and K. Hoffmann, “Applications of optical coherence tomography in dermatology,” J. Dermatol. Sci. 40(2), 85–94 (2005).   
4. R. F. Spaide, J. G. Fujimoto, N. K. Waheed, S. R. Sadda, and G. Staurenghi, “Optical coherence tomography angiography,” Prog. Retinal Eye Res. 64, 1–55 (2018).   
5. S. Asrani, L. Essaid, B. D. Alder, and C. Santiago-Turla, “Artifacts in spectral-domain optical coherence tomography measurements in glaucoma,” JAMA Ophthalmol 132(4), 396–402 (2014).   
6. Y. Liu, H. Simavli, C. J. Que, J. L. Rizzo, E. Tsikata, R. Maurer, and T. C. Chen, “Patient characteristics associated with artifacts in Spectralis optical coherence tomography imaging of the retinal nerve fiber layer in glaucoma,” Am. J. Ophthalmol. 159(3), 565–576.e2 (2015).   
7. K. E. Kim, J. W. Jeoung, K. H. Park, D. M. Kim, and S. H. Kim, “Diagnostic classification of macular ganglion cell and retinal nerve fiber layer analysis: differentiation of false-positives from glaucoma,” Ophthalmology 122(3), 502–510 (2015).   
8. A. E. Desjardins, B. J. Vakoc, W. Y. Oh, S. M. R. Motaghiannezam, G. J. Tearney, and B. E. Bouma, "Angle-resolved optical coherence tomography with sequential angular selectivity for speckle reduction," Opt. Express 15(10), 6200–6209 (2007).   
9. T. Klein, R. André, W. Wieser, T. Pfeiffer, and R. Huber, "Joint aperture detection for speckle reduction and increased collection efficiency in ophthalmic MHz OCT," Biomed. Opt. Express 4(4), 619-634 (2013).   
10. M. Pircher, E. Gotzinger, R. Leitgeb, A. F. Fercher, and C. K. Hitzenberger, “Speckle reduction in optical coherence tomography by frequency compounding,” J. Biomed. Opt. 8(3), 565–569 (2003).   
11. T. Bajraszewski, M. Wojtkowski, M. Szkulmowski, A. Szkulmowska, R. Huber, and A. Kowalczyk, “Improved spectral optical coherence tomography using optical frequency comb,” Opt. Express 16(6), 4163–4176 (2008).   
12. J. M. Schmitt, S. H. Xiang, and K. M. Yung, “Speckle in optical coherence tomography,” J. Biomed. Opt. 4(1), 95–105 (1999).   
13. V. Behar, D. Adam, and Z. Friedman, “A new method of spatial compounding imaging,” Ultrasonics 41(5), 377–384 (2003).   
14. W. Wu, O. Tan, R. R. Pappuru, H. Duan, and D. Huang, “Assessment of frame-averaging algorithms in OCT image analysis,” Ophthalmic Surg Lasers Imaging Retina 44(2), 168–175 (2013).

# Biomedical Optics EXPRESS

15. B. F. Kennedy, T. R. Hilman, A. Curatolo, and D. D. Sampson, “Speckle reduction in optical coherence tomography by strain compounding,” Opt. Lett. 35(14), 2445–2447 (2010).   
16. B. Chong and Y. K. Zhu, “Speckle reduction in optical coherence tomography images of human finger skin by wavelet modified BM3D filter,” Opt. Commun. 291, 461–469 (2013).   
17. J. Aum, J. Kim, and J. Jeong, “Effective speckle noise suppression in optical coherence tomography images using nonlocal means denoising filter with double Gaussian anisotropic kernels,” Appl. Opt. 54(13), D43–D50 (2015).   
18. M. Li, R. Idoughi, B. Choudhury, and W. Heidrich, “Statistical model for OCT image denoising,” Biomed. Opt. Express 8(9), 3903–3917 (2017).   
19. S. Chitchian, M. A. Mayer, A. R. Boretsky, F. J. van Kuijk, and M. Motamedi, "Retinal optical coherence tomography image enhancement via shrinkage denoising using double-density dual-tree complex wavelet transform," J. Biomed. Opt. 17(11), 116009 (2012).   
20. K. J. Halupka, B. J. Antony, M. H. Lee, K. A. Lucy, R. S. Rai, H. Ishikawa, G. Wollstein, J. S. Schuman, and R. Garnavi, “Retinal optical coherence tomography image enhancement via deep learning,” Biomed. Opt. Express 9(12), 6205–6221 (2018).   
21. Y. Ma, X. Chen, W. Zhu, X. Cheng, D. Xiang, and F. Shi, “Speckle noise reduction in optical coherence tomography images based on edge-sensitive cGAN,” Biomed. Opt. Express 9(11), 5129–5146 (2018).   
22. F. Shi, N. Cai, Y. Gu, D. Hu, Y. Ma, Y. Chen, and X. Chen, “DeSpecNet: a CNN-based method for speckle reduction in retinal optical coherence tomography images,” Phys. Med. Biol. 64(17), 175010 (2019).   
23. Z. Mao, A. Miki, S. Mei, Y. Dong, K. Maruyama, R. Kawasaki, S. Usui, K. Matsushita, K. Nishida, and K. Chan, "Deep learning-based noise reduction method for automatic 3D segmentation of the anterior of lamina cribrosa in optical coherence tomography volumetric scans," Biomed. Opt. Express 10(11), 5832-5851 (2019).   
24. B. Qiu, Z. Huang, X. Liu, X. Meng, Y. You, G. Liu, K. Yang, A. Maier, Q. Ren, and Y. Lu, “Noise reduction in optical coherence tomography images using a deep neural network with perceptually-sensitive loss function,” Biomed. Opt. Express 11(2), 817–830 (2020).   
25. S. K. Devalla, G. Subramanian, T. H. Pham, X. Wang, S. Perera, T. A. Tun, T. Aung, L. Schmetterer, A. H. Thiéry, and M. J. A. Girard, “A deep learning approach to denoise optical coherence tomography images of the optic nerve head,” Sci. Rep. 9(1), 14454 (2019).   
26. A. Abdelhamed, S. Lin, and M. S. Brown, “A high-quality denoising dataset for smartphone cameras,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, (2018), pp. 1692–1700.   
27. S. Nam, Y. Hwang, Y. Matsushita, and S.J. Kim, “A holistic approach to cross-channel image noise modeling and its application to image denoising,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, (2016), pp. 1683–1691.   
28. A. Krull, T. O. Buchholz, and F. Jug, “Noise2void-learning denoising from single noisy images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 2129–2137.   
29. J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, “Noise2Noise: learning image restoration without clean data,” in Proceedings of the 35th International Conference on Machine Learning, 2018, 80:2965–2974.   
30. A. Shah, L. Zhou, M. D. Abrámoff, and X. Wu, “Multiple surface segmentation using convolution neural nets: application to retinal layer segmentation in OCT images,” Biomed. Opt. Express 9(9), 4509–4526 (2018).   
31. A. C. Guei and M. Akhloufi, “Deep learning enhancement of infrared face images using generative adversarial networks,” Appl. Opt. 57(18), D98–D107 (2018).   
32. X. Ding, Y. Guo, G. Ding, and J. Han, “ACNet: strengthening the kernel skeletons for powerful CNN via asymmetric convolution blocks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1911–1920.   
33. B. A. Jonsson, G. Bjornsdottir, T. E. Thorgeirsson, L. M. Ellingsen, G. B. Walters, D. F. Gudbjartsson, H. Stefansson, K. Stefansson, and M. O. Ulfarsson, “Brain age prediction using deep learning uncovers associated sequence variants,” Nat. Commun. 10(1), 5409 (2019).   
34. D. S. Kermany, M. Goldbaum, W. Cai, C. C. S. Valentim, H. Liang, S. L. Baxter, A. McKeown, G. Yang, X. Wu, F. Yan, J. Dong, M. K. Prasadha, J. Pei, M. Y. L. Ting, J. Zhu, C. Li, S. Hewett, J. Dong, I. Ziyar, A. Shi, R. Zhang, L. Zheng, R. Hou, W. Shi, X. Fu, Y. Duan, V. A. N. Huu, C. Wen, E. D. Zhang, C. L. Zhang, O. Li, X. Wang, M. A. Singer, X. Sun, J. Xu, A. Tafreshi, M. A. Lewis, H. Xia, and K. Zhang, “Identifying medical diagnoses and treatable diseases by image-based deep learning,” Cell 172(5), 1122–1131.e9 (2018).