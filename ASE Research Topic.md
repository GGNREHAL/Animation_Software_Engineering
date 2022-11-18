# Neural Denoiser  
#### Gagandeep Singh Rehal (s5532166)

## Abstract

Neural Denoiser is a class of machine learning algorithms that use supervised learning to reduce noise in images or videos. A light transport technique with low sampling is used in the visual effects industry to reconstruct image sequences. Denoising an image has been a classic problem for decades, and neural networks have been helpful to the field as they have developed rapidly. They can, however, be used only in limited instances due to the difficulty in obtaining large quantities of noisy-clean image pairs for supervision. There are many sources of noise that complicate image denoising such as Gaussian, impulse, salt, pepper, and speckle noises. In the past, multiple attempts have been made to train image-denoising models with noisy images, but these approaches failed due to inefficient network training, loss of information, or reliance on noise modeling. The purpose of this report is to develop a supervised algorithm that aims to eliminate image noise. This model will be trained using the Smartphone Image Denoising Dataset (SIDD) and RENOIR which both combined contain over thousand noisy images captured under different lighting conditions with different cameras. In addition, this paper will explain the approach from a theoretical perspective and provides appropriate examples to further validate it.


## Detailed Description

Image denoising is an essential component of image processing which received overwhelming interest in the last decade. As per Ilesanmi et al. (2021) image denoising has become more important as a result of the increasing production of images taken in poor lighting conditions. Pang et al. (2021) say, besides serving as a fundamental component of many image recovery methods, image denoising is a significant problem in its own right. The noise in an image can be expressed as follows:

y = x + n,

Where y represents the noise, x denotes the noise-free image for recovery, and n stands for measurement noise. There are many ways to define the noise n, but the most common one is to consider the instance drawn from some distribution. Recent years have seen rapid growth in deep neural network-based image denoising. Zhang et al., (2016)  designed DnCNN to denoise images by combining convolutional neural networks and residual learning. The DnCNN algorithm outperforms traditional denoisers by a large margin when supervised by a noisy-clean paired image. As per Huang et al., (2021) these models require a huge amount of noisy-clean data for training and it is extremely challenging and expensive to collect a large amount of data for training. This is one of the limits of supervised denoisers. I aim to develop a method for denoising digital photographs that recognizes noise statistics and noise formation in low-light images by using supervised learning.

### Datasets
#### Smartphone Image Denoising Dataset (SIDD)

Abdelhamed et al., (2018) state that smartphone cameras have taken over DSLRs and point-and-shoots in the last decade. Compared to DSLRs, smartphone images have notably more noise due to their small apertures and sensors. While denoising for smartphone images is an active research area, no denoising dataset with high-quality ground truth for real noisy smartphone images exists. The Smartphone Image Denoising Dataset (SIDD), consists of over 30,000 noisy images from 10 scenes under different lighting conditions using five representative smartphone cameras. To determine the patterns and statistics of image noise, some researchers used this dataset to benchmark a number of denoising algorithms.

#### RENOIR

Anaya et al., (2018) introduced pixel and intensity-aligned clean images along with images corrupted by real low-light noise for the first time. It contains about 500 images of 120 scenes. Scientists could use the dataset to study low-light digital camera noise formation and statistics or to train and test image-denoising algorithms. Denoising algorithms must adjust their internal parameters to accommodate different levels of noise within a dataset since the images have different levels of noise.

Description of the datsets you will use, if creating your own, describe how you it will be collected.

### Arcitecture Proposal

Description of the arcitecture you will use (include diagrams and references as appropriate).

## References



