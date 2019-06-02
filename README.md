## Normalizing VGG mean activations

This is a Keras implementation of normalization of VGG-like networks so that the mean activation (over images in a dataset and positions in the respective activation map) of convolutional filters in the network is equal to 1. This is mentioned in passing in the [original style transfer paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et al:

> We normalized the network by scaling the weights such that the mean 
> activation of each convolutional ﬁlter over images and positions is 
> equal to one. Such re-scaling can be done for the VGG network without 
> changing its output, because it contains only rectifying linear 
> activation functions and no normalization or pooling over feature maps.

The benefit of doing it is that losses based on features extracted from different layers of the network (e.g. Gramian based style losses) will have comparable magnitude. In turn, per-layer loss weights will be more interpretable.

Here is a (smoothed) plot of the distribution of mean activations of the VGG16 network across a few convolutional layers before normalization (mean activations calculated over 80000 images of the COCO dataset):

![](tex/Figure.png)



#### How normalization works

First, we gather the mean activations of convolutional filters across all filter activations over all images in the dataset. Then, normalization is done sequentially, from bottom layers to top layers.

Let <img src="/tex/9da210c1056f1d4fc545b85887f01662.svg?invert_in_darkmode&sanitize=true" align=middle width=22.03204574999999pt height=27.91243950000002pt/> and <img src="/tex/74bab7444df239c6eba1eca4d906d782.svg?invert_in_darkmode&sanitize=true" align=middle width=11.705695649999988pt height=27.91243950000002pt/> be the weights and bias of the <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>-th convolutional filter in layer <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/>. The shape of the convolution kernel <img src="/tex/9da210c1056f1d4fc545b85887f01662.svg?invert_in_darkmode&sanitize=true" align=middle width=22.03204574999999pt height=27.91243950000002pt/> is <img src="/tex/fd5cc10255005dce836ac11ab58333da.svg?invert_in_darkmode&sanitize=true" align=middle width=68.97814769999998pt height=22.831056599999986pt/> (height, width, incoming channels), but for notational simplicity let's think of it as if it is reshaped to <img src="/tex/4af5f6eef7d763977335519a0ed1deae.svg?invert_in_darkmode&sanitize=true" align=middle width=35.47556264999999pt height=19.1781018pt/>, where <img src="/tex/d7758142c3378032cd2bf106a88a20b7.svg?invert_in_darkmode&sanitize=true" align=middle width=71.96134934999999pt height=22.831056599999986pt/>.

<img src="/tex/f4a5de318f6c84ee751696e5d6451704.svg?invert_in_darkmode&sanitize=true" align=middle width=216.07961264999997pt height=30.04564529999999pt/> is the activation of the <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>-th filter in layer <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> at the <img src="/tex/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode&sanitize=true" align=middle width=7.710416999999989pt height=21.68300969999999pt/>-th position in the activation map. Here <img src="/tex/22fcde5697fb6ff191e860c19adb9cf6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.219209349999991pt height=14.611911599999981pt/> designates the convolution operation (Frobenius inner product)  and <img src="/tex/a79a3bd74f6fbbb4331bcb94825840ba.svg?invert_in_darkmode&sanitize=true" align=middle width=33.88712744999999pt height=30.04564529999999pt/> is the patch of <img src="/tex/eb1660404e5a1d00dc7bf0c2d4be3a42.svg?invert_in_darkmode&sanitize=true" align=middle width=126.37134015pt height=22.831056599999986pt/> activations in layer <img src="/tex/b4ba0aa01606a3064bb00827756e5407.svg?invert_in_darkmode&sanitize=true" align=middle width=33.53874479999999pt height=22.831056599999986pt/> that the filter convolves with.

Let <img src="/tex/40d81f7863a7f4cee08e2e13fce9aeb4.svg?invert_in_darkmode&sanitize=true" align=middle width=556.43666595pt height=37.86700830000002pt/> be the mean activation of the <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>-th filter in layer <img src="/tex/2f2322dff5bde89c37bcae4116fe20a8.svg?invert_in_darkmode&sanitize=true" align=middle width=5.2283516999999895pt height=22.831056599999986pt/> over all <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> images in the dataset <img src="/tex/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode&sanitize=true" align=middle width=14.908688849999992pt height=22.465723500000017pt/> and all <img src="/tex/1d8f66048e25fe1691f3bd927460247f.svg?invert_in_darkmode&sanitize=true" align=middle width=21.963504749999988pt height=27.91243950000002pt/> positions in the filter's activation map. Note that this is a non-negative number (and, in fact, positive for the ImageNet pre-trained VGG networks).

Now, the naive approach to normalizing the activations would be to just divide weights and biases by <img src="/tex/86514716e880ed55528f9b60a1d270f7.svg?invert_in_darkmode&sanitize=true" align=middle width=14.555823149999991pt height=27.91243950000002pt/>. This would make the mean of the activation equal to 1, **if the previous layer activations were the same as the original non-normalized activations**. That is, <img src="/tex/acb2b3ee5d8260d09ffd3017029e65e4.svg?invert_in_darkmode&sanitize=true" align=middle width=239.51854245pt height=38.10404400000003pt/>, but **only if** the incoming activations <img src="/tex/a79a3bd74f6fbbb4331bcb94825840ba.svg?invert_in_darkmode&sanitize=true" align=middle width=33.88712744999999pt height=30.04564529999999pt/> would stay the same as in the original non-normalized network. It is only true for the first convolutional layer in the normalized network. For any other layer this will not only result in wrong scale, but it can actually reverse the sign of the convolution and, consequently, incorrectly zero out or switch on activations after passing through the ReLU.

There is an easy fix, though. We know exactly how we scaled the previous layer, and we can undo it in the current layer. We rescale the subset of weights in <img src="/tex/9da210c1056f1d4fc545b85887f01662.svg?invert_in_darkmode&sanitize=true" align=middle width=22.03204574999999pt height=27.91243950000002pt/> which interact with the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-th channel in layer <img src="/tex/a15d8b44f037a19a21b672e2ac5761ee.svg?invert_in_darkmode&sanitize=true" align=middle width=33.53874479999999pt height=22.831056599999986pt/> by multiplying them by <img src="/tex/3d33406f14d06bbd046fadc1f27cb8d1.svg?invert_in_darkmode&sanitize=true" align=middle width=30.95527874999999pt height=30.04564529999999pt/>. This cancels out the normalization of the previous layer.

More formally, let

<img src="/tex/a4e956cfd6d7db5c50b947fdbbb4357d.svg?invert_in_darkmode&sanitize=true" align=middle width=246.33806174999998pt height=100.64133089999999pt/> 

be the diagonal <img src="/tex/e07aa63276786f9bac5f59c25ecaf0ad.svg?invert_in_darkmode&sanitize=true" align=middle width=34.31880044999999pt height=19.1781018pt/> matrix of all <img src="/tex/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode&sanitize=true" align=middle width=7.11380504999999pt height=14.15524440000002pt/> mean activations in layer <img src="/tex/b4ba0aa01606a3064bb00827756e5407.svg?invert_in_darkmode&sanitize=true" align=middle width=33.53874479999999pt height=22.831056599999986pt/>. 

Then, <img src="/tex/e3a406f44bae51e7203867f389df7801.svg?invert_in_darkmode&sanitize=true" align=middle width=269.8883649pt height=38.10404400000003pt/>.

The above may look mathy, but it results in only 3 lines of code. For each convolutional layer after the first we do:

```python
# conv layer weights layout is: (dim1, dim2, in_channels, out_channels)
W *= prev_conv_layer_means[np.newaxis, np.newaxis, : , np.newaxis]
b /= means
W /= means[np.newaxis, np.newaxis, np.newaxis, :]
```
Note that the presence of max or average pooling layers doesn't change anything: they simply propagate the scale of the activations from the previous to the next convolutional layer. This means we can simply skip them while normalizing the network.

