# Reproduce of SimCLR


### Reproduced by 
### Chaoyi Zhu, Congwen Chen, Zhiyang Liu

---------------------

This blog aims to present the reproduction work of the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf). The paper inroduces a new method for contrastive leanring of visual representations. This work achieved strong results and outperformed previous methods for self-supervised and semi-supervised learning on ImageNet. Our objective is to reproduce the results in Table 8 of the original paper as shown below.

![](https://i.imgur.com/JORMqqp.png)

## Introduction
---------------------
For decades, a large class of ML methods rely on human-provided labels or rewards as the only form of learning signals used during the training process. These methods, known as Supervised Learning approaches, heavily rely on the amount of annotated training data available. But as is known, annotating data is not cheap. On the other hand, if we look around, data, in an unsupervised way, is abundant. This is where self-supervised learning methods plays a vital role in the process of deep learning without requiring expensive annotations.

### Self-supervised Learning

From a classification perspective, self-supervised learning is a subset of unsupervised learning. It aims to learn some general representation for downstream tasks.

Typically, in the process of self-supervised learning, as shown below, unlabeled data is used for pre-training. After that, the model is fine-tuned in a supervised way according to the downstream tasks.

![](https://i.imgur.com/PBammSs.png)

### Contrastive Self-supervised Learning

As one of the most popular topics in the past few years, Contrastive learning methods, as the name implies, learn representations by contrasting positive and negative examples. They aim to group similar samples closer and diverse samples far from each other. Contrastive self-supervised learning is to use such methods in the pre-training process, and has led to great empirical success in computer vision tasks.

The main motivation for contrastive learning comes from human learning patterns. Humans recognize objects without remembering all the little details. For example, Epstein ran an experiment in 2016 that asked subjects to draw dollars in as much detail as possible. In the figure below, the left one is the result drawn by people without any reference, and the right one is drawn with a bill (not a One Dollar bill) in hand. People are very familiar with bills, but this experiment shows that some more abstract features（Like figures in the corners, a portrait in the middle.）, intead of all the details help people to recognize or remember an item.

![](https://i.imgur.com/MScuyWA.png)

Roughly speaking, we create some kind of representations in our minds, and then we use them to recognize new objects. And the main goal of contrastive self-supervised learning is to create and generalize these representations.

More formally, for any data point x, contrastive methods aim to learn an encoder f such that:

$$\text { similarity_score }\left(f(x), f\left(x^{+}\right)\right)>>\text {similarity_score }\left(f(x), f\left(x^{-}\right)\right)$$

In the formula above, x+ is a data point similar to the input x. In other words, the observations x and x+ are correlated and the pair (x,x+) represents a positive sample. In most cases, we can implement different augmentation techniques(Image rotation, cropping and etc.) to generate those samples. In other words, in contrastive learning, we aim to minimize the difference between the positive pairs while maxmizing the difference between positives and negatives.

### A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

And here comes the method we are about to reproduce, SimCLR. It uses the principles of contrastive learning we described above. The idea of SimCLR framework is very simple. An image is taken and random transformations are applied to it to get a pair of two augmented images Xi and Xj. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations Z. The task is to maximize the similarity between these two representations Zi and Zj for the same image. The architecture of SimCLR is shonw below.

![](https://i.imgur.com/nBp4aSF.jpg)

Here we give a discription of the procedure of training steps in SimCLR. To start with, we have a training corpus, which consists of unlabeled images.

And first of all, we perform data augmentations on a batch. The actual batch size might be a big number like 8192. For the convenience of introduction, we will use a small batch size N = 2 to explain here.

![](https://i.imgur.com/tDoPcvZ.png)

Many possible operations of augmentations are available. The authors of the paper concluded, however, the composition of random cropping and random color distortion stands out.

![](https://i.imgur.com/XfqFzwT.png)

For each image in a batch, w
