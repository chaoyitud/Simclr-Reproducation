<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# Reproduce of SimCLR


### Reproduced by 
    Chaoyi Zhu, Congwen Chen, Zhiyang Liu

---------------------

This blog aims to present the reproduction work of the paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf). The paper introduces a new method for contrastive learning of visual representations. The innovation of this work is to use **aggressive data augmentations**. The resulting "harder" tasks can dramatically improve the quality of learned representations. This work achieved strong results and outperformed previous methods for self-supervised and semi-supervised learning on ImageNet and strong generalization performance on other datasets as well. Our objective is to reproduce the results in Table 8 of the original paper as shown below.

The structure of this blog can be divided into 2 parts, including introduction of the content in the paper and the work we have done. Our working generally includes reproduction of the paper, extending application on RPLAN dataset and visualization of the training result.

![](https://i.imgur.com/JORMqqp.png)
## Author Contribution
The blog writing is equally distributed among different authors.

Chaoyi Zhu: Preprocess RPLAN dataset and visualize the training result on CIFAR10 and RPLAN.

Congwen Chen: Finetune and apply linear evaluation on different models. 

Zhiyang Liu: Train the model using the official repo and convert the official checkpoint to Pytorch version.

---------------------
## Introduction
For decades, a large class of ML methods relies on human-provided labels or rewards as the only form of learning signals used during the training process. These methods, known as Supervised Learning approaches, heavily rely on the amount of annotated training data available. Although raw data is vastly available, annotating data is known to be expensive.

### Self-supervised Learning

Self-supervised learning (SSL) is a method of machine learning. It learns from unlabeled sample data, and can be regarded as an intermediate form between supervised and unsupervised learning.

As one may ask, how can we train a neural network without labels? Neural networks are generally trained on some objective function. Without labels, how can we measure the performence of a network? Self-supervised learning answers this question by proposing **tasks** for the network to solve, where performance is easy to measure. For example, in the field of Computer Vision(CV), the task could be filling in image holes, or colorizing grayscale images. Ideally, a good task will be difficult to solve if the network cannot capture some form of image semantics.

Neural networks pre-trained on these tasks can be fine-tuned on downstream tasks with less labeled data than those initialized randomly. A typical procedure of SSL is shown in the figure below.


![](https://i.imgur.com/PBammSs.png)

### Contrastive Self-supervised Learning

As one of the most popular topics in the past few years, Contrastive learning methods, as the name implies, learn representations by contrasting positive and negative examples. They aim to group similar samples closer and diverse samples far from each other. Contrastive self-supervised learning is to use such methods in the pre-training process and has led to great empirical success in computer vision tasks.

The main motivation for contrastive learning comes from human learning patterns. Humans recognize objects without remembering all the little details. For example, Epstein ran an [experiment](https://aeon.co/essays/your-brain-does-not-process-information-and-it-is-not-a-computer) in 2016 that asked subjects to draw dollars in as much detail as possible. In the figure below, the left one is the result drawn by people without any reference, and the right one is drawn with a bill (not a One Dollar bill) in hand. People are very familiar with bills, but this experiment shows some more abstract features（Like figures in the corners, a portrait in the middle.）, instead of all the details that help people to recognize or remember an item.

![](https://i.imgur.com/MScuyWA.png)

Roughly speaking, we create some kind of representation in our minds, and then we use them to recognize new objects. And the main goal of contrastive self-supervised learning is to create and generalize these representations.

More formally, for any data point $x$, contrastive methods aim to learn an encoder $f$ that maximizes $similarity(f(x), f(x^+))$ and minimizes $similarity(f(x), f(x^-))$. Here, $x^+$ is a data point similar to the input $x$. In other words, the observations $x$ and $x^+$ are correlated and the pair $(x,x^+)$ represents a positive sample, while $x$ and $x^-$ are unrelated and $(x,x^-)$ represents a negative pair. In most cases, we can implement different augmentation techniques (Image rotation, cropping and etc.) to generate positive samples. In contrastive learning, we aim to minimize the difference between the positive pairs while maximizing the difference between positives and negatives.

### A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)

And here comes the method we are about to reproduce, SimCLR[1]. It uses the principles of contrastive learning we described above. As mentioned in former parts, the essence of strength of the work is the aggressive data augmentation. By doing so, "harder" samples are generated to train the ability of the network to learn representations. In fact, the authors demonstrated in the paper that such stronger data augmentation can benefit unsupervised contrastive learning "dramatically" while the same augmenation does not improve and even hurts performance of supervised models.

The architecture of SimCLR is shown below. An image is taken and random transformations are applied to it to get a pair of two augmented images $x_i$ and $x_j$. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations $z_i$ and $z_j$. The task is to maximize the similarity between these two representations $z_i$ and $z_j$ for the same image. 

![](https://i.imgur.com/nBp4aSF.jpg)

Here we give a description of the procedure of training steps in SimCLR. To start with, we have a training corpus, which consists of unlabeled images.

**1. Data augmentation**

And first of all, we perform data augmentations on a batch. The actual batch size might be a big number like 8192. For the convenience of introduction, we will use a small batch size N = 2 to explain here.

![](https://i.imgur.com/tDoPcvZ.png)

Many possible operations of augmentations are available. In the paper, various data augmentation operations and their combinations are tested, involving spatial transformations like cropping and rotation and appearance transformations like color distortion.

The authors of the paper concluded, after experiments:
1. no single transformation suffices to learn good representations
2. the quality of representation improves dramatically when composing augmentations

And the composition of random cropping and random color distortion stands out.

![](https://i.imgur.com/XfqFzwT.png)

For each image in a batch, we get an augmented version of it. So for a batch size of N, we get 2N images.


**2. Encoding**

The pair of images $(x_i, x_j)$ then will be encoded to get the representations. Usually the encoded representations are of much lower dimensions, which is more efficient to work with. The encoder is general and can be replaced by other possible designs. ResNet-50 is used in this paper.

![](https://i.imgur.com/fgmyacm.png)

**3. Projection head**

In SimCLR, visual representations $h_i, h_j$ obtained by encoders are then processed by a projection head $g(.)$. And the final representation $z = g(h)$. In the paper, the projection head is a Multilayer Perception with 2 dense layers, and the hidden layer uses a ReLU activation function.

![](https://i.imgur.com/igTzVYr.png)

**4. Loss Calculation**

At this step, we have the final presentations $z_1, ..., z_4$.

![](https://i.imgur.com/TXElx1M.png)

Cosine Similarity is used to measure the similarity between representations.

![](https://i.imgur.com/QaMF4rg.png)

To calculate the loss over a batch, the paper uses the NT-Xent loss (Normalized Temperature-Scaled Cross-Entropy Loss). The loss of each pair of representations is calculated as:

![](https://i.imgur.com/G4Bwi1W.png)

And the Loss of a batch is the average over all the pairs:

![](https://i.imgur.com/ySjWvp1.png)

This concludes the training iteration of a batch.

**5. Fine-tune**
By far, the encoder has been trained to output representations. And now the model is ready to be fine-tuned to deal with downstream tasks.

The paper claims that accuracy of 76.5% on ImageNet can be achieved with  SimCLR using ResNet-50 (4x), and 85.8% if fine-tuned with 1% labels.

In the following sections, we introduce our implementations of SimCLR and the results.


## Our work

The goal of this blog is to show our effort in reproducing the paper "A Simple Framework for Contrastive Learning of Visual Representation". [Link to paper](https://arxiv.org/pdf/2002.05709.pdf) We reimplement SIMCLR using PyTorch based on the [official TensorFlow version](https://github.com/google-research/simclr.git). Moreover, as the requirement of the course, we reproduce the result in table 8 on the CIFAR10 dataset and get a nice visualization effect on trained image vectors. Moreover, we also extend our work to a new dataset RPLAN, and also achieves good visualization results. In general, our work can be divided into the following parts:

  - Reimplement the paper using _PyTorch_ on _Jupyter Notebook_.
  - Reproduce the result of _table 8_ in the paper using different training strategies, including finetuning and linear evaluation, by using the pretrained _ResNet(1X)_ and _ResNet(4X)_ models.
  - Extend to apply SIMCLR on the _RPLAN dataset_. The work includes applying transform on RPLAN images(so that they can fit in the model) and training on these images.
  - _Visualize_ the trained image vectors on CIFAR10 and RPLAN datasets by using _PCA_ and _SNE_, and analysis the pretraining performance of the model.

### Reproduction of the paper

This section mainly focuses on reproducing the table 8 result on CIFAR10 in the paper. We use the official pretrained checkpoint to pre-load the ResNet model before finetuning. To make our model fit the downstream classification task, we add a logistic regression layer at the end of the ResNet model. We train on the model using two different training strategies, finetuning and linear evaluation. Finetuning is just like the normal training process, the gradient passes through all the models and all parameters get updated after one backward. For linear evaluation, the parameter of the ResNet model is frozen while training, and only the parameter of the logistic regression layer gets updated. Two strategies share almost the same code, the only difference is mainly in the training and testing process. Two different versions of ResNet models are used, including ResNet50(1X) and ResNet50(4X), and we train the model for 500 epochs and compare the test accuracy with the result in the paper. We set the batch size as 64 for ResNet1X model and 32 for ResNet4X model. These are the maximum size we can use because if we increase them, an OOM error will occur. We run our codes in the CIFAR10 dataset during reproduction work but our code can also be applied to other image classification datasets, including STL10.

To transform the official TensorFlow checkpoint to the Pytorch checkpoint, we use the converter provided in this [repository]([tonylins/simclr-converter: A PyTorch converter for SimCLR checkpoints (github.com)](https://github.com/tonylins/simclr-converter)) and load the converted checkpoint during finetuning and linear evaluation.

#### Analysis of loss curve

##### Linear Evaluation

We use [weight & biase]([wandb · PyPI](https://pypi.org/project/wandb/)) to get the loss curve of the training process. As shown in the following figure. The X-axis refers to the epoch number, and the y-axis refers to the corresponding accuracy or loss and we apply log scaling on it.  During the linear evaluation, the model quickly converges and the loss almost reaches 0 in around 200 epochs. We also compare the convergence speed between ResNet1X and ResNet4X. The result shows that a larger pretraining model won't make the logistic regression layer converge faster.

[<img src="https://s1.ax1x.com/2022/04/14/L1bEJH.png" alt="L1bEJH.png" style="zoom:15%;" />](https://imgtu.com/i/L1bEJH)[<img src="https://s1.ax1x.com/2022/04/14/L1bAFe.png" alt="L1bAFe.png" style="zoom:15%;" />](https://imgtu.com/i/L1bAFe)[<img src="https://s1.ax1x.com/2022/04/14/L1bsfJ.png" alt="L1bsfJ.png" style="zoom:15%;" />](https://imgtu.com/i/L1bsfJ)[<img src="https://s1.ax1x.com/2022/04/14/L1b6p9.png" alt="L1b6p9.png" style="zoom:15%;" />](https://imgtu.com/i/L1b6p9)

##### Finetune

We also plot the learning curve of finetuning training phase and compare it with the learning curve while the model learns from scratch. The figure of loss and accuracy during training process is shown below. Different from linear evaluation, finetuning requires much more computation and takes approximately 30 minutes to run 1 epoch, so the x-axis now refers to the step number. Pretrained models converge much faster compared with the model trained from scratch and achieve much higher accuracy while training. It proves that the pretraining process does take effect.



<img src="https://cdn.discordapp.com/attachments/884910103428476989/962094184561533028/WB_Chart_4_8_2022_10_57_18_PM.png" alt="i1" style="zoom: 15%;" /><img src="https://cdn.discordapp.com/attachments/884910103428476989/962094184825765888/WB_Chart_4_8_2022_10_57_05_PM.png" alt="i2" style="zoom:15%;" /><img src="https://cdn.discordapp.com/attachments/884910103428476989/962094185064833054/WB_Chart_4_8_2022_10_57_32_PM.png" alt="i3" style="zoom:15%;" /><img src="https://cdn.discordapp.com/attachments/884910103428476989/962094185270358046/WB_Chart_4_8_2022_10_57_25_PM.png" alt="i4" style="zoom:15%;" />

#### Result comparison

In this section, we compare the performance of different training strategies and the same strategy with the performance in the paper. All the results are shown in the following table. We fail to run ResNet4X finetune because of a lack of computational resources. Our ResNet1X finetuning has almost the same performance compared with the original paper. However, after trying different training settings, our linear evaluation still can not achieve the same performance as the paper. After reading the paper and comparing our code with the official code, I think it might result from the small batch size we use. The author shows that a larger batch size over 512 can significantly increase the performance. But because of a lack of memory resources, that does not apply to us.

| Training Setup              | Note                          | Accuracy      |
| --------------------------- | ----------------------------- | ------------- |
| ResNet1X finetune           | loading pretrained checkpoint | 0.955(-0.022) |
| ResNet1X finetune           | learn from scratch            | 0.823(-0.154) |
| ResNet1X finetune           | in the original paper         | 0.977         |
| ResNet1X linear evaluation  | our implementation            | 0.852(-0.854) |
| RestNet1X linear evaluation | in the original paper         | 0.906         |
| ResNet4X linear evaluation  | our implementation            | 0.897(-0.056) |
| ResNet4X linear evaluation  | in the original paper         | 0.953         |

### Visualization
Is there an alternative approach to evaluate the performance of SimCLR or other contrastive learning methods? 
In this section, we project the CIFAR10 dataset and RPLAN dataset into the embedding space by SimCLR. And we use t-SNE to reduce the embedding space's dimensionality to visualize the embedding space. This kind of approach can give us a subjective view of the performance of contrastive learning methods.
#### Visualization on CIFAR10
The visulazation process consists of the following steps, and the detailed step by step process is included in our GitHub repository:
1. Project the CIFAR10 dataset into the embedding space by SimCLR.
2. Use t-SNE to reduce the dimensionality of the embedding space.
3. Visualize the embedding space.

We choose 4000 thousands of samples from the CIFAR10 testset to do the visualization. At first, we only associate the embedding with colorful dots in the 3D space.

<img src="https://cdn.discordapp.com/attachments/884910103428476989/964083401139306516/Webp.net-gifmaker.gif" alt="examples" width="600"/>

In this gif, different colors represent different classes. As we can see, the dots in the 3D space are denser in the center of the space. The dots in the 3D space are split very well. Different classes are separated into different sub space.

To better visualize the embedding space, we associate the embedding with the images in our space. We refer to the blog [How to visualize image feature vectors](https://hanna-shares.medium.com/how-to-visualize-image-feature-vectors-1e309d45f28f).

<img src="https://cdn.discordapp.com/attachments/884910103428476989/964089053177839616/unknown.png" alt="examples" width="600"/>
<img src="https://media.discordapp.net/attachments/884910103428476989/964089114951573534/unknown.png?width=1398&height=1302" alt="examples" width="600"/>

As you can see, each dot in the 2D space is an image. We can find some interesting findings in the 2D space.
1. Some classes are easily mixed in the space, like truck, automobile, and airplane.

<img src="https://media.discordapp.net/attachments/884910103428476989/964091046927671336/unknown.png?width=1876&height=1302" alt="examples" width="600"/>
2. Images on the dividing line between the two classes always have the same characteristics of both.
For example, birds on the dividing line between birds and deers are always ostriches.

<img src="https://media.discordapp.net/attachments/884910103428476989/964092583875854416/unknown.png?width=1384&height=1302" width="600"/>

#### Visualization on RPLAN Dataset
RPLAN dataset is a manually collected large-scale densely annotated dataset of floor plans from real residential buildings[From dataset discription](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html).

Here is an example in the RPLAN dataset:

<img src="http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/DeepLayout.png" alt="examples" width="600"/>

The main problem with the RPLAN dataset is that the images are not labeled. Therefore, we cannot use the linear evaluation to evaluate our model's performance on the RPLAN dataset. Therefore, we use the visualization method to project the image to 3D space to evaluate the performance subjectively.

The main process can be divided into four parts:
1. Change the binary images of the RPLAN dataset to color images for better visualization
2. Use the color images to pre-train the SimCLR model.
3. Use the pre-trained SimCLR model to project the RPLAN dataset into embedding space.
4. Use TensorBoard to visualize the embedding space.

In the first part, we use the [rplanpy](https://pypi.org/project/rplanpy/) library to read the RPLAN dataset and convert the binary images to color images. Here are two examples of this processing.

<img src="https://cdn.discordapp.com/attachments/884910103428476989/961703864225112074/unknown.png" alt="examples" width="600"/>

The next three parts are the same as the CIFAR10 dataset, the only difference is that we cannot get the label information. The final visualization results are shown in the images below.

<img src="https://media.discordapp.net/attachments/884557154902765572/964101556708454451/unknown.png?width=1310&height=1302" alt="examples" width="600"/>

We find that the neighbors in the space are very similar.
For example:

<img src="https://media.discordapp.net/attachments/884557154902765572/964107520903893042/Picture3.png?width=1404&height=1302" alt="examples" width="600"/>

<img src="https://media.discordapp.net/attachments/884557154902765572/964107533524561940/Picture1.png?width=1366&height=1302" alt="examples" width="600"/>


Therefore, SimCLR can also work well on the RPLAN dataset. In future work, SimCLR can be a useful tool to project the RPLAN dataset into embeddings for multipurpose applications.

 ## References
 [1] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.
 
 [2] Wu, Wenming, et al. "Data-driven interior plan generation for residential buildings." ACM Transactions on Graphics (TOG) 38.6 (2019): 1-12.
 
 [3] SimCLR PyTorch Implementation: https://github.com/Spijkervet/SimCLR
 
 [4] How to visualize image feature vectors: https://hanna-shares.medium.com/how-to-visualize-image-feature-vectors-1e309d45f28f
