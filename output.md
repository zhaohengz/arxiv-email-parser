# [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation ](https://arxiv.org/abs/1607.05368)

### Authors: Jey Han Lau, Timothy Baldwin 
### Categories: cs.CL 
### Comments: 1st Workshop on Representation Learning for NLP  
---
Recently, Le and Mikolov (2014) proposed doc2vec as an extension to word2vec
(Mikolov et al., 2013a) to learn document-level embeddings. Despite promising
results in the original paper, others have struggled to reproduce those
results. This paper presents a rigorous empirical evaluation of doc2vec over
two tasks. We compare doc2vec to two baselines and two state-of-the-art
document embedding methodologies. We found that doc2vec performs robustly when
using models trained on large external corpora, and can be further improved by
using pre-trained word embeddings. We also provide recommendations on
hyper-parameter settings for general purpose applications, and release source
code to induce document embeddings using our trained doc2vec models.
# [Discriminating between similar languages in Twitter using label propagation ](https://arxiv.org/abs/1607.05408)

### Authors: Will Radford, Matthias Galle 
### Categories: cs.CL 
---
Identifying the language of social media messages is an important first step
in linguistic processing. Existing models for Twitter focus on content
analysis, which is successful for dissimilar language pairs. We propose a label
propagation approach that takes the social graph of tweet authors into account
as well as content to better tease apart similar languages. This results in
state-of-the-art shared task performance of $76.63\%$, $1.4\%$ higher than the
top system.
# [A Supervised Authorship Attribution Framework for Bengali Language ](https://arxiv.org/abs/1607.05650)

### Authors: Shanta Phani, Shibamouli Lahiri and Arindam Biswas 
### Categories: cs.CL cs.DL 
### Comments: Submitted to the journal "International Journal of Computational Linguistics and Chinese Language Processing (IJCLCLP)"  
---
Authorship Attribution is a long-standing problem in Natural Language
Processing. Several statistical and computational methods have been used to
find a solution to this problem. In this paper, we have proposed methods to
deal with the authorship attribution problem in Bengali. More specifically, we
proposed a supervised framework consisting of lexical and shallow features, and
investigated the possibility of using topic-modeling-inspired features, to
classify documents according to their authors. We have created a corpus from
nearly all the literary works of three eminent Bengali authors, consisting of
3000 disjoint samples. Our models showed better performance than the
state-of-the-art, with more than 98% test accuracy for the shallow features,
and 100% test accuracy for the topic-based features.
# [Trainable Frontend For Robust and Far-Field Keyword Spotting ](https://arxiv.org/abs/1607.05666)

### Authors: Yuxuan Wang, Pascal Getreuer, Thad Hughes, Richard F. Lyon, Rif A. Saurous 
### Categories: cs.CL cs.NE  
---
Robust and far-field speech recognition is critical to enable true hands-free
communication. In far-field conditions, signals are attenuated due to distance.
To improve robustness to loudness variation, we introduce a novel frontend
called per-channel energy normalization (PCEN). The key ingredient of PCEN is
the use of an automatic gain control based dynamic compression to replace the
widely used static (such as log or root) compression. We evaluate PCEN on the
keyword spotting task. On our large rerecorded noisy and far-field eval sets,
we show that PCEN significantly improves recognition performance. Furthermore,
we model PCEN as neural network layers and optimize high-dimensional PCEN
parameters jointly with the keyword spotting acoustic model. The trained PCEN
frontend demonstrates significant further improvements without increasing model
complexity or inference-time cost.
# [Geometry-Informed Material Recognition ](https://arxiv.org/abs/1607.05338)

### Authors: Joseph DeGol, Mani Golparvar-Fard, Derek Hoiem 
### Categories: cs.CV 
### Comments: IEEE Conference on Computer Vision and Pattern Recognition 2016 (CVPR '16)  
---
Our goal is to recognize material categories using images and geometry
information. In many applications, such as construction management, coarse
geometry information is available. We investigate how 3D geometry (surface
normals, camera intrinsic and extrinsic parameters) can be used with 2D
features (texture and color) to improve material classification. We introduce a
new dataset, GeoMat, which is the first to provide both image and geometry data
in the form of: (i) training and testing patches that were extracted at
different scales and perspectives from real world examples of each material
category, and (ii) a large scale construction site scene that includes 160
images and over 800,000 hand labeled 3D points. Our results show that using 2D
and 3D features both jointly and independently to model materials improves
classification accuracy across multiple scales and viewing directions for both
material patches and images of a large scale construction site scene.
# [A Multi-task Deep Network for Person Re-identification ](https://arxiv.org/abs/1607.05369)

### Authors: Weihua Chen, Xiaotang Chen, Jianguo Zhang, Kaiqi Huang 
### Categories: cs.CV  
---
Person re-identification (RID) focuses on identifying people across different
scenes in video surveillance, which is usually formulated as either a binary
classification task or a ranking task in current person RID approaches. To the
best of our knowledge, none of existing work treats the two tasks
simultaneously. In this paper, we take both tasks into account and propose a
multi-task deep network (MTDnet) to jointly optimize the two tasks
simultaneously for person RID. We show that our proposed architecture
significantly boosts the performance. Furthermore, a good performance of any
deep architectures requires a sufficient training set which is usually not met
in person RID. To cope with this situation, we further extend the MTDnet and
propose a cross-domain architecture that is capable of using an auxiliary set
to assist training on small target sets. In the experiments, our approach
significantly outperforms previous state-of-the-art methods on almost all the
datasets, which clearly demonstrates the effectiveness of the proposed
approach.
# [Binary Hashing with Semidefinite Relaxation and Augmented Lagrangian ](https://arxiv.org/abs/1607.05396)

### Authors: Thanh-Toan Do, Anh-Dzung Doan, Duc-Thanh Nguyen, Ngai-Man Cheung 
### Categories: cs.CV 
### Comments: Appearing in European Conference on Computer Vision (ECCV) 2016  
---
This paper proposes two approaches for inferencing binary codes in two-step
(supervised, unsupervised) hashing. We first introduce an unified formulation
for both supervised and unsupervised hashing. Then, we cast the learning of one
bit as a Binary Quadratic Problem (BQP). We propose two approaches to solve
BQP. In the first approach, we relax BQP as a semidefinite programming problem
which its global optimum can be achieved. We theoretically prove that the
objective value of the binary solution achieved by this approach is well
bounded. In the second approach, we propose an augmented Lagrangian based
approach to solve BQP directly without relaxing the binary constraint.
Experimental results on three benchmark datasets show that our proposed methods
compare favorably with the state of the art.
# [Training Skinny Deep Neural Networks with Iterative Hard Thresholding Methods ](https://arxiv.org/abs/1607.05423)

### Authors: Xiaojie Jin, Xiaotong Yuan, Jiashi Feng, Shuicheng Yan 
### Categories: cs.CV  
---
Deep neural networks have achieved remarkable success in a wide range of
practical problems. However, due to the inherent large parameter space, deep
models are notoriously prone to overfitting and difficult to be deployed in
portable devices with limited memory. In this paper, we propose an iterative
hard thresholding (IHT) approach to train Skinny Deep Neural Networks (SDNNs).
An SDNN has much fewer parameters yet can achieve competitive or even better
performance than its full CNN counterpart. More concretely, the IHT approach
trains an SDNN through following two alternative phases: (I) perform hard
thresholding to drop connections with small activations and fine-tune the other
significant filters; (II)~re-activate the frozen connections and train the
entire network to improve its overall discriminative capability. We verify the
superiority of SDNNs in terms of efficiency and classification performance on
four benchmark object recognition datasets, including CIFAR-10, CIFAR-100,
MNIST and ImageNet. Experimental results clearly demonstrate that IHT can be
applied for training SDNN based on various CNN architectures such as NIN and
AlexNet.
# [Trunk-Branch Ensemble Convolutional Neural Networks for Video-based Face Recognition ](https://arxiv.org/abs/1607.05427)

### Authors: Changxing Ding and Dacheng Tao 
### Categories: cs.CV  
---
Human faces in surveillance videos often suffer from severe image blur,
dramatic pose variations, and occlusion. In this paper, we propose a
comprehensive framework based on Convolutional Neural Networks (CNN) to
overcome challenges in video-based face recognition (VFR). First, to learn
blur-robust face representations, we artificially blur training data composed
of clear still images to account for a shortfall in real-world video training
data. Using training data composed of both still images and artificially
blurred data, CNN is encouraged to learn blur-insensitive features
automatically. Second, to enhance robustness of CNN features to pose variations
and occlusion, we propose a Trunk-Branch Ensemble CNN model (TBE-CNN), which
extracts complementary information from holistic face images and patches
cropped around facial components. TBE-CNN is an end-to-end model that extracts
features efficiently by sharing the low- and middle-level convolutional layers
between the trunk and branch networks. Third, to further promote the
discriminative power of the representations learnt by TBE-CNN, we propose an
improved triplet loss function. Systematic experiments justify the
effectiveness of the proposed techniques. Most impressively, TBE-CNN achieves
state-of-the-art performance on three popular video face databases: PaSC, COX
Face, and YouTube Faces.
# [Collaborative Layer-wise Discriminative Learning in Deep Neural Networks ](https://arxiv.org/abs/1607.05440)

### Authors: Xiaojie Jin, Yunpeng Chen, Jian Dong, Jiashi Feng, Shuicheng Yan 
### Categories: cs.CV 
### Comments: To appear in ECCV 2016. Maybe subject to minor changes before camera-ready version  
---
Intermediate features at different layers of a deep neural network are known
to be discriminative for visual patterns of different complexities. However,
most existing works ignore such cross-layer heterogeneities when classifying
samples of different complexities. For example, if a training sample has
already been correctly classified at a specific layer with high confidence, we
argue that it is unnecessary to enforce rest layers to classify this sample
correctly and a better strategy is to encourage those layers to focus on other
samples.
  In this paper, we propose a layer-wise discriminative learning method to
enhance the discriminative capability of a deep network by allowing its layers
to work collaboratively for classification. Towards this target, we introduce
multiple classifiers on top of multiple layers. Each classifier not only tries
to correctly classify the features from its input layer, but also coordinates
with other classifiers to jointly maximize the final classification
performance. Guided by the other companion classifiers, each classifier learns
to concentrate on certain training examples and boosts the overall performance.
Allowing for end-to-end training, our method can be conveniently embedded into
state-of-the-art deep networks. Experiments with multiple popular deep
networks, including Network in Network, GoogLeNet and VGGNet, on scale-various
object classification benchmarks, including CIFAR100, MNIST and ImageNet, and
scene classification benchmarks, including MIT67, SUN397 and Places205,
demonstrate the effectiveness of our method. In addition, we also analyze the
relationship between the proposed method and classical conditional random
fields models.
# [On Differentiating Parameterized Argmin and Argmax Problems with Application to Bi-level Optimization ](https://arxiv.org/abs/1607.05447)

### Authors: Stephen Gould and Basura Fernando and Anoop Cherian and Peter Anderson and Rodrigo Santa Cruz and Edison Guo 
### Categories: cs.CV math.OC 
### Comments: 16 pages, 6 figures  
---
Some recent works in machine learning and computer vision involve the
solution of a bi-level optimization problem. Here the solution of a
parameterized lower-level problem binds variables that appear in the objective
of an upper-level problem. The lower-level problem typically appears as an
argmin or argmax optimization problem. Many techniques have been proposed to
solve bi-level optimization problems, including gradient descent, which is
popular with current end-to-end learning approaches. In this technical report
we collect some results on differentiating argmin and argmax optimization
problems with and without constraints and provide some insightful motivating
examples.
# [Supervised Transformer Network for Efficient Face Detection ](https://arxiv.org/abs/1607.05477)

### Authors: Dong Chen, Gang Hua, Fang Wen, Jian Sun 
### Categories: cs.CV  
---
Large pose variations remain to be a challenge that confronts real-word face
detection. We propose a new cascaded Convolutional Neural Network, dubbed the
name Supervised Transformer Network, to address this challenge. The first stage
is a multi-task Region Proposal Network (RPN), which simultaneously predicts
candidate face regions along with associated facial landmarks. The candidate
regions are then warped by mapping the detected facial landmarks to their
canonical positions to better normalize the face patterns. The second stage,
which is a RCNN, then verifies if the warped candidate regions are valid faces
or not. We conduct end-to-end learning of the cascaded network, including
optimizing the canonical positions of the facial landmarks. This supervised
learning of the transformations automatically selects the best scale to
differentiate face/non-face patterns. By combining feature maps from both
stages of the network, we achieve state-of-the-art detection accuracies on
several public benchmarks. For real-time performance, we run the cascaded
network only on regions of interests produced from a boosting cascade face
detector. Our detector runs at 30 FPS on a single CPU core for a VGA-resolution
image.
# [Dendritic Spine Shape Analysis: A Clustering Perspective ](https://arxiv.org/abs/1607.05523)

### Authors: Muhammad Usman Ghani, Ertunc Erdil, Sumeyra Demir Kanik, Ali Ozgur Argunsah, Anna Felicity Hobbiss, Inbal Israely, Devrim Unay, Tolga Tasdizen, Mujdat Cetin 
### Categories: cs.CV 
### Comments: Accepted for BioImageComputing workshop at ECCV 2016  
---
Functional properties of neurons are strongly coupled with their morphology.
Changes in neuronal activity alter morphological characteristics of dendritic
spines. First step towards understanding the structure-function relationship is
to group spines into main spine classes reported in the literature. Shape
analysis of dendritic spines can help neuroscientists understand the underlying
relationships. Due to unavailability of reliable automated tools, this analysis
is currently performed manually which is a time-intensive and subjective task.
Several studies on spine shape classification have been reported in the
literature, however, there is an on-going debate on whether distinct spine
shape classes exist or whether spines should be modeled through a continuum of
shape variations. Another challenge is the subjectivity and bias that is
introduced due to the supervised nature of classification approaches. In this
paper, we aim to address these issues by presenting a clustering perspective.
In this context, clustering may serve both confirmation of known patterns and
discovery of new ones. We perform cluster analysis on two-photon microscopic
images of spines using morphological, shape, and appearance based features and
gain insights into the spine shape analysis problem. We use histogram of
oriented gradients (HOG), disjunctive normal shape models (DNSM), morphological
features, and intensity profile based features for cluster analysis. We use
x-means to perform cluster analysis that selects the number of clusters
automatically using the Bayesian information criterion (BIC). For all features,
this analysis produces 4 clusters and we observe the formation of at least one
cluster consisting of spines which are difficult to be assigned to a known
class. This observation supports the argument of intermediate shape types.
# [Dual Purpose Hashing ](https://arxiv.org/abs/1607.05529)

### Authors: Haomiao Liu, Ruiping Wang, Shiguang Shan, Xilin Chen 
### Categories: cs.CV 
### Comments: With supplementary materials added to the end  
---
Recent years have seen more and more demand for a unified framework to
address multiple realistic image retrieval tasks concerning both category and
attributes. Considering the scale of modern datasets, hashing is favorable for
its low complexity. However, most existing hashing methods are designed to
preserve one single kind of similarity, thus improper for dealing with the
different tasks simultaneously. To overcome this limitation, we propose a new
hashing method, named Dual Purpose Hashing (DPH), which jointly preserves the
category and attribute similarities by exploiting the Convolutional Neural
Network (CNN) models to hierarchically capture the correlations between
category and attributes. Since images with both category and attribute labels
are scarce, our method is designed to take the abundant partially labelled
images on the Internet as training inputs. With such a framework, the binary
codes of new-coming images can be readily obtained by quantizing the network
outputs of a binary-like layer, and the attributes can be recovered from the
codes easily. Experiments on two large-scale datasets show that our dual
purpose hash codes can achieve comparable or even better performance than those
state-of-the-art methods specifically designed for each individual retrieval
task, while being more compact than the compared methods.
# [A Local-Global Approach to Semantic Segmentation in Aerial Images ](https://arxiv.org/abs/1607.05620)

### Authors: Alina Elena Marcu 
### Categories: cs.CV 
### Comments: 50 pages, 18 figures. Master's Thesis, University Politehnica of Bucharest  
---
Aerial images are often taken under poor lighting conditions and contain low
resolution objects, many times occluded by other objects. In this domain,
visual context could be of great help, but there are still very few papers that
consider context in aerial image understanding and still remains an open
problem in computer vision. We propose a dual-stream deep neural network that
processes information along two independent pathways. Our model learns to
combine local and global appearance in a complementary way, such that together
form a powerful classifier. We test our dual-stream network on the task of
buildings segmentation in aerial images and obtain state-of-the-art results on
the Massachusetts Buildings Dataset. We study the relative importance of local
appearance versus the larger scene, as well as their performance in combination
on three new buildings datasets. We clearly demonstrate the effectiveness of
visual context in conjunction with deep neural networks for aerial image
understanding.
# [Information-theoretical label embeddings for large-scale image classification ](https://arxiv.org/abs/1607.05691)

### Authors: Fran\c{c}ois Chollet 
### Categories: cs.CV cs.LG stat.ML  
---
We present a method for training multi-label, massively multi-class image
classification models, that is faster and more accurate than supervision via a
sigmoid cross-entropy loss (logistic regression). Our method consists in
embedding high-dimensional sparse labels onto a lower-dimensional dense sphere
of unit-normed vectors, and treating the classification problem as a cosine
proximity regression problem on this sphere. We test our method on a dataset of
300 million high-resolution images with 17,000 labels, where it yields
considerably faster convergence, as well as a 7% higher mean average precision
compared to logistic regression.
# [FusionNet: 3D Object Classification Using Multiple Data Representations ](https://arxiv.org/abs/1607.05695)

### Authors: Vishakh Hegde, Reza Zadeh 
### Categories: cs.CV  
---
High-quality 3D object recognition is an important component of many vision
and robotics systems. We tackle the object recognition problem using two data
representations, to achieve leading results on the Princeton ModelNet
challenge. The two representations: 1. Volumetric representation: the 3D object
is discretized spatially as binary voxels - $1$ if the voxel is occupied and
$0$ otherwise. 2. Pixel representation: the 3D object is represented as a set
of projected 2D pixel images. Current leading submissions to the ModelNet
Challenge use Convolutional Neural Networks (CNNs) on pixel representations.
However, we diverge from this trend and additionally, use Volumetric CNNs to
bridge the gap between the efficiency of the above two representations. We
combine both representations and exploit them to learn new features, which
yield a significantly better classifier than using either of the
representations in isolation. To do this, we introduce new Volumetric CNN
(V-CNN) architectures.
# [A Semiparametric Model for Bayesian Reader Identification ](https://arxiv.org/abs/1607.05271)

### Authors: Ahmed Abdelwahab, Reinhold Kliegl and Niels Landwehr 
### Categories: cs.LG  
---
We study the problem of identifying individuals based on their characteristic
gaze patterns during reading of arbitrary text. The motivation for this problem
is an unobtrusive biometric setting in which a user is observed during access
to a document, but no specific challenge protocol requiring the user's time and
attention is carried out. Existing models of individual differences in gaze
control during reading are either based on simple aggregate features of eye
movements, or rely on parametric density models to describe, for instance,
saccade amplitudes or word fixation durations. We develop flexible
semiparametric models of eye movements during reading in which densities are
inferred under a Gaussian process prior centered at a parametric distribution
family that is expected to approximate the true distribution well. An empirical
study on reading data from 251 individuals shows significant improvements over
the state of the art.
# [A Novel Information Theoretic Framework for Finding Semantic Similarity in WordNet ](https://arxiv.org/abs/1607.05422)

### Authors: Abhijit Adhikari, Shivang Singh, Deepjyoti Mondal, Biswanath Dutta, Animesh Dutta 
### Categories: cs.IR cs.CL  
---
Information content (IC) based measures for finding semantic similarity is
gaining preferences day by day. Semantics of concepts can be highly
characterized by information theory. The conventional way for calculating IC is
based on the probability of appearance of concepts in corpora. Due to data
sparseness and corpora dependency issues of those conventional approaches, a
new corpora independent intrinsic IC calculation measure has evolved. In this
paper, we mainly focus on such intrinsic IC model and several topological
aspects of the underlying ontology. Accuracy of intrinsic IC calculation and
semantic similarity measure rely on these aspects deeply. Based on these
analysis we propose an information theoretic framework which comprises an
intrinsic IC calculator and a semantic similarity model. Our approach is
compared with state of the art semantic similarity measures based on corpora
dependent IC calculation as well as intrinsic IC based methods using several
benchmark data set. We also compare our model with the related Edge based,
Feature based and Distributional approaches. Experimental results show that our
intrinsic IC model gives high correlation value when applied to different
semantic similarity models. Our proposed semantic similarity model also
achieves significant results when embedded with some state of the art IC models
including ours.
# [Generating Images Part by Part with Composite Generative Adversarial Networks ](https://arxiv.org/abs/1607.05387)

### Authors: Hanock Kwak, Byoung-Tak Zhang 
### Categories: cs.AI cs.CV cs.LG 
### Comments: IJCAI 2016 Workshop on Deep Learning for Artificial Intelligence  
---
Image generation remains a fundamental problem in artificial intelligence in
general and deep learning in specific. The generative adversarial network (GAN)
was successful in generating high quality samples of natural images. We propose
a model called composite generative adversarial network, that reveals the
complex structure of images with multiple generators in which each generator
generates some part of the image. Those parts are combined by alpha blending
process to create a new single image. It can generate, for example, background
and face sequentially with two generators, after training on face dataset.
Training was done in an unsupervised way without any labels about what each
generator should generate. We found possibilities of learning the structure by
using this generative model empirically.
# [Runtime Configurable Deep Neural Networks for Energy-Accuracy Trade-off ](https://arxiv.org/abs/1607.05418)

### Authors: Hokchhay Tann, Soheil Hashemi, R. Iris Bahar, Sherief Reda 
### Categories: cs.NE cs.CV 
---
We present a novel dynamic configuration technique for deep neural networks
that permits step-wise energy-accuracy trade-offs during runtime. Our
configuration technique adjusts the number of channels in the network
dynamically depending on response time, power, and accuracy targets. To enable
this dynamic configuration technique, we co-design a new training algorithm,
where the network is incrementally trained such that the weights in channels
trained in earlier steps are fixed. Our technique provides the flexibility of
multiple networks while storing and utilizing one set of weights. We evaluate
our techniques using both an ASIC-based hardware accelerator as well as a
low-power embedded GPGPU and show that our approach leads to only a small or
negligible loss in the final network accuracy. We analyze the performance of
our proposed methodology using three well-known networks for MNIST, CIFAR-10,
and SVHN datasets, and we show that we are able to achieve up to 95\% energy
reduction with less than 1\% accuracy loss across the three benchmarks. In
addition, compared to prior work on dynamic network reconfiguration, we show
that our approach leads to approximately 50\% savings in storage requirements,
while achieving similar accuracy.
# [Multidimensional Dynamic Pricing for Welfare Maximization ](https://arxiv.org/abs/1607.05397)

### Authors: Aaron Roth, Aleksandrs Slivkins, Jonathan Ullman, Zhiwei Steven Wu 
### Categories: cs.DS cs.GT cs.LG  
---
We study the problem of a seller dynamically pricing $d$ distinct types of
goods, when faced with the online arrival of buyers drawn independently from an
unknown distribution. The seller observes only the bundle of goods purchased at
each day, but nothing else about the buyer's valuation function. When buyers
have strongly concave, Holder continuous valuation functions, we give a pricing
scheme that finds a pricing that optimizes welfare (including the seller's cost
of production) in time and number of rounds that are polynomial in $d$ and the
accuracy parameter. We are able to do this despite the fact that (i) welfare is
a non-concave function of the prices, and (ii) the welfare is not observable to
the seller. We also extend our results to a limited-supply setting.
