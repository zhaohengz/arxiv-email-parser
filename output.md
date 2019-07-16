# Towards Universal Dialogue Act Tagging for Task-Oriented Dialogues 
Machine learning approaches for building task-oriented dialogue systems
require large conversational datasets with labels to train on. We are
interested in building task-oriented dialogue systems from human-human
conversations, which may be available in ample amounts in existing customer
care center logs or can be collected from crowd workers. Annotating these
datasets can be prohibitively expensive. Recently multiple annotated
task-oriented human-machine dialogue datasets have been released, however their
annotation schema varies across different collections, even for well-defined
categories such as dialogue acts (DAs). We propose a Universal DA schema for
task-oriented dialogues and align existing annotated datasets with our schema.
Our aim is to train a Universal DA tagger (U-DAT) for task-oriented dialogues
and use it for tagging human-human conversations. We investigate multiple
datasets, propose manual and automated approaches for aligning the different
schema, and present results on a target corpus of human-human dialogues. In
unsupervised learning experiments we achieve an F1 score of 54.1% on system
turns in human-human dialogues. In a semi-supervised setup, the F1 score
increases to 57.7% which would otherwise require at least 1.7K manually
annotated turns. For new domains, we show further improvements when unlabeled
or labeled target domain data is available.
# BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer 
An important yet rarely tackled problem in dialogue state tracking (DST) is
scalability for dynamic ontology (e.g., movie, restaurant) and unseen slot
values. We focus on a specific condition, where the ontology is unknown to the
state tracker, but the target slot value (except for none and dontcare),
possibly unseen during training, can be found as word segment in the dialogue
context. Prior approaches often rely on candidate generation from n-gram
enumeration or slot tagger outputs, which can be inefficient or suffer from
error propagation. We propose BERT-DST, an end-to-end dialogue state tracker
which directly extracts slot values from the dialogue context. We use BERT as
dialogue context encoder whose contextualized language representations are
suitable for scalable DST to identify slot values from their semantic context.
Furthermore, we employ encoder parameter sharing across all slots with two
advantages: (1) Number of parameters does not grow linearly with the ontology.
(2) Language representation knowledge can be transferred among slots. Empirical
evaluation shows BERT-DST with cross-slot parameter sharing outperforms prior
work on the benchmark scalable DST datasets Sim-M and Sim-R, and achieves
competitive performance on the standard DSTC2 and WOZ 2.0 datasets.
# Exploiting Out-of-Domain Parallel Data through Multilingual Transfer Learning for Low-Resource Neural Machine Translation 
This paper proposes a novel multilingual multistage fine-tuning approach for
low-resource neural machine translation (NMT), taking a challenging
Japanese--Russian pair for benchmarking. Although there are many solutions for
low-resource scenarios, such as multilingual NMT and back-translation, we have
empirically confirmed their limited success when restricted to in-domain data.
We therefore propose to exploit out-of-domain data through transfer learning,
by using it to first train a multilingual NMT model followed by multistage
fine-tuning on in-domain parallel and back-translated pseudo-parallel data. Our
approach, which combines domain adaptation, multilingualism, and
back-translation, helps improve the translation quality by more than 3.7 BLEU
points, over a strong baseline, for this extremely low-resource scenario.
# Improved low-resource Somali speech recognition by semi-supervised acoustic and language model training 
We present improvements in automatic speech recognition (ASR) for Somali, a
currently extremely under-resourced language. This forms part of a continuing
United Nations (UN) effort to employ ASR-based keyword spotting systems to
support humanitarian relief programmes in rural Africa. Using just 1.57 hours
of annotated speech data as a seed corpus, we increase the pool of training
data by applying semi-supervised training to 17.55 hours of untranscribed
speech. We make use of factorised time-delay neural networks (TDNN-F) for
acoustic modelling, since these have recently been shown to be effective in
resource-scarce situations. Three semi-supervised training passes were
performed, where the decoded output from each pass was used for acoustic model
training in the subsequent pass. The automatic transcriptions from the best
performing pass were used for language model augmentation. To ensure the
quality of automatic transcriptions, decoder confidence is used as a threshold.
The acoustic and language models obtained from the semi-supervised approach
show significant improvement in terms of WER and perplexity compared to the
baseline. Incorporating the automatically generated transcriptions yields a
6.55\% improvement in language model perplexity. The use of 17.55 hour of
Somali acoustic data in semi-supervised training shows an improvement of 7.74\%
relative over the baseline.
# Short Text Conversation Based on Deep Neural Network and Analysis on Evaluation Measures 
With the development of Natural Language Processing, Automatic
question-answering system such as Waston, Siri, Alexa, has become one of the
most important NLP applications. Nowadays, enterprises try to build automatic
custom service chatbots to save human resources and provide a 24-hour customer
service. Evaluation of chatbots currently relied greatly on human annotation
which cost a plenty of time. Thus, has initiated a new Short Text Conversation
subtask called Dialogue Quality (DQ) and Nugget Detection (ND) which aim to
automatically evaluate dialogues generated by chatbots. In this paper, we solve
the DQ and ND subtasks by deep neural network. We proposed two models for both
DQ and ND subtasks which is constructed by hierarchical structure: embedding
layer, utterance layer, context layer and memory layer, to hierarchical learn
dialogue representation from word level, sentence level, context level to long
range context level. Furthermore, we apply gating and attention mechanism at
utterance layer and context layer to improve the performance. We also tried
BERT to replace embedding layer and utterance layer as sentence representation.
The result shows that BERT produced a better utterance representation than
multi-stack CNN for both DQ and ND subtasks and outperform other models
proposed by other researches. The evaluation measures are proposed by , that
is, NMD, RSNOD for DQ and JSD, RNSS for ND, which is not traditional evaluation
measures such as accuracy, precision, recall and f1-score. Thus, we have done a
series of experiments by using traditional evaluation measures and analyze the
performance and error.
# ANETAC: Arabic Named Entity Transliteration and Classification Dataset 
In this paper, we make freely accessible ANETAC our English-Arabic named
entity transliteration and classification dataset that we built from freely
available parallel translation corpora. The dataset contains 79,924 instances,
each instance is a triplet (e, a, c), where e is the English named entity, a is
its Arabic transliteration and c is its class that can be either a Person, a
Location, or an Organization. The ANETAC dataset is mainly aimed for the
researchers that are working on Arabic named entity transliteration, but it can
also be used for named entity classification purposes.
# Best Practices for Learning Domain-Specific Cross-Lingual Embeddings 
Cross-lingual embeddings aim to represent words in multiple languages in a
shared vector space by capturing semantic similarities across languages. They
are a crucial component for scaling tasks to multiple languages by transferring
knowledge from languages with rich resources to low-resource languages. A
common approach to learning cross-lingual embeddings is to train monolingual
embeddings separately for each language and learn a linear projection from the
monolingual spaces into a shared space, where the mapping relies on a small
seed dictionary. While there are high-quality generic seed dictionaries and
pre-trained cross-lingual embeddings available for many language pairs, there
is little research on how they perform on specialised tasks. In this paper, we
investigate the best practices for constructing the seed dictionary for a
specific domain. We evaluate the embeddings on the sequence labelling task of
Curriculum Vitae parsing and show that the size of a bilingual dictionary, the
frequency of the dictionary words in the domain corpora and the source of data
(task-specific vs generic) influence the performance. We also show that the
less training data is available in the low-resource language, the more the
construction of the bilingual dictionary matters, and demonstrate that some of
the choices are crucial in the zero-shot transfer learning case.
# Exploring difference in public perceptions on HPV vaccine between gender groups from Twitter using deep learning 
In this study, we proposed a convolutional neural network model for gender
prediction using English Twitter text as input. Ensemble of proposed model
achieved an accuracy at 0.8237 on gender prediction and compared favorably with
the state-of-the-art performance in a recent author profiling task. We further
leveraged the trained models to predict the gender labels from an HPV vaccine
related corpus and identified gender difference in public perceptions regarding
HPV vaccine. The findings are largely consistent with previous survey-based
studies.
# Applying a Pre-trained Language Model to Spanish Twitter Humor Prediction 
Our entry into the HAHA 2019 Challenge placed $3^{rd}$ in the classification
task and $2^{nd}$ in the regression task. We describe our system and
innovations, as well as comparing our results to a Naive Bayes baseline. A
large Twitter based corpus allowed us to train a language model from scratch
focused on Spanish and transfer that knowledge to our competition model. To
overcome the inherent errors in some labels we reduce our class confidence with
label smoothing in the loss function. All the code for our project is included
in a GitHub repository for easy reference and to enable replication by others.
# Evolutionary Algorithm for Sinhala to English Translation 
Machine Translation (MT) is an area in natural language processing, which
focus on translating from one language to another. Many approaches ranging from
statistical methods to deep learning approaches are used in order to achieve
MT. However, these methods either require a large number of data or a clear
understanding about the language. Sinhala language has less digital text which
could be used to train a deep neural network. Furthermore, Sinhala has complex
rules therefore, it is harder to create statistical rules in order to apply
statistical methods in MT. This research focuses on Sinhala to English
translation using an Evolutionary Algorithm (EA). EA is used to identifying the
correct meaning of Sinhala text and to translate it to English. The Sinhala
text is passed to identify the meaning in order to get the correct meaning of
the sentence. With the use of the EA the translation is carried out. The
translated text is passed on to grammatically correct the sentence. This has
shown to achieve accurate results.
# Joint Lifelong Topic Model and Manifold Ranking for Document Summarization 
Due to the manifold ranking method has a significant effect on the ranking of
unknown data based on known data by using a weighted network, many researchers
use the manifold ranking method to solve the document summarization task.
However, their models only consider the original features but ignore the
semantic features of sentences when they construct the weighted networks for
the manifold ranking method. To solve this problem, we proposed two improved
models based on the manifold ranking method. One is combining the topic model
and manifold ranking method (JTMMR) to solve the document summarization task.
This model not only uses the original feature, but also uses the semantic
feature to represent the document, which can improve the accuracy of the
manifold ranking method. The other one is combining the lifelong topic model
and manifold ranking method (JLTMMR). On the basis of the JTMMR, this model
adds the constraint of knowledge to improve the quality of the topic. At the
same time, we also add the constraint of the relationship between documents to
dig out a better document semantic features. The JTMMR model can improve the
effect of the manifold ranking method by using the better semantic feature.
Experiments show that our models can achieve a better result than other
baseline models for multi-document summarization task. At the same time, our
models also have a good performance on the single document summarization task.
After combining with a few basic surface features, our model significantly
outperforms some model based on deep learning in recent years. After that, we
also do an exploring work for lifelong machine learning by analyzing the effect
of adding feedback. Experiments show that the effect of adding feedback to our
model is significant.
# Graph based Neural Networks for Event Factuality Prediction using Syntactic and Semantic Structures 
Event factuality prediction (EFP) is the task of assessing the degree to
which an event mentioned in a sentence has happened. For this task, both
syntactic and semantic information are crucial to identify the important
context words. The previous work for EFP has only combined these information in
a simple way that cannot fully exploit their coordination. In this work, we
introduce a novel graph-based neural network for EFP that can integrate the
semantic and syntactic information more effectively. Our experiments
demonstrate the advantage of the proposed model for EFP.
# Zero-Shot Open Entity Typing as Type-Compatible Grounding 
The problem of entity-typing has been studied predominantly in supervised
learning fashion, mostly with task-specific annotations (for coarse types) and
sometimes with distant supervision (for fine types). While such approaches have
strong performance within datasets, they often lack the flexibility to transfer
across text genres and to generalize to new type taxonomies. In this work we
propose a zero-shot entity typing approach that requires no annotated data and
can flexibly identify newly defined types. Given a type taxonomy defined as
Boolean functions of FREEBASE "types", we ground a given mention to a set of
type-compatible Wikipedia entries and then infer the target mention's types
using an inference algorithm that makes use of the types of these entries. We
evaluate our system on a broad range of datasets, including standard
fine-grained and coarse-grained entity typing datasets, and also a dataset in
the biological domain. Our system is shown to be competitive with
state-of-the-art supervised NER systems and outperforms them on out-of-domain
datasets. We also show that our system significantly outperforms other
zero-shot fine typing systems.
# Improving Cross-Domain Performance for Relation Extraction via Dependency Prediction and Information Flow Control 
Relation Extraction (RE) is one of the fundamental tasks in Information
Extraction and Natural Language Processing. Dependency trees have been shown to
be a very useful source of information for this task. The current deep learning
models for relation extraction has mainly exploited this dependency information
by guiding their computation along the structures of the dependency trees. One
potential problem with this approach is it might prevent the models from
capturing important context information beyond syntactic structures and cause
the poor cross-domain generalization. This paper introduces a novel method to
use dependency trees in RE for deep learning models that jointly predicts
dependency and semantics relations. We also propose a new mechanism to control
the information flow in the model based on the input entity mentions. Our
extensive experiments on benchmark datasets show that the proposed model
outperforms the existing methods for RE significantly.
# NIESR: Nuisance Invariant End-to-end Speech Recognition 
Deep neural network models for speech recognition have achieved great success
recently, but they can learn incorrect associations between the target and
nuisance factors of speech (e.g., speaker identities, background noise, etc.),
which can lead to overfitting. While several methods have been proposed to
tackle this problem, existing methods incorporate additional information about
nuisance factors during training to develop invariant models. However,
enumeration of all possible nuisance factors in speech data and the collection
of their annotations is difficult and expensive. We present a robust training
scheme for end-to-end speech recognition that adopts an unsupervised
adversarial invariance induction framework to separate out essential factors
for speech-recognition from nuisances without using any supplementary labels
besides the transcriptions. Experiments show that the speech recognition model
trained with the proposed training scheme achieves relative improvements of
5.48% on WSJ0, 6.16% on CHiME3, and 6.61% on TIMIT dataset over the base model.
Additionally, the proposed method achieves a relative improvement of 14.44% on
the combined WSJ0+CHiME3 dataset.
# A Natural Language Corpus of Common Grounding under Continuous and Partially-Observable Context 
Common grounding is the process of creating, repairing and updating mutual
understandings, which is a critical aspect of sophisticated human
communication. However, traditional dialogue systems have limited capability of
establishing common ground, and we also lack task formulations which introduce
natural difficulty in terms of common grounding while enabling easy evaluation
and analysis of complex models. In this paper, we propose a minimal dialogue
task which requires advanced skills of common grounding under continuous and
partially-observable context. Based on this task formulation, we collected a
largescale dataset of 6,760 dialogues which fulfills essential requirements of
natural language corpora. Our analysis of the dataset revealed important
phenomena related to common grounding that need to be considered. Finally, we
evaluate and analyze baseline neural models on a simple subtask that requires
recognition of the created common ground. We show that simple baseline models
perform decently but leave room for further improvement. Overall, we show that
our proposed task will be a fundamental testbed where we can train, evaluate,
and analyze dialogue system's ability for sophisticated common grounding.
# Correct-and-Memorize: Learning to Translate from Interactive Revisions 
State-of-the-art machine translation models are still not on par with human
translators. Previous work takes human interactions into the neural machine
translation process to obtain improved results in target languages. However,
not all model-translation errors are equal -- some are critical while others
are minor. In the meanwhile, the same translation mistakes occur repeatedly in
a similar context. To solve both issues, we propose CAMIT, a novel method for
translating in an interactive environment. Our proposed method works with
critical revision instructions, therefore allows human to correct arbitrary
words in model-translated sentences. In addition, CAMIT learns from and softly
memorizes revision actions based on the context, alleviating the issue of
repeating mistakes. Experiments in both ideal and real interactive translation
settings demonstrate that our proposed \method enhances machine translation
results significantly while requires fewer revision instructions from human
compared to previous methods.
# Searching for Effective Neural Extractive Summarization: What Works and What's Next 
The recent years have seen remarkable success in the use of deep neural
networks on text summarization.
  However, there is no clear understanding of \textit{why} they perform so
well, or \textit{how} they might be improved.
  In this paper, we seek to better understand how neural extractive
summarization systems could benefit from different types of model
architectures, transferable knowledge and
  learning schemas. Additionally, we find an effective way to improve current
frameworks and achieve the state-of-the-art result on CNN/DailyMail by a large
margin based on our
  observations and analyses. Hopefully, our work could provide more clues for
future research on extractive summarization.
# Early Discovery of Emerging Entities in Microblogs 
Keeping up to date on emerging entities that appear every day is
indispensable for various applications, such as social-trend analysis and
marketing research. Previous studies have attempted to detect unseen entities
that are not registered in a particular knowledge base as emerging entities and
consequently find non-emerging entities since the absence of entities in
knowledge bases does not guarantee their emergence. We therefore introduce a
novel task of discovering truly emerging entities when they have just been
introduced to the public through microblogs and propose an effective method
based on time-sensitive distant supervision, which exploits distinctive
early-stage contexts of emerging entities. Experimental results with a
large-scale Twitter archive show that the proposed method achieves 83.2%
precision of the top 500 discovered emerging entities, which outperforms
baselines based on unseen entity recognition with burst detection. Besides
notable emerging entities, our method can discover massive long-tail and
homographic emerging entities. An evaluation of relative recall shows that the
method detects 80.4% emerging entities newly registered in Wikipedia; 92.4% of
them are discovered earlier than their registration in Wikipedia, and the
average lead-time is more than one year (571 days).
# Multiple Generative Models Ensemble for Knowledge-Driven Proactive Human-Computer Dialogue Agent 
Multiple sequence to sequence models were used to establish an end-to-end
multi-turns proactive dialogue generation agent, with the aid of data
augmentation techniques and variant encoder-decoder structure designs. A
rank-based ensemble approach was developed for boosting performance. Results
indicate that our single model, in average, makes an obvious improvement in the
terms of F1-score and BLEU over the baseline by 18.67% on the DuConv dataset.
In particular, the ensemble methods further significantly outperform the
baseline by 35.85%.
# Knowledge-aware Pronoun Coreference Resolution 
Resolving pronoun coreference requires knowledge support, especially for
particular domains (e.g., medicine). In this paper, we explore how to leverage
different types of knowledge to better resolve pronoun coreference with a
neural model. To ensure the generalization ability of our model, we directly
incorporate knowledge in the format of triplets, which is the most common
format of modern knowledge graphs, instead of encoding it with features or
rules as that in conventional approaches. Moreover, since not all knowledge is
helpful in certain contexts, to selectively use them, we propose a knowledge
attention module, which learns to select and use informative knowledge based on
contexts, to enhance our model. Experimental results on two datasets from
different domains prove the validity and effectiveness of our model, where it
outperforms state-of-the-art baselines by a large margin. Moreover, since our
model learns to use external knowledge rather than only fitting the training
data, it also demonstrates superior performance to baselines in the
cross-domain setting.
# Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters 
In Vision-and-Language Navigation (VLN), an embodied agent needs to reach a
target destination with the only guidance of a natural language instruction. To
explore the environment and progress towards the target location, the agent
must perform a series of low-level actions, such as rotate, before stepping
ahead. In this paper, we propose to exploit dynamic convolutional filters to
encode the visual information and the lingual description in an efficient way.
Differently from some previous works that abstract from the agent perspective
and use high-level navigation spaces, we design a policy which decodes the
information provided by dynamic convolution into a series of low-level, agent
friendly actions. Results show that our model exploiting dynamic filters
performs better than other architectures with traditional convolution, being
the new state of the art for embodied VLN in the low-level action space.
Additionally, we attempt to categorize recent work on VLN depending on their
architectural choices and distinguish two main groups: we call them low-level
actions and high-level actions models. To the best of our knowledge, we are the
first to propose this analysis and categorization for VLN.
# Blind Universal Bayesian Image Denoising with Gaussian Noise Level Learning 
Blind and universal image denoising consists of a unique model that denoises
images with any level of noise. It is especially practical as noise levels do
not need to be known when the model is developed or at test time. We propose a
theoretically-grounded blind and universal deep learning image denoiser for
Gaussian noise. Our network is based on an optimal denoising solution, which we
call fusion denoising. It is derived theoretically with a Gaussian image prior
assumption. Synthetic experiments show our network's generalization strength to
unseen noise levels. We also adapt the fusion denoising network architecture
for real image denoising. Our approach improves real-world grayscale image
denoising PSNR results by up to $0.7dB$ for training noise levels and by up to
$2.82dB$ on noise levels not seen during training. It also improves
state-of-the-art color image denoising performance on every single noise level,
by an average of $0.1dB$, whether trained on or not.
# Dependency-aware Attention Control for Unconstrained Face Recognition with Image Sets 
This paper targets the problem of image set-based face verification and
identification. Unlike traditional single media (an image or video) setting, we
encounter a set of heterogeneous contents containing orderless images and
videos. The importance of each image is usually considered either equal or
based on their independent quality assessment. How to model the relationship of
orderless images within a set remains a challenge. We address this problem by
formulating it as a Markov Decision Process (MDP) in the latent space.
Specifically, we first present a dependency-aware attention control (DAC)
network, which resorts to actor-critic reinforcement learning for sequential
attention decision of each image embedding to fully exploit the rich
correlation cues among the unordered images. Moreover, we introduce its
sample-efficient variant with off-policy experience replay to speed up the
learning process. The pose-guided representation scheme can further boost the
performance at the extremes of the pose variation.
# Video Question Generation via Cross-Modal Self-Attention Networks Learning 
Video Question Answering (Video QA) is a critical and challenging task in
multimedia comprehension. While deep learning based models are extremely
capable of representing and understanding videos, these models heavily rely on
massive data, which is expensive to label. In this paper, we introduce a novel
task for automatically generating questions given a sequence of video frames
and the corresponding subtitles from a clip of video to reduce the huge
annotation cost. Learning to ask a question based on a video requires the model
to comprehend the rich semantics in the scene and the interplay between the
vision and the language. To address this, we propose a novel cross-modal
self-attention (CMSA) network to aggregate the diverse features from video
frames and subtitles. Excitingly, we demonstrate that our proposed model can
improve the (strong) baseline from 0.0738 to 0.1374 in BLEU4 score -- more than
0.063 improvement (i.e., 85\% relatively). Most of all, We arguably pave a
novel path toward solving the challenging Video QA task and provide detailed
analysis which ushers the avenues for future investigations.
# Deep Learning for Fine-Grained Image Analysis: A Survey 
Computer vision (CV) is the process of using machines to understand and
analyze imagery, which is an integral branch of artificial intelligence. Among
various research areas of CV, fine-grained image analysis (FGIA) is a
longstanding and fundamental problem, and has become ubiquitous in diverse
real-world applications. The task of FGIA targets analyzing visual objects from
subordinate categories, \eg, species of birds or models of cars. The small
inter-class variations and the large intra-class variations caused by the
fine-grained nature makes it a challenging problem. During the booming of deep
learning, recent years have witnessed remarkable progress of FGIA using deep
learning techniques. In this paper, we aim to give a survey on recent advances
of deep learning based FGIA techniques in a systematic way. Specifically, we
organize the existing studies of FGIA techniques into three major categories:
fine-grained image recognition, fine-grained image retrieval and fine-grained
image generation. In addition, we also cover some other important issues of
FGIA, such as publicly available benchmark datasets and its related domain
specific applications. Finally, we conclude this survey by highlighting several
directions and open problems which need be further explored by the community in
the future.
# AMD Severity Prediction And Explainability Using Image Registration And Deep Embedded Clustering 
We propose a method to predict severity of age related macular degeneration
(AMD) from input optical coherence tomography (OCT) images. Although there is
no standard clinical severity scale for AMD, we leverage deep learning (DL)
based image registration and clustering methods to identify diseased cases and
predict their severity. Experiments demonstrate our approach's disease
classification performance matches state of the art methods. The predicted
disease severity performs well on previously unseen data. Registration output
provides better explainability than class activation maps regarding label and
severity decisions
# Bilevel Integrative Optimization for Ill-posed Inverse Problems 
Classical optimization techniques often formulate the feasibility of the
problems as set, equality or inequality constraints. However, explicitly
designing these constraints is indeed challenging for complex real-world
applications and too strict constraints may even lead to intractable
optimization problems. On the other hand, it is still hard to incorporate
data-dependent information into conventional numerical iterations. To partially
address the above limits and inspired by the leader-follower gaming
perspective, this work first introduces a bilevel-type formulation to jointly
investigate the feasibility and optimality of nonconvex and nonsmooth
optimization problems. Then we develop an algorithmic framework to couple
forward-backward proximal computations to optimize our established bilevel
leader-follower model. We prove its convergence and estimate the convergence
rate. Furthermore, a learning-based extension is developed, in which we
establish an unrolling strategy to incorporate data-dependent network
architectures into our iterations. Fortunately, it can be proved that by
introducing some mild checking conditions, all our original convergence results
can still be preserved for this learnable extension. As a nontrivial byproduct,
we demonstrate how to apply this ensemble-like methodology to address different
low-level vision tasks. Extensive experiments verify the theoretical results
and show the advantages of our method against existing state-of-the-art
approaches.
# SAN: Scale-Aware Network for Semantic Segmentation of High-Resolution Aerial Images 
High-resolution aerial images have a wide range of applications, such as
military exploration, and urban planning. Semantic segmentation is a
fundamental method extensively used in the analysis of high-resolution aerial
images. However, the ground objects in high-resolution aerial images have the
characteristics of inconsistent scales, and this feature usually leads to
unexpected predictions. To tackle this issue, we propose a novel scale-aware
module (SAM). In SAM, we employ the re-sampling method aimed to make pixels
adjust their positions to fit the ground objects with different scales, and it
implicitly introduces spatial attention by employing a re-sampling map as the
weighted map. As a result, the network with the proposed module named
scale-aware network (SANet) has a stronger ability to distinguish the ground
objects with inconsistent scale. Other than this, our proposed modules can
easily embed in most of the existing network to improve their performance. We
evaluate our modules on the International Society for Photogrammetry and Remote
Sensing Vaihingen Dataset, and the experimental results and comprehensive
analysis demonstrate the effectiveness of our proposed module.
# Fast Universal Style Transfer for Artistic and Photorealistic Rendering 
Universal style transfer is an image editing task that renders an input
content image using the visual style of arbitrary reference images, including
both artistic and photorealistic stylization. Given a pair of images as the
source of content and the reference of style, existing solutions usually first
train an auto-encoder (AE) to reconstruct the image using deep features and
then embeds pre-defined style transfer modules into the AE reconstruction
procedure to transfer the style of the reconstructed image through modifying
the deep features. While existing methods typically need multiple rounds of
time-consuming AE reconstruction for better stylization, our work intends to
design novel neural network architectures on top of AE for fast style transfer
with fewer artifacts and distortions all in one pass of end-to-end inference.
To this end, we propose two network architectures named ArtNet and PhotoNet to
improve artistic and photo-realistic stylization, respectively. Extensive
experiments demonstrate that ArtNet generates images with fewer artifacts and
distortions against the state-of-the-art artistic transfer algorithms, while
PhotoNet improves the photorealistic stylization results by creating sharp
images faithfully preserving rich details of the input content. Moreover,
ArtNet and PhotoNet can achieve 3X to 100X speed-up over the state-of-the-art
algorithms, which is a major advantage for large content images.
# Revisiting Metric Learning for Few-Shot Image Classification 
The goal of few-shot learning is to recognize new visual concepts with just a
few amount of labeled samples in each class. Recent effective metric-based
few-shot approaches employ neural networks to learn a feature similarity
comparison between query and support examples. However, the importance of
feature embedding, i.e., exploring the relationship among training samples, is
neglected. In this work, we present a simple yet powerful baseline for few-shot
classification by emphasizing the importance of feature embedding.
Specifically, we revisit the classical triplet network from deep metric
learning, and extend it into a deep K-tuplet network for few-shot learning,
utilizing the relationship among the input samples to learn a general
representation learning via episode-training. Once trained, our network is able
to extract discriminative features for unseen novel categories and can be
seamlessly incorporated with a non-linear distance metric function to
facilitate the few-shot classification. Our result on the miniImageNet
benchmark outperforms other metric-based few-shot classification methods. More
importantly, when evaluated on completely different datasets (Caltech-101,
CUB-200, Stanford Dogs and Cars) using the model trained with miniImageNet, our
method significantly outperforms prior methods, demonstrating its superior
capability to generalize to unseen classes.
# Multi-level Wavelet Convolutional Neural Networks 
In computer vision, convolutional networks (CNNs) often adopts pooling to
enlarge receptive field which has the advantage of low computational
complexity. However, pooling can cause information loss and thus is detrimental
to further operations such as features extraction and analysis. Recently,
dilated filter has been proposed to trade off between receptive field size and
efficiency. But the accompanying gridding effect can cause a sparse sampling of
input images with checkerboard patterns. To address this problem, in this
paper, we propose a novel multi-level wavelet CNN (MWCNN) model to achieve
better trade-off between receptive field size and computational efficiency. The
core idea is to embed wavelet transform into CNN architecture to reduce the
resolution of feature maps while at the same time, increasing receptive field.
Specifically, MWCNN for image restoration is based on U-Net architecture, and
inverse wavelet transform (IWT) is deployed to reconstruct the high resolution
(HR) feature maps. The proposed MWCNN can also be viewed as an improvement of
dilated filter and a generalization of average pooling, and can be applied to
not only image restoration tasks, but also any CNNs requiring a pooling
operation. The experimental results demonstrate effectiveness of the proposed
MWCNN for tasks such as image denoising, single image super-resolution, JPEG
image artifacts removal and object classification.
# Unsupervised cycle-consistent deformation for shape matching 
We propose a self-supervised approach to deep surface deformation. Given a
pair of shapes, our algorithm directly predicts a parametric transformation
from one shape to the other respecting correspondences. Our insight is to use
cycle-consistency to define a notion of good correspondences in groups of
objects and use it as a supervisory signal to train our network. Our method
does not rely on a template, assume near isometric deformations or rely on
point-correspondence supervision. We demonstrate the efficacy of our approach
by using it to transfer segmentation across shapes. We show, on Shapenet, that
our approach is competitive with comparable state-of-the-art methods when
annotated training data is readily available, but outperforms them by a large
margin in the few-shot segmentation scenario.
# Multimodal Fusion with Deep Neural Networks for Audio-Video Emotion Recognition 
This paper presents a novel deep neural network (DNN) for multimodal fusion
of audio, video and text modalities for emotion recognition. The proposed DNN
architecture has independent and shared layers which aim to learn the
representation for each modality, as well as the best combined representation
to achieve the best prediction. Experimental results on the AVEC Sentiment
Analysis in the Wild dataset indicate that the proposed DNN can achieve a
higher level of Concordance Correlation Coefficient (CCC) than other
state-of-the-art systems that perform early fusion of modalities at
feature-level (i.e., concatenation) and late fusion at score-level (i.e.,
weighted average) fusion. The proposed DNN has achieved CCCs of 0.606, 0.534,
and 0.170 on the development partition of the dataset for predicting arousal,
valence and liking, respectively.
# Skin Lesion Analyser: An Efficient Seven-Way Multi-Class Skin Cancer Classification Using MobileNet 
Skin cancer, a major form of cancer, is a critical public health problem with
123,000 newly diagnosed melanoma cases and between 2 and 3 million non-melanoma
cases worldwide each year. The leading cause of skin cancer is high exposure of
skin cells to UV radiation, which can damage the DNA inside skin cells leading
to uncontrolled growth of skin cells. Skin cancer is primarily diagnosed
visually employing clinical screening, a biopsy, dermoscopic analysis, and
histopathological examination. It has been demonstrated that the dermoscopic
analysis in the hands of inexperienced dermatologists may cause a reduction in
diagnostic accuracy. Early detection and screening of skin cancer have the
potential to reduce mortality and morbidity. Previous studies have shown Deep
Learning ability to perform better than human experts in several visual
recognition tasks. In this paper, we propose an efficient seven-way automated
multi-class skin cancer classification system having performance comparable
with expert dermatologists. We used a pretrained MobileNet model to train over
HAM10000 dataset using transfer learning. The model classifies skin lesion
image with a categorical accuracy of 83.1 percent, top2 accuracy of 91.36
percent and top3 accuracy of 95.34 percent. The weighted average of precision,
recall, and f1-score were found to be 0.89, 0.83, and 0.83 respectively. The
model has been deployed as a web application for public use at
(https://saketchaturvedi.github.io). This fast, expansible method holds the
potential for substantial clinical impact, including broadening the scope of
primary care practice and augmenting clinical decision-making for dermatology
specialists.
# FC$^2$N: Fully Channel-Concatenated Network for Single Image Super-Resolution 
Most current image super-resolution (SR) methods based on deep convolutional
neural networks (CNNs) use residual learning in network structural design,
which contributes to effective back propagation, thus improving SR performance
by increasing model scale. However, deep residual network suffers some
redundancy in model representational capacity by introducing short paths, thus
hindering the full mining of model capacity. In addition, blindly enlarging the
model scale will cause more problems in model training, even with residual
learning. In this work, a novel network architecture is introduced to fully
exploit the representational capacity of the model, where all skip connections
are implemented by weighted channel concatenation, followed by a 1$\times$1
conv layer. Based on this weighted skip connection, we construct the building
modules of our model, and improve the global feature fusion (GFF). Unlike most
previous models, all skip connections in our network are channel-concatenated
and no residual connection is adopted. It is therefore termed as fully
channel-concatenated network (FC$^2$N). Due to the full exploitation of model
capacity, the proposed FC$^2$N achieves better performance than other advanced
models with fewer model parameters. Extensive experiments demonstrate the
superiority of our method to other methods, in terms of both quantitative
metrics and visual quality.
# ASCNet: Adaptive-Scale Convolutional Neural Networks for Multi-Scale Feature Learning 
Extracting multi-scale information is key to semantic segmentation. However,
the classic convolutional neural networks (CNNs) encounter difficulties in
achieving multi-scale information extraction: expanding convolutional kernel
incurs the high computational cost and using maximum pooling sacrifices image
information. The recently developed dilated convolution solves these problems,
but with the limitation that the dilation rates are fixed and therefore the
receptive field cannot fit for all objects with different sizes in the image.
We propose an adaptivescale convolutional neural network (ASCNet), which
introduces a 3-layer convolution structure in the end-to-end training, to
adaptively learn an appropriate dilation rate for each pixel in the image. Such
pixel-level dilation rates produce optimal receptive fields so that the
information of objects with different sizes can be extracted at the
corresponding scale. We compare the segmentation results using the classic CNN,
the dilated CNN and the proposed ASCNet on two types of medical images (The
Herlev dataset and SCD RBC dataset). The experimental results show that ASCNet
achieves the highest accuracy. Moreover, the automatically generated dilation
rates are positively correlated to the sizes of the objects, confirming the
effectiveness of the proposed method.
# Tree-gated Deep Regressor Ensemble For Face Alignment In The Wild 
Face alignment consists in aligning a shape model on a face in an image. It
is an active domain in computer vision as it is a preprocessing for
applications like facial expression recognition, face recognition and tracking,
face animation, etc. Current state-of-the-art methods already perform well on
"easy" datasets, i.e. those that present moderate variations in head pose,
expression, illumination or partial occlusions, but may not be robust to
"in-the-wild" data. In this paper, we address this problem by using an ensemble
of deep regressors instead of a single large regressor. Furthermore, instead of
averaging the outputs of each regressor, we propose an adaptive weighting
scheme that uses a tree-structured gate. Experiments on several challenging
face datasets demonstrate that our approach outperforms the state-of-the-art
methods.
# A Novel Teacher-Student Learning Framework For Occluded Person Re-Identification 
Person re-identification (re-id) has made great progress in recent years, but
occlusion is still a challenging problem which significantly degenerates the
identification performance. In this paper, we design a teacher-student learning
framework to learn an occlusion-robust model from the full-body person domain
to the occluded person domain. Notably, the teacher network only uses
large-scale full-body person data to simulate the learning process of occluded
person re-id. Based on the teacher network, the student network then trains a
better model by using inadequate real-world occluded person data. In order to
transfer more knowledge from the teacher network to the student network, we
equip the proposed framework with a co-saliency network and a cross-domain
simulator. The co-saliency network extracts the backbone features, and two
separated collaborative branches are followed by the backbone. One branch is a
classification branch for identity recognition and the other is a co-saliency
branch for guiding the network to highlight meaningful parts without any manual
annotation. The cross-domain simulator generates artificial occlusions on
full-body person data under a growing probability so that the teacher network
could train a cross-domain model by observing more and more occluded cases.
Experiments on four occluded person re-id benchmarks show that our method
outperforms other state-of-the-art methods.
# ELF: Embedded Localisation of Features in pre-trained CNN 
This paper introduces a novel feature detector based only on information
embedded inside a CNN trained on standard tasks (e.g. classification). While
previous works already show that the features of a trained CNN are suitable
descriptors, we show here how to extract the feature locations from the network
to build a detector. This information is computed from the gradient of the
feature map with respect to the input image. This provides a saliency map with
local maxima on relevant keypoint locations. Contrary to recent CNN-based
detectors, this method requires neither supervised training nor finetuning. We
evaluate how repeatable and how matchable the detected keypoints are with the
repeatability and matching scores. Matchability is measured with a simple
descriptor introduced for the sake of the evaluation. This novel detector
reaches similar performances on the standard evaluation HPatches dataset, as
well as comparable robustness against illumination and viewpoint changes on
Webcam and photo-tourism images. These results show that a CNN trained on a
standard task embeds feature location information that is as relevant as when
the CNN is specifically trained for feature detection.
# Dual Adversarial Learning with Attention Mechanism for Fine-grained Medical Image Synthesis 
Medical imaging plays a critical role in various clinical applications.
However, due to multiple considerations such as cost and risk, the acquisition
of certain image modalities could be limited. To address this issue, many
cross-modality medical image synthesis methods have been proposed. However, the
current methods cannot well model the hard-to-synthesis regions (e.g., tumor or
lesion regions). To address this issue, we propose a simple but effective
strategy, that is, we propose a dual-discriminator (dual-D) adversarial
learning system, in which, a global-D is used to make an overall evaluation for
the synthetic image, and a local-D is proposed to densely evaluate the local
regions of the synthetic image. More importantly, we build an adversarial
attention mechanism which targets at better modeling hard-to-synthesize regions
(e.g., tumor or lesion regions) based on the local-D. Experimental results show
the robustness and accuracy of our method in synthesizing fine-grained target
images from the corresponding source images. In particular, we evaluate our
method on two datasets, i.e., to address the tasks of generating T2 MRI from T1
MRI for the brain tumor images and generating MRI from CT. Our method
outperforms the state-of-the-art methods under comparison in all datasets and
tasks. And the proposed difficult-region-aware attention mechanism is also
proved to be able to help generate more realistic images, especially for the
hard-to-synthesize regions.
# Spacetime Graph Optimization for Video Object Segmentation 
In this paper we address the challenging task of object discovery and
segmentation in video. We introduce an efficient method that can be applied in
supervised and unsupervised scenarios, using a graph-based representation in
both space and time. Our method exploits the consistency in appearance and
motion patterns of pixels belonging to the same object. We formulate the task
as a clustering problem: graph nodes at the pixel level that belong to the
object of interest should form a strong cluster, linked through long range
optical flow chains and with similar motion and appearance features along those
chains. On one hand, the optimization problem aims to maximize the segmentation
clustering score based on the structure of pixel motions through space and
time. On the other, the segmentation should be consistent with the features at
the level of nodes, s.t. these features should be able to predict the
segmentation labels. The solution to our problem relates to spectral clustering
as well as to the classical regression analysis. It leads to a fast algorithm
that converges in a few iterations to a global optimum of the relaxed problem,
using fixed point iteration. The proposed method, namely GO-VOS, is relatively
fast and accurate. It can be used both as a standalone and completely
unsupervised method or in combination with other segmentation methods. In
experiments, we demonstrate top performance on several challenging datasets:
DAVIS, SegTrack and YouTube-Objects.
# Learning joint lesion and tissue segmentation from task-specific hetero-modal datasets 
Brain tissue segmentation from multimodal MRI is a key building block of many
neuroscience analysis pipelines. It could also play an important role in many
clinical imaging scenarios. Established tissue segmentation approaches have
however not been developed to cope with large anatomical changes resulting from
pathology. The effect of the presence of brain lesions, for example, on their
performance is thus currently uncontrolled and practically unpredictable.
Contrastingly, with the advent of deep neural networks (DNNs), segmentation of
brain lesions has matured significantly and is achieving performance levels
making it of interest for clinical use. However, few existing approaches allow
for jointly segmenting normal tissue and brain lesions. Developing a DNN for
such joint task is currently hampered by the fact that annotated datasets
typically address only one specific task and rely on a task-specific
hetero-modal imaging protocol. In this work, we propose a novel approach to
build a joint tissue and lesion segmentation model from task-specific
hetero-modal and partially annotated datasets. Starting from a variational
formulation of the joint problem, we show how the expected risk can be
decomposed and optimised empirically. We exploit an upper-bound of the risk to
deal with missing imaging modalities. For each task, our approach reaches
comparable performance than task-specific and fully-supervised models.
# Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation 
Despite the recent improvements in overall accuracy, deep learning systems
still exhibit low levels of robustness. Detecting possible failures is critical
for a successful clinical integration of these systems, where each data point
corresponds to an individual patient. Uncertainty measures are a promising
direction to improve failure detection since they provide a measure of a
system's confidence. Although many uncertainty estimation methods have been
proposed for deep learning, little is known on their benefits and current
challenges for medical image segmentation. Therefore, we report results of
evaluating common voxel-wise uncertainty measures with respect to their
reliability, and limitations on two medical image segmentation datasets.
Results show that current uncertainty methods perform similarly and although
they are well-calibrated at the dataset level, they tend to be miscalibrated at
subject-level. Therefore, the reliability of uncertainty estimates is
compromised, highlighting the importance of developing subject-wise uncertainty
estimations. Additionally, among the benchmarked methods, we found auxiliary
networks to be a valid alternative to common uncertainty methods since they can
be applied to any previously trained segmentation model.
# Learning Structural Graph Layouts and 3D Shapes for Long Span Bridges 3D Reconstruction 
A learning-based 3D reconstruction method for long-span bridges is proposed
in this paper. 3D reconstruction generates a 3D computer model of a real object
or scene from images, it involves many stages and open problems. Existing
point-based methods focus on generating 3D point clouds and their reconstructed
polygonal mesh or fitting-based geometrical models in urban scenes civil
structures reconstruction within Manhattan world constrains and have made great
achievements. Difficulties arise when an attempt is made to transfer these
systems to structures with complex topology and part relations like steel
trusses and long-span bridges, this could be attributed to point clouds are
often unevenly distributed with noise and suffer from occlusions and
incompletion, recovering a satisfactory 3D model from these highly unstructured
point clouds in a bottom-up pattern while preserving the geometrical and
topological properties makes enormous challenge to existing algorithms.
Considering the prior human knowledge that these structures are in conformity
to regular spatial layouts in terms of components, a learning-based
topology-aware 3D reconstruction method which can obtain high-level structural
graph layouts and low-level 3D shapes from images is proposed in this paper. We
demonstrate the feasibility of this method by testing on two real long-span
steel truss cable-stayed bridges.
# Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks 
Predicting the future trajectories of multiple interacting agents in a scene
has become an increasingly important problem for many different applications
ranging from control of autonomous vehicles and social robots to security and
surveillance. This problem is compounded by the presence of social interactions
between humans and their physical interactions with the scene. While the
existing literature has explored some of these cues, they mainly ignored the
multimodal nature of each human's future trajectory. In this paper, we present
Social-BiGAT, a graph-based generative adversarial network that generates
realistic, multimodal trajectory predictions by better modelling the social
interactions of pedestrians in a scene. Our method is based on a graph
attention network (GAT) that learns reliable feature representations that
encode the social interactions between humans in the scene, and a recurrent
encoder-decoder architecture that is trained adversarially to predict, based on
the features, the humans' paths. We explicitly account for the multimodal
nature of the prediction problem by forming a reversible transformation between
each scene and its latent noise vector, as in Bicycle-GAN. We show that our
framework achieves state-of-the-art performance comparing it to several
baselines on existing trajectory forecasting benchmarks.
# Facial Makeup Transfer Combining Illumination Transfer 
To meet the women appearance needs, we present a novel virtual experience
approach of facial makeup transfer, developed into windows platform application
software. The makeup effects could present on the user's input image in real
time, with an only single reference image. The input image and reference image
are divided into three layers by facial feature points landmarked: facial
structure layer, facial color layer, and facial detail layer. Except for the
above layers are processed by different algorithms to generate output image, we
also add illumination transfer, so that the illumination effect of the
reference image is automatically transferred to the input image. Our approach
has the following three advantages: (1) Black or dark and white facial makeup
could be effectively transferred by introducing illumination transfer; (2)
Efficiently transfer facial makeup within seconds compared to those methods
based on deep learning frameworks; (3) Reference images with the air-bangs
could transfer makeup perfectly.
# Data Distillation, Face-Related Tasks, Multi Task Learning, Semi-Supervised Learning 
We propose a new semi-supervised learning method on face-related tasks based
on Multi-Task Learning (MTL) and data distillation. The proposed method
exploits multiple datasets with different labels for different-but-related
tasks such as simultaneous age, gender, race, facial expression estimation.
Specifically, when there are only a few well-labeled data for a specific task
among the multiple related ones, we exploit the labels of other related tasks
in different domains. Our approach is composed of (1) a new MTL method which
can deal with weakly labeled datasets and perform several tasks simultaneously,
and (2) an MTL-based data distillation framework which enables network
generalization for the training and test data from different domains.
Experiments show that the proposed multi-task system performs each task better
than the baseline single task. It is also demonstrated that using different
domain datasets along with the main dataset can enhance network generalization
and overcome the domain differences between datasets. Also, comparing data
distillation both on the baseline and MTL framework, the latter shows more
accurate predictions on unlabeled data from different domains. Furthermore, by
proposing a new learning-rate optimization method, our proposed network is able
to dynamically tune its learning rate.
# Bootstrap Model Ensemble and Rank Loss for Engagement Intensity Regression 
This paper presents our approach for the engagement intensity regression task
of EmotiW 2019. The task is to predict the engagement intensity value of a
student when he or she is watching an online MOOCs video in various conditions.
Based on our winner solution last year, we mainly explore head features and
body features with a bootstrap strategy and two novel loss functions in this
paper. We maintain the framework of multi-instance learning with long
short-term memory (LSTM) network, and make three contributions. First, besides
of the gaze and head pose features, we explore facial landmark features in our
framework. Second, inspired by the fact that engagement intensity can be ranked
in values, we design a rank loss as a regularization which enforces a distance
margin between the features of distant category pairs and adjacent category
pairs. Third, we use the classical bootstrap aggregation method to perform
model ensemble which randomly samples a certain training data by several times
and then averages the model predictions. We evaluate the performance of our
method and discuss the influence of each part on the validation dataset. Our
methods finally win 3rd place with MSE of 0.0626 on the testing set.
# Perceptual representations of structural information in images: application to quality assessment of synthesized view in FTV scenario 
As the immersive multimedia techniques like Free-viewpoint TV (FTV) develop
at an astonishing rate, user's demand for high-quality immersive contents
increases dramatically. Unlike traditional uniform artifacts, the distortions
within immersive contents could be non-uniform structure-related and thus are
challenging for commonly used quality metrics. Recent studies have demonstrated
that the representation of visual features can be extracted from multiple
levels of the hierarchy. Inspired by the hierarchical representation mechanism
in the human visual system (HVS), in this paper, we explore to adopt structural
representations to quantitatively measure the impact of such structure-related
distortion on perceived quality in FTV scenario. More specifically, a
bio-inspired full reference image quality metric is proposed based on 1)
low-level contour descriptor; 2) mid-level contour category descriptor; and 3)
task-oriented non-natural structure descriptor. The experimental results show
that the proposed model outperforms significantly the state-of-the-art metrics.
# A unified neural network for object detection, multiple object tracking and vehicle re-identification 
Deep SORT\cite{wojke2017simple} is a tracking-by-detetion approach to
multiple object tracking with a detector and a RE-ID model.
  Both separately training and inference with the two model is time-comsuming.
  In this paper, we unify the detector and RE-ID model into an end-to-end
network, by adding an additional track branch for tracking in Faster RCNN
architecture. With a unified network, we are able to train the whole model
end-to-end with multi loss, which has shown much benefit in other recent works.
  The RE-ID model in Deep SORT needs to use deep CNNs to extract feature map
from detected object images, However, track branch in our proposed network
straight make use of
  the RoI feature vector in Faster RCNN baseline, which reduced the amount of
calculation.
  Since the single image lacks the same object which is necessary when we use
the triplet loss to optimizer the track branch, we concatenate the neighbouring
frames in a video to construct our training dataset.
  We have trained and evaluated our model on AIC19 vehicle tracking dataset,
experiment shows that our model with resnet101 backbone can achieve 57.79 \%
mAP and track vehicle well.
# A Deep Learning Approach for Real-Time 3D Human Action Recognition from Skeletal Data 
We present a new deep learning approach for real-time 3D human action
recognition from skeletal data and apply it to develop a vision-based
intelligent surveillance system. Given a skeleton sequence, we propose to
encode skeleton poses and their motions into a single RGB image. An Adaptive
Histogram Equalization (AHE) algorithm is then applied on the color images to
enhance their local patterns and generate more discriminative features. For
learning and classification tasks, we design Deep Neural Networks based on the
Densely Connected Convolutional Architecture (DenseNet) to extract features
from enhanced-color images and classify them into classes. Experimental results
on two challenging datasets show that the proposed method reaches
state-of-the-art accuracy, whilst requiring low computational time for training
and inference. This paper also introduces CEMEST, a new RGB-D dataset depicting
passenger behaviors in public transport. It consists of 203 untrimmed
real-world surveillance videos of realistic normal and anomalous events. We
achieve promising results on real conditions of this dataset with the support
of data augmentation and transfer learning techniques. This enables the
construction of real-world applications based on deep learning for enhancing
monitoring and security in public transport.
# Linking Art through Human Poses 
We address the discovery of composition transfer in artworks based on their
visual content. Automated analysis of large art collections, which are growing
as a result of art digitization among museums and galleries, is an important
tool for art history and assists cultural heritage preservation. Modern image
retrieval systems offer good performance on visually similar artworks, but fail
in the cases of more abstract composition transfer. The proposed approach links
artworks through a pose similarity of human figures depicted in images. Human
figures are the subject of a large fraction of visual art from middle ages to
modernity and their distinctive poses were often a source of inspiration among
artists. The method consists of two steps -- fast pose matching and robust
spatial verification. We experimentally show that explicit human pose matching
is superior to standard content-based image retrieval methods on a manually
annotated art composition transfer dataset.
# Unified Attentional Generative Adversarial Network for Brain Tumor Segmentation From Multimodal Unpaired Images 
In medical applications, the same anatomical structures may be observed in
multiple modalities despite the different image characteristics. Currently,
most deep models for multimodal segmentation rely on paired registered images.
However, multimodal paired registered images are difficult to obtain in many
cases. Therefore, developing a model that can segment the target objects from
different modalities with unpaired images is significant for many clinical
applications. In this work, we propose a novel two-stream translation and
segmentation unified attentional generative adversarial network (UAGAN), which
can perform any-to-any image modality translation and segment the target
objects simultaneously in the case where two or more modalities are available.
The translation stream is used to capture modality-invariant features of the
target anatomical structures. In addition, to focus on segmentation-related
features, we add attentional blocks to extract valuable features from the
translation stream. Experiments on three-modality brain tumor segmentation
indicate that UAGAN outperforms the existing methods in most cases.
# Variational Context: Exploiting Visual and Textual Context for Grounding Referring Expressions 
We focus on grounding (i.e., localizing or linking) referring expressions in
images, e.g., ``largest elephant standing behind baby elephant''. This is a
general yet challenging vision-language task since it does not only require the
localization of objects, but also the multimodal comprehension of context --
visual attributes (e.g., ``largest'', ``baby'') and relationships (e.g.,
``behind'') that help to distinguish the referent from other objects,
especially those of the same category. Due to the exponential complexity
involved in modeling the context associated with multiple image regions,
existing work oversimplifies this task to pairwise region modeling by multiple
instance learning. In this paper, we propose a variational Bayesian method,
called Variational Context, to solve the problem of complex context modeling in
referring expression grounding. Specifically, our framework exploits the
reciprocal relation between the referent and context, i.e., either of them
influences estimation of the posterior distribution of the other, and thereby
the search space of context can be greatly reduced. In addition to reciprocity,
our framework considers the semantic information of context, i.e., the
referring expression can be reproduced based on the estimated context. We also
extend the model to unsupervised setting where no annotation for the referent
is available. Extensive experiments on various benchmarks show consistent
improvement over state-of-the-art methods in both supervised and unsupervised
settings.
# Unsupervised Domain Alignment to Mitigate Low Level Dataset Biases 
Dataset bias is a well-known problem in the field of computer vision. The
presence of implicit bias in any image collection hinders a model trained and
validated on a particular dataset to yield similar accuracies when tested on
other datasets. In this paper, we propose a novel debiasing technique to reduce
the effects of a biased training dataset. Our goal is to augment the training
data using a generative network by learning a non-linear mapping from the
source domain (training set) to the target domain (testing set) while retaining
training set labels. The cycle consistency loss and adversarial loss for
generative adversarial networks are used to learn the mapping. A structured
similarity index (SSIM) loss is used to enforce label retention while
augmenting the training set. Our methods and hypotheses are supported by
quantitative comparisons with prior debiasing techniques. These comparisons
showcase the superiority of our method and its potential to mitigate the
effects of dataset bias during the inference stage.
# Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud 
In this paper, we propose the part-aware and aggregation neural network
(Part-A^2 net) for 3D object detection from point cloud. The whole framework
consists of the part-aware stage and the part-aggregation stage. Firstly, the
part-aware stage learns to simultaneously predict coarse 3D proposals and
accurate intra-object part locations with the free-of-charge supervisions
derived from 3D ground-truth boxes. The predicted intra-object part locations
within the same proposals are grouped by our new-designed RoI-aware point cloud
pooling module, which results in an effective representation to encode the
features of 3D proposals. Then the part-aggregation stage learns to re-score
the box and refine the box location based on the pooled part locations. We
present extensive experiments on the KITTI 3D object detection dataset, which
demonstrate that both the predicted intra-object part locations and the
proposed RoI-aware point cloud pooling scheme benefit 3D object detection and
our Part-A^2 net outperforms state-of-the-art methods by utilizing only point
cloud data.
# Correlation via synthesis: end-to-end nodule image generation and radiogenomic map learning based on generative adversarial network 
Radiogenomic map linking image features and gene expression profiles is
useful for noninvasively identifying molecular properties of a particular type
of disease. Conventionally, such map is produced in three separate steps: 1)
gene-clustering to "metagenes", 2) image feature extraction, and 3) statistical
correlation between metagenes and image features. Each step is independently
performed and relies on arbitrary measurements. In this work, we investigate
the potential of an end-to-end method fusing gene data with image features to
generate synthetic image and learn radiogenomic map simultaneously. To achieve
this goal, we develop a generative adversarial network (GAN) conditioned on
both background images and gene expression profiles, synthesizing the
corresponding image. Image and gene features are fused at different scales to
ensure the realism and quality of the synthesized image. We tested our method
on non-small cell lung cancer (NSCLC) dataset. Results demonstrate that the
proposed method produces realistic synthetic images, and provides a promising
way to find gene-image relationship in a holistic end-to-end manner.
# Point-Voxel CNN for Efficient 3D Deep Learning 
We present Point-Voxel CNN (PVCNN) for efficient, fast 3D deep learning.
Previous work processes 3D data using either voxel-based or point-based NN
models. However, both approaches are computationally inefficient. The
computation cost and memory footprints of the voxel-based models grow cubically
with the input resolution, making it memory-prohibitive to scale up the
resolution. As for point-based networks, up to 80% of the time is wasted on
structuring the irregular data which have rather poor memory locality, not on
the actual feature extraction. In this paper, we propose PVCNN that represents
the 3D input data in points to reduce the memory consumption, while performing
the convolutions in voxels to largely reduce the irregular data access and
improve the locality. Our PVCNN model is both memory and computation efficient.
Evaluated on semantic and part segmentation datasets, it achieves much higher
accuracy than the voxel-based baseline with 10x GPU memory reduction; it also
outperforms the state-of-the-art point-based models with 7x measured speedup on
average. Remarkably, narrower version of PVCNN achieves 2x speedup over
PointNet (an extremely efficient model) on part and scene segmentation
benchmarks with much higher accuracy. We validate the general effectiveness of
our PVCNN on 3D object detection: by replacing the primitives in Frustrum
PointNet with PVConv, it outperforms Frustrum PointNet++ by 2.4% mAP on average
with 1.5x measured speedup and GPU memory reduction.
# Self-supervised Learning of Distance Functions for Goal-Conditioned Reinforcement Learning 
Goal-conditioned policies are used in order to break down complex
reinforcement learning (RL) problems by using subgoals, which can be defined
either in state space or in a latent feature space. This can increase the
efficiency of learning by using a curriculum, and also enables simultaneous
learning and generalization across goals. A crucial requirement of
goal-conditioned policies is to be able to determine whether the goal has been
achieved. Having a notion of distance to a goal is thus a crucial component of
this approach. However, it is not straightforward to come up with an
appropriate distance, and in some tasks, the goal space may not even be known a
priori. In this work we learn a distance-to-goal estimate which is computed in
terms of the number of actions that would need to be carried out in a
self-supervised approach. Our method solves complex tasks without prior domain
knowledge in the online setting in three different scenarios in the context of
goal-conditioned policies a) the goal space is the same as the state space b)
the goal space is given but an appropriate distance is unknown and c) the state
space is accessible, but only a subset of the state space represents desired
goals, and this subset is known a priori. We also propose a goal-generation
mechanism as a secondary contribution.
# Gaussian Processes for Analyzing Positioned Trajectories in Sports 
Kernel-based machine learning approaches are gaining increasing interest for
exploring and modeling large dataset in recent years. Gaussian process (GP) is
one example of such kernel-based approaches, which can provide very good
performance for nonlinear modeling problems. In this work, we first propose a
grey-box modeling approach to analyze the forces in cross country skiing races.
To be more precise, a disciplined set of kinetic motion model formulae is
combined with data-driven Gaussian process regression model, which accounts for
everything unknown in the system. Then, a modeling approach is proposed to
analyze the kinetic flow of both individual and clusters of skiers. The
proposed approaches can be generally applied to use cases where positioned
trajectories and kinetic measurements are available. The proposed approaches
are evaluated using data collected from the Falun Nordic World Ski
Championships 2015, in particular the Men's cross country $4\times10$ km relay.
Forces during the cross country skiing races are analyzed and compared.
Velocity models for skiers at different competition stages are also evaluated.
Finally, the comparisons between the grey-box and black-box approach are
carried out, where the grey-box approach can reduce the predictive uncertainty
by $30\%$ to $40\%$.
# Learning a Behavioral Repertoire from Demonstrations 
Imitation Learning (IL) is a machine learning approach to learn a policy from
a dataset of demonstrations. IL can be useful to kick-start learning before
applying reinforcement learning (RL) but it can also be useful on its own, e.g.
to learn to imitate human players in video games. However, a major limitation
of current IL approaches is that they learn only a single "average" policy
based on a dataset that possibly contains demonstrations of numerous different
types of behaviors. In this paper, we propose a new approach called Behavioral
Repertoire Imitation Learning (BRIL) that instead learns a repertoire of
behaviors from a set of demonstrations by augmenting the state-action pairs
with behavioral descriptions. The outcome of this approach is a single neural
network policy conditioned on a behavior description that can be precisely
modulated. We apply this approach to train a policy on 7,777 human replays to
perform build-order planning in StarCraft II. Principal Component Analysis
(PCA) is applied to construct a low-dimensional behavioral space from the
high-dimensional army unit composition of each demonstration. The results
demonstrate that the learned policy can be effectively manipulated to express
distinct behaviors. Additionally, by applying the UCB1 algorithm, we are able
to adapt the behavior of the policy - in-between games - to reach a performance
beyond that of the traditional IL baseline approach.
# Jointly Aligning and Predicting Continuous Emotion Annotations 
Time-continuous dimensional descriptions of emotions (e.g., arousal, valence)
allow researchers to characterize short-time changes and to capture long-term
trends in emotion expression. However, continuous emotion labels are generally
not synchronized with the input speech signal due to delays caused by
reaction-time, which is inherent in human evaluations. To deal with this
challenge, we introduce a new convolutional neural network (multi-delay sinc
network) that is able to simultaneously align and predict labels in an
end-to-end manner. The proposed network is a stack of convolutional layers
followed by an aligner network that aligns the speech signal and emotion
labels. This network is implemented using a new convolutional layer that we
introduce, the delayed sinc layer. It is a time-shifted low-pass (sinc) filter
that uses a gradient-based algorithm to learn a single delay. Multiple delayed
sinc layers can be used to compensate for a non-stationary delay that is a
function of the acoustic space. We test the efficacy of this system on two
common emotion datasets, RECOLA and SEWA, and show that this approach obtains
state-of-the-art speech-only results by learning time-varying delays while
predicting dimensional descriptors of emotions.
# A Communication-Efficient Multi-Agent Actor-Critic Algorithm for Distributed Reinforcement Learning 
This paper considers a distributed reinforcement learning problem in which a
network of multiple agents aim to cooperatively maximize the globally averaged
return through communication with only local neighbors. A randomized
communication-efficient multi-agent actor-critic algorithm is proposed for
possibly unidirectional communication relationships depicted by a directed
graph. It is shown that the algorithm can solve the problem for strongly
connected graphs by allowing each agent to transmit only two scalar-valued
variables at one time.
# Generative Counterfactual Introspection for Explainable Deep Learning 
In this work, we propose an introspection technique for deep neural networks
that relies on a generative model to instigate salient editing of the input
image for model interpretation. Such modification provides the fundamental
interventional operation that allows us to obtain answers to counterfactual
inquiries, i.e., what meaningful change can be made to the input image in order
to alter the prediction. We demonstrate how to reveal interesting properties of
the given classifiers by utilizing the proposed introspection approach on both
the MNIST and the CelebA dataset.
# Playing Flappy Bird via Asynchronous Advantage Actor Critic Algorithm 
Flappy Bird, which has a very high popularity, has been trained in many
algorithms. Some of these studies were trained from raw pixel values of game
and some from specific attributes. In this study, the model was trained with
raw game images, which had not been seen before. The trained model has learned
as reinforcement when to make which decision. As an input to the model, the
reward or penalty at the end of each step was returned and the training was
completed. Flappy Bird game was trained with the Reinforcement Learning
algorithm Deep Q-Network and Asynchronous Advantage Actor Critic (A3C)
algorithms.
# Regularizing linear inverse problems with convolutional neural networks 
Deep convolutional neural networks trained on large datsets have emerged as
an intriguing alternative for compressing images and solving inverse problems
such as denoising and compressive sensing. However, it has only recently been
realized that even without training, convolutional networks can function as
concise image models, and thus regularize inverse problems. In this paper, we
provide further evidence for this finding by studying variations of
convolutional neural networks that map few weight parameters to an image. The
networks we consider only consist of convolutional operations, with either
fixed or parameterized filters followed by ReLU non-linearities. We demonstrate
that with both fixed and parameterized convolutional filters those networks
enable representing images with few coefficients. What is more, the
underparameterization enables regularization of inverse problems, in particular
recovering an image from few observations. We show that, similar to standard
compressive sensing guarantees, on the order of the number of model parameters
many measurements suffice for recovering an image from compressive
measurements. Finally, we demonstrate that signal recovery with a un-trained
convolutional network outperforms standard l1 and total variation minimization
for magnetic resonance imaging (MRI).
# Adversarial Fault Tolerant Training for Deep Neural Networks 
Deep Learning Accelerators are prone to faults which manifest in the form of
errors in Neural Networks. Fault Tolerance in Neural Networks is crucial in
real-time safety critical applications requiring computation for long
durations. Neural Networks with high regularisation exhibit superior fault
tolerance, however, at the cost of classification accuracy. In the view of
difference in functionality, a Neural Network is modelled as two separate
networks, i.e, the Feature Extractor with unsupervised learning objective and
the Classifier with a supervised learning objective. Traditional approaches of
training the entire network using a single supervised learning objective is
insufficient to achieve the objectives of the individual components optimally.
In this work, a novel multi-criteria objective function, combining unsupervised
training of the Feature Extractor followed by supervised tuning with Classifier
Network is proposed. The unsupervised training solves two games simultaneously
in the presence of adversary neural networks with conflicting objectives to the
Feature Extractor. The first game minimises the loss in reconstructing the
input image for indistinguishability given the features from the Extractor, in
the presence of a generative decoder. The second game solves a minimax
constraint optimisation for distributional smoothening of feature space to
match a prior distribution, in the presence of a Discriminator network. The
resultant strongly regularised Feature Extractor is combined with the
Classifier Network for supervised fine-tuning. The proposed Adversarial Fault
Tolerant Neural Network Training is scalable to large networks and is
independent of the architecture. The evaluation on benchmarking datasets:
FashionMNIST and CIFAR10, indicates that the resultant networks have high
accuracy with superior tolerance to stuck at "0" faults compared to widely used
regularisers.
# Intrinsic Motivation Driven Intuitive Physics Learning using Deep Reinforcement Learning with Intrinsic Reward Normalization 
At an early age, human infants are able to learn and build a model of the
world very quickly by constantly observing and interacting with objects around
them. One of the most fundamental intuitions human infants acquire is intuitive
physics. Human infants learn and develop these models, which later serve as
prior knowledge for further learning. Inspired by such behaviors exhibited by
human infants, we introduce a graphical physics network integrated with deep
reinforcement learning. Specifically, we introduce an intrinsic reward
normalization method that allows our agent to efficiently choose actions that
can improve its intuitive physics model the most.
  Using a 3D physics engine, we show that our graphical physics network is able
to infer object's positions and velocities very effectively, and our deep
reinforcement learning network encourages an agent to improve its model by
making it continuously interact with objects only using intrinsic motivation.
We experiment our model in both stationary and non-stationary state problems
and show benefits of our approach in terms of the number of different actions
the agent performs and the accuracy of agent's intuition model.
  Videos are at https://www.youtube.com/watch?v=pDbByp91r3M&t=2s
# AutoSlim: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates 
Structured weight pruning is a representative model compression technique of
DNNs to reduce the storage and computation requirements and accelerate
inference. An automatic hyperparameter determination process is necessary due
to the large number of flexible hyperparameters. This work proposes AutoSlim,
an automatic structured pruning framework with the following key performance
improvements: (i) effectively incorporate the combination of structured pruning
schemes in the automatic process; (ii) adopt the state-of-art ADMM-based
structured weight pruning as the core algorithm, and propose an innovative
additional purification step for further weight reduction without accuracy
loss; and (iii) develop effective heuristic search method enhanced by
experience-based guided search, replacing the prior deep reinforcement learning
technique which has underlying incompatibility with the target pruning problem.
Extensive experiments on CIFAR-10 and ImageNet datasets demonstrate that
AutoSlim is the key to achieve ultra-high pruning rates on the number of
weights and FLOPs that cannot be achieved before. As an example, AutoSlim
outperforms the prior work on automatic model compression by up to 33$\times$
in pruning rate under the same accuracy. We release all models of this work at
anonymous link: http://bit.ly/2VZ63dS.
# Diachronic Embedding for Temporal Knowledge Graph Completion 
Knowledge graphs (KGs) typically contain temporal facts indicating
relationships among entities at different times. Due to their incompleteness,
several approaches have been proposed to infer new facts for a KG based on the
existing ones-a problem known as KG completion. KG embedding approaches have
proved effective for KG completion, however, they have been developed mostly
for static KGs. Developing temporal KG embedding models is an increasingly
important problem. In this paper, we build novel models for temporal KG
completion through equipping static models with a diachronic entity embedding
function which provides the characteristics of entities at any point in time.
This is in contrast to the existing temporal KG embedding approaches where only
static entity features are provided. The proposed embedding function is
model-agnostic and can be potentially combined with any static model. We prove
that combining it with SimplE, a recent model for static KG embedding, results
in a fully expressive model for temporal KG completion. Our experiments
indicate the superiority of our proposal compared to existing baselines.
# Intelligent Systems Design for Malware Classification Under Adversarial Conditions 
The use of machine learning and intelligent systems has become an established
practice in the realm of malware detection and cyber threat prevention. In an
environment characterized by widespread accessibility and big data, the
feasibility of malware classification without the use of artificial
intelligence-based techniques has been diminished exponentially. Also
characteristic of the contemporary realm of automated, intelligent malware
detection is the threat of adversarial machine learning. Adversaries are
looking to target the underlying data and/or algorithm responsible for the
functionality of malware classification to map its behavior or corrupt its
functionality. The ends of such adversaries are bypassing the cyber security
measures and increasing malware effectiveness. The focus of this research is
the design of an intelligent systems approach using machine learning that can
accurately and robustly classify malware under adversarial conditions. Such an
outcome ultimately relies on increased flexibility and adaptability to build a
model robust enough to identify attacks on the underlying algorithm.
# Towards Debugging Deep Neural Networks by Generating Speech Utterances 
Deep neural networks (DNN) are able to successfully process and classify
speech utterances. However, understanding the reason behind a classification by
DNN is difficult. One such debugging method used with image classification DNNs
is activation maximization, which generates example-images that are classified
as one of the classes. In this work, we evaluate applicability of this method
to speech utterance classifiers as the means to understanding what DNN "listens
to". We trained a classifier using the speech command corpus and then use
activation maximization to pull samples from the trained model. Then we
synthesize audio from features using WaveNet vocoder for subjective analysis.
We measure the quality of generated samples by objective measurements and
crowd-sourced human evaluations. Results show that when combined with the prior
of natural speech, activation maximization can be used to generate examples of
different classes. Based on these results, activation maximization can be used
to start opening up the DNN black-box in speech tasks.
# Weakly-supervised Knowledge Graph Alignment with Adversarial Learning 
This paper studies aligning knowledge graphs from different sources or
languages. Most existing methods train supervised methods for the alignment,
which usually require a large number of aligned knowledge triplets. However,
such a large number of aligned knowledge triplets may not be available or are
expensive to obtain in many domains. Therefore, in this paper we propose to
study aligning knowledge graphs in fully-unsupervised or weakly-supervised
fashion, i.e., without or with only a few aligned triplets. We propose an
unsupervised framework to align the entity and relation embddings of different
knowledge graphs with an adversarial learning framework. Moreover, a
regularization term which maximizes the mutual information between the
embeddings of different knowledge graphs is used to mitigate the problem of
mode collapse when learning the alignment functions. Such a framework can be
further seamlessly integrated with existing supervised methods by utilizing a
limited number of aligned triples as guidance. Experimental results on multiple
datasets prove the effectiveness of our proposed approach in both the
unsupervised and the weakly-supervised settings.
# What graph neural networks cannot learn: depth vs width 
This paper studies the capacity limits of graph neural networks (GNN). Rather
than focusing on a specific architecture, the networks considered here are
those that fall within the message-passing framework, a model that encompasses
several state-of-the-art networks. Two main results are presented. First, GNN
are shown to be Turing universal under sufficient conditions on their depth,
width, node identification, and layer expressiveness. In addition, it is
discovered that GNN can lose a significant portion of their power when their
depth and width is restricted. The proposed impossibility statements stem from
a new technique that enables the re-purposing of seminal results from
theoretical computer science. This leads to lower bounds for an array of
decision, optimization, and estimation problems involving graphs. Strikingly,
several of these problems are deemed impossible unless the product of a GNN's
depth and width exceeds the graph size; this dependence remains significant
even for tasks that appear simple or when considering approximation.
# Towards Robust, Locally Linear Deep Networks 
Deep networks realize complex mappings that are often understood by their
locally linear behavior at or around points of interest. For example, we use
the derivative of the mapping with respect to its inputs for sensitivity
analysis, or to explain (obtain coordinate relevance for) a prediction. One key
challenge is that such derivatives are themselves inherently unstable. In this
paper, we propose a new learning problem to encourage deep networks to have
stable derivatives over larger regions. While the problem is challenging in
general, we focus on networks with piecewise linear activation functions. Our
algorithm consists of an inference step that identifies a region around a point
where linear approximation is provably stable, and an optimization step to
expand such regions. We propose a novel relaxation to scale the algorithm to
realistic models. We illustrate our method with residual and recurrent networks
on image and sequence datasets.
# Deep Exponential-Family Auto-Encoders 
We consider the problem of learning recurring convolutional patterns from
data that are not necessarily real valued, such as binary or count-valued data.
We cast the problem as one of learning a convolutional dictionary, subject to
sparsity constraints, given observations drawn from a distribution that belongs
to the canonical exponential family. We propose two general approaches towards
its solution. The first approach uses the $\ell_0$ pseudo-norm to enforce
sparsity and is reminiscent of the alternating-minimization algorithm for
classical convolutional dictionary learning (CDL). The second approach, which
uses the $\ell_1$ norm to enforce sparsity, generalizes to the exponential
family the recently-shown connection between CDL and a class of ReLU
auto-encoders for Gaussian observations. The two approaches can each be
interpreted as an auto-encoder, the weights of which are in one-to-one
correspondence with the parameters of the convolutional dictionary. Our key
insight is that, unless the observations are Gaussian valued, the input fed
into the encoder ought to be modified iteratively, and in a specific manner,
using the parameters of the dictionary. Compared to the $\ell_0$ approach, once
trained, the forward pass through the $\ell_1$ encoder computes sparse codes
orders of magnitude more efficiently. We apply the two approaches to the
unsupervised learning of the stimulus effect from neural spiking data acquired
in the barrel cortex of mice in response to periodic whisker deflections. We
demonstrate that they are both superior to generalized linear models, which
rely on hand-crafted features.
# Quantitative $W_1$ Convergence of Langevin-Like Stochastic Processes with Non-Convex Potential State-Dependent Noise 
We prove quantitative convergence rates at which discrete Langevin-like
processes converge to the invariant distribution of a related stochastic
differential equation. We study the setup where the additive noise can be
non-Gaussian and state-dependent and the potential function can be non-convex.
We show that the key properties of these processes depend on the potential
function and the second moment of the additive noise. We apply our theoretical
findings to studying the convergence of Stochastic Gradient Descent (SGD) for
non-convex problems and corroborate them with experiments using SGD to train
deep neural networks on the CIFAR-10 dataset.
# Quantum-inspired canonical correlation analysis for exponentially large dimensional data 
Canonical correlation analysis (CCA) is a technique to find statistical
dependencies between a pair of multivariate data. However, its application to
high dimensional data is limited due to the resulting time complexity. While
the conventional CCA algorithm requires polynomial time, we have developed an
algorithm that approximates CCA with computational time proportional to the
logarithm of the input dimensionality using quantum-inspired computation. The
computational efficiency and approximation performance of the proposed
quantum-inspired CCA (qiCCA) algorithm are experimentally demonstrated.
Furthermore, the fast computation of qiCCA allows us to directly apply CCA even
after nonlinearly mapping raw input data into very high dimensional spaces.
Experiments performed using a benchmark dataset demonstrated that, by mapping
the raw input data into the high dimensional spaces with second-order
monomials, the proposed qiCCA extracted more correlations than linear CCA and
was comparable to deep CCA and kernel CCA. These results suggest that qiCCA is
considerably useful and quantum-inspired computation has the potential to
unlock a new field in which exponentially large dimensional data can be
analyzed.
# Resource-Efficient Computing in Wearable Systems 
We propose two optimization techniques to minimize memory usage and
computation while meeting system timing constraints for real-time
classification in wearable systems. Our method derives a hierarchical
classifier structure for Support Vector Machine (SVM) in order to reduce the
amount of computations, based on the probability distribution of output classes
occurrences. Also, we propose a memory optimization technique based on SVM
parameters, which results in storing fewer support vectors and as a result
requiring less memory. To demonstrate the efficiency of our proposed
techniques, we performed an activity recognition experiment and were able to
save up to 35% and 56% in memory storage when classifying 14 and 6 different
activities, respectively. In addition, we demonstrated that there is a
trade-off between accuracy of classification and memory savings, which can be
controlled based on application requirements.
# Resource-Efficient Wearable Computing for Real-Time Reconfigurable 
Advances in embedded systems have enabled integration of many lightweight
sensory devices within our daily life. In particular, this trend has given rise
to continuous expansion of wearable sensors in a broad range of applications
from health and fitness monitoring to social networking and military
surveillance. Wearables leverage machine learning techniques to profile
behavioral routine of their end-users through activity recognition algorithms.
Current research assumes that such machine learning algorithms are trained
offline. In reality, however, wearables demand continuous reconfiguration of
their computational algorithms due to their highly dynamic operation.
Developing a personalized and adaptive machine learning model requires
real-time reconfiguration of the model. Due to stringent computation and memory
constraints of these embedded sensors, the training/re-training of the
computational algorithms need to be memory- and computation-efficient. In this
paper, we propose a framework, based on the notion of online learning, for
real-time and on-device machine learning training. We propose to transform the
activity recognition problem from a multi-class classification problem to a
hierarchical model of binary decisions using cascading online binary
classifiers. Our results, based on Pegasos online learning, demonstrate that
the proposed approach achieves 97% accuracy in detecting activities of varying
intensities using a limited memory while power usages of the system is reduced
by more than 40%.
# A Human-Grounded Evaluation of SHAP for Alert Processing 
In the past years, many new explanation methods have been proposed to achieve
interpretability of machine learning predictions. However, the utility of these
methods in practical applications has not been researched extensively. In this
paper we present the results of a human-grounded evaluation of SHAP, an
explanation method that has been well-received in the XAI and related
communities. In particular, we study whether this local model-agnostic
explanation method can be useful for real human domain experts to assess the
correctness of positive predictions, i.e. alerts generated by a classifier. We
performed experimentation with three different groups of participants (159 in
total), who had basic knowledge of explainable machine learning. We performed a
qualitative analysis of recorded reflections of experiment participants
performing alert processing with and without SHAP information. The results
suggest that the SHAP explanations do impact the decision-making process,
although the model's confidence score remains to be a leading source of
evidence. We statistically test whether there is a significant difference in
task utility metrics between tasks for which an explanation was available and
tasks in which it was not provided. As opposed to common intuitions, we did not
find a significant difference in alert processing performance when a SHAP
explanation is available compared to when it is not.
# Fast ES-RNN: A GPU Implementation of the ES-RNN Algorithm 
Due to their prevalence, time series forecasting is crucial in multiple
domains. We seek to make state-of-the-art forecasting fast, accessible, and
generalizable. ES-RNN is a hybrid between classical state space forecasting
models and modern RNNs that achieved a 9.4% sMAPE improvement in the M4
competition. Crucially, ES-RNN implementation requires per-time series
parameters. By vectorizing the original implementation and porting the
algorithm to a GPU, we achieve up to 322x training speedup depending on batch
size with similar results as those reported in the original submission. Our
code can be found at: https://github.com/damitkwr/ESRNN-GPU
# Case-Based Reasoning for Assisting Domain Experts in Processing Fraud Alerts of Black-Box Machine Learning Models 
In many contexts, it can be useful for domain experts to understand to what
extent predictions made by a machine learning model can be trusted. In
particular, estimates of trustworthiness can be useful for fraud analysts who
process machine learning-generated alerts of fraudulent transactions. In this
work, we present a case-based reasoning (CBR) approach that provides evidence
on the trustworthiness of a prediction in the form of a visualization of
similar previous instances. Different from previous works, we consider
similarity of local post-hoc explanations of predictions and show empirically
that our visualization can be useful for processing alerts. Furthermore, our
approach is perceived useful and easy to use by fraud analysts at a major Dutch
bank.
# Fast and Provable ADMM for Learning with Generative Priors 
In this work, we propose a (linearized) Alternating Direction
Method-of-Multipliers (ADMM) algorithm for minimizing a convex function subject
to a nonconvex constraint. We focus on the special case where such constraint
arises from the specification that a variable should lie in the range of a
neural network. This is motivated by recent successful applications of
Generative Adversarial Networks (GANs) in tasks like compressive sensing,
denoising and robustness against adversarial examples. The derived rates for
our algorithm are characterized in terms of certain geometric properties of the
generator network, which we show hold for feedforward architectures, under mild
assumptions. Unlike gradient descent (GD), it can efficiently handle non-smooth
objectives as well as exploit efficient partial minimization procedures, thus
being faster in many practical scenarios.
# Individual Regret in Cooperative Nonstochastic Multi-Armed Bandits 
We study agents communicating over an underlying network by exchanging
messages, in order to optimize their individual regret in a common
nonstochastic multi-armed bandit problem. We derive regret minimization
algorithms that guarantee for each agent $v$ an individual expected regret of
\[
\widetilde{O}\left(\sqrt{\left(1+\frac{K}{\left|\mathcal{N}\left(v\right)\right|}\right)T}\right),
\] where $T$ is the number of time steps, $K$ is the number of actions and
$\mathcal{N}\left(v\right)$ is the set of neighbors of agent $v$ in the
communication graph. We present algorithms both for the case that the
communication graph is known to all the agents, and for the case that the graph
is unknown. When the graph is unknown, each agent knows only the set of its
neighbors and an upper bound on the total number of agents. The individual
regret between the models differs only by a logarithmic factor. Our work
resolves an open problem from [Cesa-Bianchi et al., 2019b].
# Improving Detection of Credit Card Fraudulent Transactions using Generative Adversarial Networks 
In this study, we employ Generative Adversarial Networks as an oversampling
method to generate artificial data to assist with the classification of credit
card fraudulent transactions. GANs is a generative model based on the idea of
game theory, in which a generator G and a discriminator D are trying to
outsmart each other. The objective of the generator is to confuse the
discriminator. The objective of the discriminator is to distinguish the
instances coming from the generator and the instances coming from the original
dataset. By training GANs on a set of credit card fraudulent transactions, we
are able to improve the discriminatory power of classifiers. The experiment
results show that the Wasserstein-GAN is more stable in training and produce
more realistic fraudulent transactions than the other GANs. On the other hand,
the conditional version of GANs in which labels are set by k-means clustering
does not necessarily improve the non-conditional versions of GANs.
# Copula & Marginal Flows: Disentangling the Marginal from its Joint 
Deep generative networks such as GANs and normalizing flows flourish in the
context of high-dimensional tasks such as image generation. However, so far
exact modeling or extrapolation of distributional properties such as the tail
asymptotics generated by a generative network is not available. In this paper,
we address this issue for the first time in the deep learning literature by
making two novel contributions. First, we derive upper bounds for the tails
that can be expressed by a generative network and demonstrate Lp-space related
properties. There we show specifically that in various situations an optimal
generative network does not exist. Second, we introduce and propose copula and
marginal generative flows (CM flows) which allow for an exact modeling of the
tail and any prior assumption on the CDF up to an approximation of the uniform
distribution. Our numerical results support the use of CM flows.
# Privacy-Preserving Classification with Secret Vector Machines 
Today, large amounts of valuable data are distributed among millions of
user-held devices, such as personal computers, phones, or Internet-of-things
devices. Many companies collect such data with the goal of using it for
training machine learning models allowing them to improve their services.
However, user-held data is often sensitive, and collecting it is problematic in
terms of privacy. We address this issue by proposing a novel way of training a
supervised classifier in a distributed setting akin to the recently proposed
federated learning paradigm (McMahan et al. 2017), but under the stricter
privacy requirement that the server that trains the model is assumed to be
untrusted and potentially malicious; we thus preserve user privacy by design,
rather than by trust. In particular, our framework, called secret vector
machine (SecVM), provides an algorithm for training linear support vector
machines (SVM) in a setting in which data-holding clients communicate with an
untrusted server by exchanging messages designed to not reveal any personally
identifiable information. We evaluate our model in two ways. First, in an
offline evaluation, we train SecVM to predict user gender from tweets, showing
that we can preserve user privacy without sacrificing classification
performance. Second, we implement SecVM's distributed framework for the Cliqz
web browser and deploy it for predicting user gender in a large-scale online
evaluation with thousands of clients, outperforming baselines by a large margin
and thus showcasing that SecVM is practicable in production environments.
Overall, this work demonstrates the feasibility of machine learning on data
from thousands of users without collecting any personal data. We believe this
is an innovative approach that will help reconcile machine learning with data
privacy.
# Etalumis: Bringing Probabilistic Programming to Scientific Simulators at Scale 
Probabilistic programming languages (PPLs) are receiving widespread attention
for performing Bayesian inference in complex generative models. However,
applications to science remain limited because of the impracticability of
rewriting complex scientific simulators in a PPL, the computational cost of
inference, and the lack of scalable implementations. To address these, we
present a novel PPL framework that couples directly to existing scientific
simulators through a cross-platform probabilistic execution protocol and
provides Markov chain Monte Carlo (MCMC) and deep-learning-based inference
compilation (IC) engines for tractable inference. To guide IC inference, we
perform distributed training of a dynamic 3DCNN--LSTM architecture with a
PyTorch-MPI-based framework on 1,024 32-core CPU nodes of the Cori
supercomputer with a global minibatch size of 128k: achieving a performance of
450 Tflop/s through enhancements to PyTorch. We demonstrate a Large Hadron
Collider (LHC) use-case with the C++ Sherpa simulator and achieve the
largest-scale posterior inference in a Turing-complete PPL.
# Blending-target Domain Adaptation by Adversarial Meta-Adaptation Networks 
(Unsupervised) Domain Adaptation (DA) seeks for classifying target instances
when solely provided with source labeled and target unlabeled examples for
training. Learning domain-invariant features helps to achieve this goal,
whereas it underpins unlabeled samples drawn from a single or multiple explicit
target domains (Multi-target DA). In this paper, we consider a more realistic
transfer scenario: our target domain is comprised of multiple sub-targets
implicitly blended with each other, so that learners could not identify which
sub-target each unlabeled sample belongs to. This Blending-target Domain
Adaptation (BTDA) scenario commonly appears in practice and threatens the
validities of most existing DA algorithms, due to the presence of domain gaps
and categorical misalignments among these hidden sub-targets.
  To reap the transfer performance gains in this new scenario, we propose
Adversarial Meta-Adaptation Network (AMEAN). AMEAN entails two adversarial
transfer learning processes. The first is a conventional adversarial transfer
to bridge our source and mixed target domains. To circumvent the intra-target
category misalignment, the second process presents as ``learning to adapt'': It
deploys an unsupervised meta-learner receiving target data and their ongoing
feature-learning feedbacks, to discover target clusters as our
``meta-sub-target'' domains. These meta-sub-targets auto-design our
meta-sub-target DA loss, which empirically eliminates the implicit category
mismatching in our mixed target. We evaluate AMEAN and a variety of DA
algorithms in three benchmarks under the BTDA setup. Empirical results show
that BTDA is a quite challenging transfer setup for most existing DA
algorithms, yet AMEAN significantly outperforms these state-of-the-art
baselines and effectively restrains the negative transfer effects in BTDA.
# The Price of Interpretability 
When quantitative models are used to support decision-making on complex and
important topics, understanding a model's ``reasoning'' can increase trust in
its predictions, expose hidden biases, or reduce vulnerability to adversarial
attacks. However, the concept of interpretability remains loosely defined and
application-specific. In this paper, we introduce a mathematical framework in
which machine learning models are constructed in a sequence of interpretable
steps. We show that for a variety of models, a natural choice of interpretable
steps recovers standard interpretability proxies (e.g., sparsity in linear
models). We then generalize these proxies to yield a parametrized family of
consistent measures of model interpretability. This formal definition allows us
to quantify the ``price'' of interpretability, i.e., the tradeoff with
predictive accuracy. We demonstrate practical algorithms to apply our framework
on real and synthetic datasets.
# On-Policy Robot Imitation Learning from a Converging Supervisor 
Existing on-policy imitation learning algorithms, such as DAgger, assume
access to a fixed supervisor. However, there are many settings where the
supervisor may converge during policy learning, such as a human performing a
novel task or an improving algorithmic controller. We formalize imitation
learning from a "converging supervisor" and provide sublinear static and
dynamic regret guarantees against the best policy in hindsight with labels from
the converged supervisor, even when labels during learning are only from
intermediate supervisors. We then show that this framework is closely connected
to a recent class of reinforcement learning (RL) algorithms known as dual
policy iteration (DPI), which alternate between training a reactive learner
with imitation learning and a model-based supervisor with data from the
learner. Experiments suggest that when this framework is applied with the
state-of-the-art deep model-based RL algorithm PETS as an improving supervisor,
it outperforms deep RL baselines on continuous control tasks and provides up to
an 80-fold speedup in policy evaluation.
# Multivariate-Information Adversarial Ensemble for Scalable Joint Distribution Matching 
A broad range of cross-$m$-domain generation researches boil down to matching
a joint distribution by deep generative models (DGMs). Hitherto algorithms
excel in pairwise domains while as $m$ increases, remain struggling to scale
themselves to fit a joint distribution. In this paper, we propose a
domain-scalable DGM, i.e., MMI-ALI for $m$-domain joint distribution matching.
As an $m$-domain ensemble model of ALIs \cite{dumoulin2016adversarially},
MMI-ALI is adversarially trained with maximizing Multivariate Mutual
Information (MMI) w.r.t. joint variables of each pair of domains and their
shared feature. The negative MMIs are upper bounded by a series of feasible
losses that provably lead to matching $m$-domain joint distributions. MMI-ALI
linearly scales as $m$ increases and thus, strikes a right balance between
efficacy and scalability. We evaluate MMI-ALI in diverse challenging $m$-domain
scenarios and verify its superiority.
# Generalized Control Functions via Variational Decoupling 
Causal estimation relies on separating the variation in the outcome due to
the confounders from that due to the treatment. To achieve this separation,
practitioners can use external sources of randomness that only influence the
treatment called instrumental variables (IVs). Traditional IV-methods rely on
structural assumptions that limit the effect that the confounders can have on
both outcome and treatment. To relax these assumptions we develop a new
estimator called the generalized control-function method (GCFN). GCFN's first
stage called variational decoupling (VDE) recovers the residual variation in
the treatment given the IV. In the second stage, GCFN regresses the outcome on
the treatment and residual variation to compute the causal effect. We evaluate
GCFN on simulated data and on recovering the causal effect of slave export on
community trust. We show how VDE can help unify IV-estimators and
non-IV-estimators.
# Quantifying Transparency of Machine Learning Systems through Analysis of Contributions 
Increased adoption and deployment of machine learning (ML) models into
business, healthcare and other organisational processes, will result in a
growing disconnect between the engineers and researchers who developed the
models and the model's users and other stakeholders, such as regulators or
auditors. This disconnect is inevitable, as models begin to be used over a
number of years or are shared among third parties through user communities or
via commercial marketplaces, and it will become increasingly difficult for
users to maintain ongoing insight into the suitability of the parties who
created the model, or the data that was used to train it. This could become
problematic, particularly where regulations change and once-acceptable
standards become outdated, or where data sources are discredited, perhaps
judged to be biased or corrupted, either deliberately or unwittingly. In this
paper we present a method for arriving at a quantifiable metric capable of
ranking the transparency of the process pipelines used to generate ML models
and other data assets, such that users, auditors and other stakeholders can
gain confidence that they will be able to validate and trust the data sources
and human contributors in the systems that they rely on for their business
operations. The methodology for calculating the transparency metric, and the
type of criteria that could be used to make judgements on the visibility of
contributions to systems are explained and illustrated through an example
scenario.
# Physics Informed Extreme Learning Machine (PIELM) -- A rapid method for the numerical solution of partial differential equations 
There has been rapid progress recently on the application of deep networks to
the solution of partial differential equations, collectively labelled as
Physics Informed Neural Networks (PINNs). In this paper, we develop Physics
Informed Extreme Learning Machine (PIELM), a rapid version of PINNs which can
be applied to stationary and time dependent linear partial differential
equations. We demonstrate that PIELM matches or exceeds the accuracy of PINNs
on a range of problems. We also discuss the limitations of neural network based
approaches, including our PIELM, in the solution of PDEs on large domains and
suggest an extension, a distributed version of our algorithm -{}- DPIELM. We
show that DPIELM produces excellent results comparable to conventional
numerical techniques in the solution of time-dependent problems. Collectively,
this work contributes towards making the use of neural networks in the solution
of partial differential equations in complex domains as a competitive
alternative to conventional discretization techniques.
# A Multi-Stage Clustering Framework for Automotive Radar Data 
Radar sensors provide a unique method for executing environmental perception
tasks towards autonomous driving. Especially their capability to perform well
in adverse weather conditions often makes them superior to other sensors such
as cameras or lidar. Nevertheless, the high sparsity and low dimensionality of
the commonly used detection data level is a major challenge for subsequent
signal processing. Therefore, the data points are often merged in order to form
larger entities from which more information can be gathered. The merging
process is often implemented in form of a clustering algorithm. This article
describes a novel approach for first filtering out static background data
before applying a twostage clustering approach. The two-stage clustering
follows the same paradigm as the idea for data association itself: First,
clustering what is ought to belong together in a low dimensional parameter
space, then, extracting additional features from the newly created clusters in
order to perform a final clustering step. Parameters are optimized for
filtering and both clustering steps. All techniques are assessed both
individually and as a whole in order to demonstrate their effectiveness. Final
results indicate clear benefits of the first two methods and also the cluster
merging process under specific circumstances.
# ShrinkML: End-to-End ASR Model Compression Using Reinforcement Learning 
End-to-end automatic speech recognition (ASR) models are increasingly large
and complex to achieve the best possible accuracy. In this paper, we build an
AutoML system that uses reinforcement learning (RL) to optimize the per-layer
compression ratios when applied to a state-of-the-art attention based
end-to-end ASR model composed of several LSTM layers. We use singular value
decomposition (SVD) low-rank matrix factorization as the compression method.
For our RL-based AutoML system, we focus on practical considerations such as
the choice of the reward/punishment functions, the formation of an effective
search space, and the creation of a representative but small data set for quick
evaluation between search steps. Finally, we present accuracy results on
LibriSpeech of the model compressed by our AutoML system, and we compare it to
manually-compressed models. Our results show that in the absence of retraining
our RL-based search is an effective and practical method to compress a
production-grade ASR system. When retraining is possible, we show that our
AutoML system can select better highly-compressed seed models compared to
manually hand-crafted rank selection, thus allowing for more compression than
previously possible.
# Data Efficient Reinforcement Learning for Legged Robots 
We present a model-based framework for robot locomotion that achieves walking
based on only 4.5 minutes (45,000 control steps) of data collected on a
quadruped robot. To accurately model the robot's dynamics over a long horizon,
we introduce a loss function that tracks the model's prediction over multiple
timesteps. We adapt model predictive control to account for planning latency,
which allows the learned model to be used for real time control. Additionally,
to ensure safe exploration during model learning, we embed prior knowledge of
leg trajectories into the action space. The resulting system achieves fast and
robust locomotion. Unlike model-free methods, which optimize for a particular
task, our planner can use the same learned dynamics for various tasks, simply
by changing the reward function. To the best of our knowledge, our approach is
more than an order of magnitude more sample efficient than current model-free
methods.
# General non-linear Bellman equations 
We consider a general class of non-linear Bellman equations. These open up a
design space of algorithms that have interesting properties, which has two
potential advantages. First, we can perhaps better model natural phenomena. For
instance, hyperbolic discounting has been proposed as a mathematical model that
matches human and animal data well, and can therefore be used to explain
preference orderings. We present a different mathematical model that matches
the same data, but that makes very different predictions under other
circumstances. Second, the larger design space can perhaps lead to algorithms
that perform better, similar to how discount factors are often used in practice
even when the true objective is undiscounted. We show that many of the
resulting Bellman operators still converge to a fixed point, and therefore that
the resulting algorithms are reasonable and inherit many beneficial properties
of their linear counterparts.
# TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications 
Ball trajectory data are one of the most fundamental and useful information
in the evaluation of players' performance and analysis of game strategies.
Although vision-based object tracking techniques have been developed to analyze
sport competition videos, it is still challenging to recognize and position a
high-speed and tiny ball accurately. In this paper, we develop a deep learning
network, called TrackNet, to track the tennis ball from broadcast videos in
which the ball images are small, blurry, and sometimes with afterimage tracks
or even invisible. The proposed heatmap-based deep learning network is trained
to not only recognize the ball image from a single frame but also learn flying
patterns from consecutive frames. TrackNet takes images with a size of
$640\times360$ to generate a detection heatmap from either a single frame or
several consecutive frames to position the ball and can achieve high precision
even on public domain videos. The network is evaluated on the video of the
men's singles final at the 2017 Summer Universiade, which is available on
YouTube. The precision, recall, and F1-measure of TrackNet reach $99.7\%$,
$97.3\%$, and $98.5\%$, respectively. To prevent overfitting, 9 additional
videos are partially labeled together with a subset from the previous dataset
to implement 10-fold cross-validation, and the precision, recall, and
F1-measure are $95.3\%$, $75.7\%$, and $84.3\%$, respectively. A conventional
image processing algorithm is also implemented to compare with TrackNet. Our
experiments indicate that TrackNet outperforms conventional method by a big
margin and achieves exceptional ball tracking performance. The dataset and demo
video are available at https://nol.cs.nctu.edu.tw/ndo3je6av9/.
# Policy-Gradient Algorithms Have No Guarantees of Convergence in Continuous Action and State Multi-Agent Settings 
We show by counterexample that policy-gradient algorithms have no guarantees
of even local convergence to Nash equilibria in continuous action and state
space multi-agent settings. To do so, we analyze gradient-play in $N$-player
general-sum linear quadratic games. In such games the state and action spaces
are continuous and the unique global Nash equilibrium can be found be solving
coupled Ricatti equations. Further, gradient-play in LQ games is equivalent to
multi-agent policy gradient. We first prove that the only critical point of the
gradient dynamics in these games is the unique global Nash equilibrium. We then
give sufficient conditions under which policy gradient will avoid the Nash
equilibrium, and generate a large number of general-sum linear quadratic games
that satisfy these conditions. The existence of such games indicates that one
of the most popular approaches to solving reinforcement learning problems in
the classic reinforcement learning setting has no guarantee of convergence in
multi-agent settings. Further, the ease with which we can generate these
counterexamples suggests that such situations are not mere edge cases and are
in fact quite common.
# Predicting Customer Call Intent by Analyzing Phone Call Transcripts based on CNN for Multi-Class Classification 
Auto dealerships receive thousands of calls daily from customers who are
interested in sales, service, vendors and jobseekers. With so many calls, it is
very important for auto dealers to understand the intent of these calls to
provide positive customer experiences that ensure customer satisfaction, deep
customer engagement to boost sales and revenue, and optimum allocation of
agents or customer service representatives across the business. In this paper,
we define the problem of customer phone call intent as a multi-class
classification problem stemming from the large database of recorded phone call
transcripts. To solve this problem, we develop a convolutional neural network
(CNN)-based supervised learning model to classify the customer calls into four
intent categories: sales, service, vendor and jobseeker. Experimental results
show that with the thrust of our scalable data labeling method to provide
sufficient training data, the CNN-based predictive model performs very well on
long text classification according to the quantitative metrics of F1-Score,
precision, recall, and accuracy.
# Expressive power of tensor-network factorizations for probabilistic modeling, with applications from hidden Markov models to quantum machine learning 
Tensor-network techniques have enjoyed outstanding success in physics, and
have recently attracted attention in machine learning, both as a tool for the
formulation of new learning algorithms and for enhancing the mathematical
understanding of existing methods. Inspired by these developments, and the
natural correspondence between tensor networks and probabilistic graphical
models, we provide a rigorous analysis of the expressive power of various
tensor-network factorizations of discrete multivariate probability
distributions. These factorizations include non-negative tensor-trains/MPS,
which are in correspondence with hidden Markov models, and Born machines, which
are naturally related to local quantum circuits. When used to model probability
distributions, they exhibit tractable likelihoods and admit efficient learning
algorithms. Interestingly, we prove that there exist probability distributions
for which there are unbounded separations between the resource requirements of
some of these tensor-network factorizations. Particularly surprising is the
fact that using complex instead of real tensors can lead to an arbitrarily
large reduction in the number of parameters of the network. Additionally, we
introduce locally purified states (LPS), a new factorization inspired by
techniques for the simulation of quantum systems, with provably better
expressive power than all other representations considered. The ramifications
of this result are explored through numerical experiments. Our findings imply
that LPS should be considered over hidden Markov models, and furthermore
provide guidelines for the design of local quantum circuits for probabilistic
modeling.
# Un Modelo Ontol\'ogico para el Gobierno Electr\'onico 
Decision making often requires information that must be Provided with the
rich data format. Addressing these new requirements appropriately makes it
necessary for government agencies to orchestrate large amounts of information
from different sources and formats, to be efficiently delivered through the
devices commonly used by people, such as computers, netbooks, tablets and
smartphones. To overcome these problems, a model is proposed for the conceptual
representation of the State's organizational units, seen as georeferenced
entities of Electronic Government, based on ontologies designed under the
principles of Linked Open Data, which allows the automatic extraction of
information through the machines, which supports the process of governmental
decision making and gives citizens full access to find and process through
mobile technologies.
# MigrationMiner: An Automated Detection Tool of Third-Party Java Library Migration at the Method Level 
In this paper we introduce, MigrationMiner, an automated tool that detects
code migrations performed between Java third-party library. Given a list of
open source projects, the tool detects potential library migration code changes
and collects the specific code fragments in which the developer replaces
methods from the retired library with methods from the new library. To support
the migration process, MigrationMiner collects the library documentation that
is associated with every method involved in the migration. We evaluate our tool
on a benchmark of manually validated library migrations. Results show that
MigrationMiner achieves an accuracy of 100%. A demo video of MigrationMiner is
available at https://youtu.be/sAlR1HNetXc.
# NeuType: A Simple and Effective Neural Network Approach for Predicting Missing Entity Type Information in Knowledge Bases 
Knowledge bases store information about the semantic types of entities, which
can be utilized in a range of information access tasks. This information,
however, is often incomplete, due to new entities emerging on a daily basis. We
address the task of automatically assigning types to entities in a knowledge
base from a type taxonomy. Specifically, we present two neural network
architectures, which take short entity descriptions and, optionally,
information about related entities as input. Using the DBpedia knowledge base
for experimental evaluation, we demonstrate that these simple architectures
yield significant improvements over the current state of the art.
# Informative Visual Storytelling with Cross-modal Rules 
Existing methods in the Visual Storytelling field often suffer from the
problem of generating general descriptions, while the image contains a lot of
meaningful contents remaining unnoticed. The failure of informative story
generation can be concluded to the model's incompetence of capturing enough
meaningful concepts. The categories of these concepts include entities,
attributes, actions, and events, which are in some cases crucial to grounded
storytelling. To solve this problem, we propose a method to mine the
cross-modal rules to help the model infer these informative concepts given
certain visual input. We first build the multimodal transactions by
concatenating the CNN activations and the word indices. Then we use the
association rule mining algorithm to mine the cross-modal rules, which will be
used for the concept inference. With the help of the cross-modal rules, the
generated stories are more grounded and informative. Besides, our proposed
method holds the advantages of interpretation, expandability, and
transferability, indicating potential for wider application. Finally, we
leverage these concepts in our encoder-decoder framework with the attention
mechanism. We conduct several experiments on the VIsual StoryTelling~(VIST)
dataset, the results of which demonstrate the effectiveness of our approach in
terms of both automatic metrics and human evaluation. Additional experiments
are also conducted showing that our mined cross-modal rules as additional
knowledge helps the model gain better performance when trained on a small
dataset.
# A Formal Axiomatization of Computation 
We introduce a set of axioms for the notion of computation, and show that P=
NP is not derivable from this set of axioms.
# Adaptive Weighting Depth-variant Deconvolution of Fluorescence Microscopy Images with Convolutional Neural Network 
Fluorescence microscopy plays an important role in biomedical research. The
depth-variant point spread function (PSF) of a fluorescence microscope produces
low-quality images especially in the out-of-focus regions of thick specimens.
Traditional deconvolution to restore the out-of-focus images is usually
insufficient since a depth-invariant PSF is assumed. This article aims at
handling fluorescence microscopy images by learning-based depth-variant PSF and
reducing artifacts. We propose adaptive weighting depth-variant deconvolution
(AWDVD) with defocus level prediction convolutional neural network (DelpNet) to
restore the out-of-focus images. Depth-variant PSFs of image patches can be
obtained by DelpNet and applied in the afterward deconvolution. AWDVD is
adopted for a whole image which is patch-wise deconvolved and appropriately
cropped before deconvolution. DelpNet achieves the accuracy of 98.2%, which
outperforms the best-ever one using the same microscopy dataset. Image patches
of 11 defocus levels after deconvolution are validated with maximum improvement
in the peak signal-to-noise ratio and structural similarity index of 6.6 dB and
11%, respectively. The adaptive weighting of the patch-wise deconvolved image
can eliminate patch boundary artifacts and improve deconvolved image quality.
The proposed method can accurately estimate depth-variant PSF and effectively
recover out-of-focus microscopy images. To our acknowledge, this is the first
study of handling out-of-focus microscopy images using learning-based
depth-variant PSF. Facing one of the most common blurs in fluorescence
microscopy, the novel method provides a practical technology to improve the
image quality.
# Travel Time Estimation without Road Networks: An Urban Morphological Layout Representation Approach 
Travel time estimation is a crucial task for not only personal travel
scheduling but also city planning. Previous methods focus on modeling toward
road segments or sub-paths, then summing up for a final prediction, which have
been recently replaced by deep neural models with end-to-end training. Usually,
these methods are based on explicit feature representations, including
spatio-temporal features, traffic states, etc. Here, we argue that the local
traffic condition is closely tied up with the land-use and built environment,
i.e., metro stations, arterial roads, intersections, commercial area,
residential area, and etc, yet the relation is time-varying and too complicated
to model explicitly and efficiently. Thus, this paper proposes an end-to-end
multi-task deep neural model, named Deep Image to Time (DeepI2T), to learn the
travel time mainly from the built environment images, a.k.a. the morphological
layout images, and showoff the new state-of-the-art performance on real-world
datasets in two cities. Moreover, our model is designed to tackle both
path-aware and path-blind scenarios in the testing phase. This work opens up
new opportunities of using the publicly available morphological layout images
as considerable information in multiple geography-related smart city
applications.
# Segway DRIVE Benchmark: Place Recognition and SLAM Data Collected by A Fleet of Delivery Robots 
Visual place recognition and simultaneous localization and mapping (SLAM)
have recently begun to be used in real-world autonomous navigation tasks like
food delivery. Existing datasets for SLAM research are often not representative
of in situ operations, leaving a gap between academic research and real-world
deployment. In response, this paper presents the Segway DRIVE benchmark, a
novel and challenging dataset suite collected by a fleet of Segway delivery
robots. Each robot is equipped with a global-shutter fisheye camera, a
consumer-grade IMU synced to the camera on chip, two low-cost wheel encoders,
and a removable high-precision lidar for generating reference solutions. As
they routinely carry out tasks in office buildings and shopping malls while
collecting data, the dataset spanning a year is characterized by planar
motions, moving pedestrians in scenes, and changing environment and lighting.
Such factors typically pose severe challenges and may lead to failures for SLAM
algorithms. Moreover, several metrics are proposed to evaluate metric place
recognition algorithms. With these metrics, sample SLAM and metric place
recognition methods were evaluated on this benchmark.
  The first release of our benchmark has hundreds of sequences, covering more
than 50 km of indoor floors. More data will be added as the robot fleet
continues to operate in real life. The benchmark is available at
http://drive.segwayrobotics.com/#/dataset/download.
# Deep Learning-Based Semantic Segmentation of Microscale Objects 
Accurate estimation of the positions and shapes of microscale objects is
crucial for automated imaging-guided manipulation using a non-contact technique
such as optical tweezers. Perception methods that use traditional computer
vision algorithms tend to fail when the manipulation environments are crowded.
In this paper, we present a deep learning model for semantic segmentation of
the images representing such environments. Our model successfully performs
segmentation with a high mean Intersection Over Union score of 0.91.
# Prediction of Soil Moisture Content Based On Satellite Data and Sequence-to-Sequence Networks 
The main objective of this study is to combine remote sensing and machine
learning to detect soil moisture content. Growing population and food
consumption has led to the need to improve agricultural yield and to reduce
wastage of natural resources. In this paper, we propose a neural network
architecture, based on recent work by the research community, that can make a
strong social impact and aid United Nations Sustainable Development Goal of
Zero Hunger. The main aims here are to: improve efficiency of water usage;
reduce dependence on irrigation; increase overall crop yield; minimise risk of
crop loss due to drought and extreme weather conditions. We achieve this by
applying satellite imagery, crop segmentation, soil classification and NDVI and
soil moisture prediction on satellite data, ground truth and climate data
records. By applying machine learning to sensor data and ground data, farm
management systems can evolve into a real time AI enabled platform that can
provide actionable recommendations and decision support tools to the farmers.
# RED: A ReRAM-based Deconvolution Accelerator 
Deconvolution has been widespread in neural networks. For example, it is
essential for performing unsupervised learning in generative adversarial
networks or constructing fully convolutional networks for semantic
segmentation. Resistive RAM (ReRAM)-based processing-in-memory architecture has
been widely explored in accelerating convolutional computation and demonstrates
good performance. Performing deconvolution on existing ReRAM-based accelerator
designs, however, suffers from long latency and high energy consumption because
deconvolutional computation includes not only convolution but also extra add-on
operations. To realize the more efficient execution for deconvolution, we
analyze its computation requirement and propose a ReRAM-based accelerator
design, namely, RED. More specific, RED integrates two orthogonal methods, the
pixel-wise mapping scheme for reducing redundancy caused by zero-inserting
operations and the zero-skipping data flow for increasing the computation
parallelism and therefore improving performance. Experimental evaluations show
that compared to the state-of-the-art ReRAM-based accelerator, RED can speed up
operation 3.69x~1.15x and reduce 8%~88.36% energy consumption.
# Deep learning in ultrasound imaging 
We consider deep learning strategies in ultrasound systems, from the
front-end to advanced applications. Our goal is to provide the reader with a
broad understanding of the possible impact of deep learning methodologies on
many aspects of ultrasound imaging. In particular, we discuss methods that lie
at the interface of signal acquisition and machine learning, exploiting both
data structure (e.g. sparsity in some domain) and data dimensionality (big
data) already at the raw radio-frequency channel stage. As some examples, we
outline efficient and effective deep learning solutions for adaptive
beamforming and adaptive spectral Doppler through artificial agents, learn
compressive encodings for color Doppler, and provide a framework for structured
signal recovery by learning fast approximations of iterative minimization
problems, with applications to clutter suppression and super-resolution
ultrasound. These emerging technologies may have considerable impact on
ultrasound imaging, showing promise across key components in the receive
processing chain.
# Financial Time Series Data Processing for Machine Learning 
This article studies the financial time series data processing for machine
learning. It introduces the most frequent scaling methods, then compares the
resulting stationarity and preservation of useful information for trend
forecasting. It proposes an empirical test based on the capability to learn
simple data relationship with simple models. It also speaks about the data
split method specific to time series, avoiding unwanted overfitting and
proposes various labelling for classification and regression.
# Faking and Discriminating the Navigation Data of a Micro Aerial Vehicle Using Quantum Generative Adversarial Networks 
We show that the Quantum Generative Adversarial Network (QGAN) paradigm can
be employed by an adversary to learn generating data that deceives the
monitoring of a Cyber-Physical System (CPS) and to perpetrate a covert attack.
As a test case, the ideas are elaborated considering the navigation data of a
Micro Aerial Vehicle (MAV). A concrete QGAN design is proposed to generate fake
MAV navigation data. Initially, the adversary is entirely ignorant about the
dynamics of the CPS, the strength of the approach from the point of view of the
bad guy. A design is also proposed to discriminate between genuine and fake MAV
navigation data. The designs combine classical optimization, qubit quantum
computing and photonic quantum computing. Using the PennyLane software
simulation, they are evaluated over a classical computing platform. We assess
the learning time and accuracy of the navigation data generator and
discriminator versus space complexity, i.e., the amount of quantum memory
needed to solve the problem.
# Global Aggregations of Local Explanations for Black Box models 
The decision-making process of many state-of-the-art machine learning models
is inherently inscrutable to the extent that it is impossible for a human to
interpret the model directly: they are black box models. This has led to a call
for research on explaining black box models, for which there are two main
approaches. Global explanations that aim to explain a model's decision making
process in general, and local explanations that aim to explain a single
prediction. Since it remains challenging to establish fidelity to black box
models in globally interpretable approximations, much attention is put on local
explanations. However, whether local explanations are able to reliably
represent the black box model and provide useful insights remains an open
question. We present Global Aggregations of Local Explanations (GALE) with the
objective to provide insights in a model's global decision making process.
Overall, our results reveal that the choice of aggregation matters. We find
that the global importance introduced by Local Interpretable Model-agnostic
Explanations (LIME) does not reliably represent the model's global behavior.
Our proposed aggregations are better able to represent how features affect the
model's predictions, and to provide global insights by identifying
distinguishing features.
# Uncovering Download Fraud Activities in Mobile App Markets 
Download fraud is a prevalent threat in mobile App markets, where fraudsters
manipulate the number of downloads of Apps via various cheating approaches.
Purchased fake downloads can mislead recommendation and search algorithms and
further lead to bad user experience in App markets. In this paper, we
investigate download fraud problem based on a company's App Market, which is
one of the most popular Android App markets. We release a honeypot App on the
App Market and purchase fake downloads from fraudster agents to track fraud
activities in the wild. Based on our interaction with the fraudsters, we
categorize download fraud activities into three types according to their
intentions: boosting front end downloads, optimizing App search ranking, and
enhancing user acquisition&retention rate. For the download fraud aimed at
optimizing App search ranking, we select, evaluate, and validate several
features in identifying fake downloads based on billions of download data. To
get a comprehensive understanding of download fraud, we further gather stances
of App marketers, fraudster agencies, and market operators on download fraud.
The followed analysis and suggestions shed light on the ways to mitigate
download fraud in App markets and other social platforms. To the best of our
knowledge, this is the first work that investigates the download fraud problem
in mobile App markets.
# MRI Super-Resolution with Ensemble Learning and Complementary Priors 
Magnetic resonance imaging (MRI) is a widely used medical imaging modality.
However, due to the limitations in hardware, scan time, and throughput, it is
often clinically challenging to obtain high-quality MR images. The
super-resolution approach is potentially promising to improve MR image quality
without any hardware upgrade. In this paper, we propose an ensemble learning
and deep learning framework for MR image super-resolution. In our study, we
first enlarged low resolution images using 5 commonly used super-resolution
algorithms and obtained differentially enlarged image datasets with
complementary priors. Then, a generative adversarial network (GAN) is trained
with each dataset to generate super-resolution MR images. Finally, a
convolutional neural network is used for ensemble learning that synergizes the
outputs of GANs into the final MR super-resolution images. According to our
results, the ensemble learning results outcome any one of GAN outputs. Compared
with some state-of-the-art deep learning-based super-resolution methods, our
approach is advantageous in suppressing artifacts and keeping more image
details.
# Estimating location parameters in entangled single-sample distributions 
We consider the problem of estimating the common mean of independently
sampled data, where samples are drawn in a possibly non-identical manner from
symmetric, unimodal distributions with a common mean. This generalizes the
setting of Gaussian mixture modeling, since the number of distinct mixture
components may diverge with the number of observations. We propose an estimator
that adapts to the level of heterogeneity in the data, achieving
near-optimality in both the i.i.d. setting and some heterogeneous settings,
where the fraction of ``low-noise'' points is as small as $\frac{\log n}{n}$.
Our estimator is a hybrid of the modal interval, shorth, and median estimators
from classical statistics; however, the key technical contributions rely on
novel empirical process theory results that we derive for independent but
non-i.i.d. data. In the multivariate setting, we generalize our theory to mean
estimation for mixtures of radially symmetric distributions, and derive minimax
lower bounds on the expected error of any estimator that is agnostic to the
scales of individual data points. Finally, we describe an extension of our
estimators applicable to linear regression. In the multivariate mean estimation
and regression settings, we present computationally feasible versions of our
estimators that run in time polynomial in the number of data points.
# Takens-inspired neuromorphic processor: a downsizing tool for random recurrent neural networks via feature extraction 
We describe a new technique which minimizes the amount of neurons in the
hidden layer of a random recurrent neural network (rRNN) for time series
prediction. Merging Takens-based attractor reconstruction methods with machine
learning, we identify a mechanism for feature extraction that can be leveraged
to lower the network size. We obtain criteria specific to the particular
prediction task and derive the scaling law of the prediction error. The
consequences of our theory are demonstrated by designing a Takens-inspired
hybrid processor, which extends a rRNN with a priori designed delay external
memory. Our hybrid architecture is therefore designed including both, real and
virtual nodes. Via this symbiosis, we show performance of the hybrid processor
by stabilizing an arrhythmic neural model. Thanks to our obtained design rules,
we can reduce the stabilizing neural network's size by a factor of 15 with
respect to a standard system.
# Precision annealing Monte Carlo methods for statistical data assimilation and machine learning 
In statistical data assimilation (SDA) and supervised machine learning (ML),
we wish to transfer information from observations to a model of the processes
underlying those observations. For SDA, the model consists of a set of
differential equations that describe the dynamics of a physical system. For ML,
the model is usually constructed using other strategies. In this paper, we
develop a systematic formulation based on Monte Carlo sampling to achieve such
information transfer. Following the derivation of an appropriate target
distribution, we present the formulation based on the standard
Metropolis-Hasting (MH) procedure and the Hamiltonian Monte Carlo (HMC) method
for performing the high dimensional integrals that appear. To the extensive
literature on MH and HMC, we add (1) an annealing method using a hyperparameter
that governs the precision of the model to identify and explore the highest
probability regions of phase space dominating those integrals, and (2) a
strategy for initializing the state space search. The efficacy of the proposed
formulation is demonstrated using a nonlinear dynamical model with chaotic
solutions widely used in geophysics.
# ReLU Networks as Surrogate Models in Mixed-Integer Linear Programs 
We consider the embedding of piecewise-linear deep neural networks (ReLU
networks) as surrogate models in mixed-integer linear programming (MILP)
problems. A MILP formulation of ReLU networks has recently been applied by many
authors to probe for various model properties subject to input bounds. The
formulation is obtained by programming each ReLU operator with a binary
variable and applying the big-M method. The efficiency of the formulation
hinges on the tightness of the bounds defined by the big-M values. When ReLU
networks are embedded in a larger optimization problem, the presence of output
bounds can be exploited in bound tightening. To this end, we devise and study
several bound tightening procedures that consider both input and output bounds.
Our numerical results show that bound tightening may reduce solution times
considerably, and that small-sized ReLU networks are suitable as surrogate
models in mixed-integer linear programs.
# XGBoostLSS -- An extension of XGBoost to probabilistic forecasting 
We propose a new framework of XGBoost that predicts the entire conditional
distribution of a univariate response variable. In particular, XGBoostLSS
models all moments of a parametric distribution, i.e., mean, location, scale
and shape (LSS), instead of the conditional mean only. Chosing from a wide
range of continuous, discrete and mixed discrete-continuous distribution,
modeling and predicting the entire conditional distribution greatly enhances
the flexibility of XGBoost, as it allows to gain additional insight into the
data generating process, as well as to create probabilistic forecasts from
which prediction intervals and quantiles of interest can be derived. We present
both a simulation study and real world examples that highlight the benefits of
our approach.
# TEALS: Time-aware Text Embedding Approach to Leverage Subgraphs 
Given a graph over which the contagions (e.g. virus, gossip) propagate,
leveraging subgraphs with highly correlated nodes is beneficial to many
applications. Yet, challenges abound. First, the propagation pattern between a
pair of nodes may change in various temporal dimensions. Second, not always the
same contagion is propagated. Hence, state-of-the-art text mining approaches
ranging from similarity measures to topic-modeling cannot use the textual
contents to compute the weights between the nodes. Third, the word-word
co-occurrence patterns may differ in various temporal dimensions, which
increases the difficulty to employ current word embedding approaches. We argue
that inseparable multi-aspect temporal collaborations are inevitably needed to
better calculate the correlation metrics in dynamical processes. In this work,
we showcase a sophisticated framework that on the one hand, integrates a neural
network based time-aware word embedding component that can collectively
construct the word vectors through an assembly of infinite latent temporal
facets, and on the other hand, uses an elaborate generative model to compute
the edge weights through heterogeneous temporal attributes. After computing the
intra-nodes weights, we utilize our Max-Heap Graph cutting algorithm to exploit
subgraphs. We then validate our model through comprehensive experiments on
real-world propagation data. The results show that the knowledge gained from
the versatile temporal dynamics is not only indispensable for word embedding
approaches but also plays a significant role in the understanding of the
propagation behaviors. Finally, we demonstrate that compared with other rivals,
our model can dominantly exploit the subgraphs with highly coordinated nodes.
# Volume Doubling Condition and a Local Poincar\'e Inequality on Unweighted Random Geometric Graphs 
The aim of this paper is to establish two fundamental measure-metric
properties of particular random geometric graphs. We consider
$\varepsilon$-neighborhood graphs whose vertices are drawn independently and
identically distributed from a common distribution defined on a regular
submanifold of $\mathbb{R}^K$. We show that a volume doubling condition (VD)
and local Poincar\'e inequality (LPI) hold for the random geometric graph (with
high probability, and uniformly over all shortest path distance balls in a
certain radius range) under suitable regularity conditions of the underlying
submanifold and the sampling distribution.
# Composable Core-sets for Determinant Maximization: A Simple Near-Optimal Algorithm 
``Composable core-sets'' are an efficient framework for solving optimization
problems in massive data models. In this work, we consider efficient
construction of composable core-sets for the determinant maximization problem.
This can also be cast as the MAP inference task for determinantal point
processes, that have recently gained a lot of interest for modeling diversity
and fairness. The problem was recently studied in [IMOR'18], where they
designed composable core-sets with the optimal approximation bound of $\tilde
O(k)^k$. On the other hand, the more practical Greedy algorithm has been
previously used in similar contexts. In this work, first we provide a
theoretical approximation guarantee of $O(C^{k^2})$ for the Greedy algorithm in
the context of composable core-sets; Further, we propose to use a Local Search
based algorithm that while being still practical, achieves a nearly optimal
approximation bound of $O(k)^{2k}$; Finally, we implement all three algorithms
and show the effectiveness of our proposed algorithm on standard data sets.
# IRNet: A General Purpose Deep Residual Regression Framework for Materials Discovery 
Materials discovery is crucial for making scientific advances in many
domains. Collections of data from experiments and first-principle computations
have spurred interest in applying machine learning methods to create predictive
models capable of mapping from composition and crystal structures to materials
properties. Generally, these are regression problems with the input being a 1D
vector composed of numerical attributes representing the material composition
and/or crystal structure. While neural networks consisting of fully connected
layers have been applied to such problems, their performance often suffers from
the vanishing gradient problem when network depth is increased. In this paper,
we study and propose design principles for building deep regression networks
composed of fully connected layers with numerical vectors as input. We
introduce a novel deep regression network with individual residual learning,
IRNet, that places shortcut connections after each layer so that each layer
learns the residual mapping between its output and input. We use the problem of
learning properties of inorganic materials from numerical attributes derived
from material composition and/or crystal structure to compare IRNet's
performance against that of other machine learning techniques. Using multiple
datasets from the Open Quantum Materials Database (OQMD) and Materials Project
for training and evaluation, we show that IRNet provides significantly better
prediction performance than the state-of-the-art machine learning approaches
currently used by domain scientists. We also show that IRNet's use of
individual residual learning leads to better convergence during the training
phase than when shortcut connections are between multi-layer stacks while
maintaining the same number of parameters.
# Deep Learning based Wireless Resource Allocation with Application to Vehicular Networks 
It has been a long-held belief that judicious resource allocation is critical
to mitigating interference, improving network efficiency, and ultimately
optimizing wireless communication performance. The traditional wisdom is to
explicitly formulate resource allocation as an optimization problem and then
exploit mathematical programming to solve the problem to a certain level of
optimality. Nonetheless, as wireless networks become increasingly diverse and
complex, e.g., the high-mobility vehicular networks, the current design
methodologies face significant challenges and thus call for rethinking of the
traditional design philosophy. Meanwhile, deep learning, with many success
stories in various disciplines, represents a promising alternative due to its
remarkable power to leverage data for problem solving. In this paper, we
discuss the key motivations and roadblocks of using deep learning for wireless
resource allocation with application to vehicular networks. We review major
recent studies that mobilize the deep learning philosophy in wireless resource
allocation and achieve impressive results. We first discuss deep learning
assisted optimization for resource allocation. We then highlight the deep
reinforcement learning approach to address resource allocation problems that
are difficult to handle in the traditional optimization framework. We also
identify some research directions that deserve further investigation.
# Smart Grid Cyber Attacks Detection using Supervised Learning and Heuristic Feature Selection 
False Data Injection (FDI) attacks are a common form of Cyber-attack
targetting smart grids. Detection of stealthy FDI attacks is impossible by the
current bad data detection systems. Machine learning is one of the alternative
methods proposed to detect FDI attacks. This paper analyzes three various
supervised learning techniques, each to be used with three different feature
selection (FS) techniques. These methods are tested on the IEEE 14-bus, 57-bus,
and 118-bus systems for evaluation of versatility. Accuracy of the
classification is used as the main evaluation method for each detection
technique. Simulation study clarify the supervised learning combined with
heuristic FS methods result in an improved performance of the classification
algorithms for FDI attack detection.
# Search-Based Serving Architecture of Embeddings-Based Recommendations 
Over the past 10 years, many recommendation techniques have been based on
embedding users and items in latent vector spaces, where the inner product of a
(user,item) pair of vectors represents the predicted affinity of the user to
the item. A wealth of literature has focused on the various modeling approaches
that result in embeddings, and has compared their quality metrics, learning
complexity, etc. However, much less attention has been devoted to the issues
surrounding productization of an embeddings-based high throughput, low latency
recommender system. In particular, how the system might keep up with the
changing embeddings as new models are learnt. This paper describes a reference
architecture of a high-throughput, large scale recommendation service which
leverages a search engine as its runtime core. We describe how the search index
and the query builder adapt to changes in the embeddings, which often happen at
a different cadence than index builds. We provide solutions for both id-based
and feature-based embeddings, as well as for batch indexing and incremental
indexing setups. The described system is at the core of a Web content discovery
service that serves tens of billions recommendations per day in response to
billions of user requests.
# QUOTIENT: Two-Party Secure Neural Network Training and Prediction 
Recently, there has been a wealth of effort devoted to the design of secure
protocols for machine learning tasks. Much of this is aimed at enabling secure
prediction from highly-accurate Deep Neural Networks (DNNs). However, as DNNs
are trained on data, a key question is how such models can be also trained
securely. The few prior works on secure DNN training have focused either on
designing custom protocols for existing training algorithms, or on developing
tailored training algorithms and then applying generic secure protocols. In
this work, we investigate the advantages of designing training algorithms
alongside a novel secure protocol, incorporating optimizations on both fronts.
We present QUOTIENT, a new method for discretized training of DNNs, along with
a customized secure two-party protocol for it. QUOTIENT incorporates key
components of state-of-the-art DNN training such as layer normalization and
adaptive gradient methods, and improves upon the state-of-the-art in DNN
training in two-party computation. Compared to prior work, we obtain an
improvement of 50X in WAN time and 6% in absolute accuracy.
# Unbiased estimators for random design regression 
In linear regression we wish to estimate the optimum linear least squares
predictor for a distribution over d-dimensional input points and real-valued
responses, based on a small sample. Under standard random design analysis,
where the sample is drawn i.i.d. from the input distribution, the least squares
solution for that sample can be viewed as the natural estimator of the optimum.
Unfortunately, this estimator almost always incurs an undesirable bias coming
from the randomness of the input points. In this paper we show that it is
possible to draw a non-i.i.d. sample of input points such that, regardless of
the response model, the least squares solution is an unbiased estimator of the
optimum. Moreover, this sample can be produced efficiently by augmenting a
previously drawn i.i.d. sample with an additional set of d points drawn jointly
from the input distribution rescaled by the squared volume spanned by the
points. Motivated by this, we develop a theoretical framework for studying
volume-rescaled sampling, and in the process prove a number of new matrix
expectation identities. We use them to show that for any input distribution and
$\epsilon>0$ there is a random design consisting of $O(d\log d+ d/\epsilon)$
points from which an unbiased estimator can be constructed whose square loss
over the entire distribution is with high probability bounded by $1+\epsilon$
times the loss of the optimum. We provide efficient algorithms for generating
such unbiased estimators in a number of practical settings and support our
claims experimentally.
# Deep splitting method for parabolic PDEs 
In this paper we introduce a numerical method for parabolic PDEs that
combines operator splitting with deep learning. It divides the PDE
approximation problem into a sequence of separate learning problems. Since the
computational graph for each of the subproblems is comparatively small, the
approach can handle extremely high-dimensional PDEs. We test the method on
different examples from physics, stochastic control, and mathematical finance.
In all cases, it yields very good results in up to 10,000 dimensions with short
run times.
# Non-Invasive MGMT Status Prediction in GBM Cancer Using Magnetic 
Background and aim: This study aimed to predict methylation status of the O-6
methyl guanine-DNA methyl transferase (MGMT) gene promoter status by using MRI
radiomics features, as well as univariate and multivariate analysis.
  Material and Methods: Eighty-two patients who had a MGMT methylation status
were include in this study. Tumor were manually segmented in the four regions
of MR images, a) whole tumor, b) active/enhanced region, c) necrotic regions
and d) edema regions (E). About seven thousand radiomics features were
extracted for each patient. Feature selection and classifier were used to
predict MGMT status through different machine learning algorithms. The area
under the curve (AUC) of receiver operating characteristic (ROC) curve was used
for model evaluations.
  Results: Regarding univariate analysis, the Inverse Variance feature from
gray level co-occurrence matrix (GLCM) in Whole Tumor segment with 4.5 mm Sigma
of Laplacian of Gaussian filter with AUC: 0.71 (p-value: 0.002) was found to be
the best predictor. For multivariate analysis, the decision tree classifier
with Select from Model feature selector and LOG filter in Edema region had the
highest performance (AUC: 0.78), followed by Ada Boost classifier with Select
from Model feature selector and LOG filter in Edema region (AUC: 0.74).
  Conclusion: This study showed that radiomics using machine learning
algorithms is a feasible, noninvasive approach to predict MGMT methylation
status in GBM cancer patients
  Keywords: Radiomics, Radiogenomics, GBM, MRI, MGMT
# DeepAcid: Classification of macromolecule type based on sequences of amino acids 
The study of the amino acid sequence is vital in life sciences. In this
paper, we are using deep learning to solve macromolecule classification problem
using amino acids. Deep learning has emerged as a strong and efficient
framework that can be applied to a broad spectrum of complex learning problems
which were difficult to solve using traditional machine learning techniques in
the past. We are using word embedding from NLP to represent the amino acid
sequence as vectors. We are using different deep learning model for
classification of macromolecules like CNN, LSTM, and GRU. Convolution neural
network can extract features from amino acid sequences which are represented by
vectors. The extracted features will be feed to a different type of model to
train a robust classifier. our results show that Word2vec as embedding combine
with VGG-16 has better performance than LSTM and GRU. our approach gets an
error rate of 1.5%. Code is available at https://github.com/say2sarwar/DeepAcid
# Towards Explainable Music Emotion Recognition: The Route via Mid-level Features 
Emotional aspects play an important part in our interaction with music.
However, modelling these aspects in MIR systems have been notoriously
challenging since emotion is an inherently abstract and subjective experience,
thus making it difficult to quantify or predict in the first place, and to make
sense of the predictions in the next. In an attempt to create a model that can
give a musically meaningful and intuitive explanation for its predictions, we
propose a VGG-style deep neural network that learns to predict emotional
characteristics of a musical piece together with (and based on)
human-interpretable, mid-level perceptual features. We compare this to
predicting emotion directly with an identical network that does not take into
account the mid-level features and observe that the loss in predictive
performance of going through the mid-level features is surprisingly low, on
average. The design of our network allows us to visualize the effects of
perceptual features on individual emotion predictions, and we argue that the
small loss in performance in going through the mid-level features is justified
by the gain in explainability of the predictions.
# A New Approach to Distributed Hypothesis Testing and Non-Bayesian 
We study a setting where a group of agents, each receiving partially
informative private signals, seek to collaboratively learn the true underlying
state of the world (from a finite set of hypotheses) that generates their joint
observation profiles. To solve this problem, we propose a distributed learning
rule that differs fundamentally from existing approaches, in that it does not
employ any form of "belief-averaging". Instead, agents update their beliefs
based on a min-rule. Under standard assumptions on the observation model and
the network structure, we establish that each agent learns the truth
asymptotically almost surely. As our main contribution, we prove that with
probability 1, each false hypothesis is ruled out by every agent exponentially
fast at a network-independent rate that is strictly larger than existing rates.
We then develop a computationally-efficient variant of our learning rule that
is provably resilient to agents who do not behave as expected (as represented
by a Byzantine adversary model) and deliberately try to spread misinformation.
# Feature-Based Image Clustering and Segmentation Using Wavelets 
Pixel intensity is a widely used feature for clustering and segmentation
algorithms, the resulting segmentation using only intensity values might suffer
from noises and lack of spatial context information. Wavelet transform is often
used for image denoising and classification. We proposed a novel method to
incorporate Wavelet features in segmentation and clustering algorithms. The
conventional K-means, Fuzzy c-means (FCM), and Active contour without edges
(ACWE) algorithms were modified to adapt Wavelet features, leading to robust
clustering/segmentation algorithms. A weighting parameter to control the weight
of low-frequency sub-band information was also introduced. The new algorithms
showed the capability to converge to different segmentation results based on
the frequency information derived from the Wavelet sub-bands.
# Contraction Clustering (RASTER): A Very Fast Big Data Algorithm for Sequential and Parallel Density-Based Clustering in Linear Time, Constant Memory, and a Single Pass 
Clustering is an essential data mining tool for analyzing and grouping
similar objects. In big data applications, however, many clustering algorithms
are infeasible due to their high memory requirements and/or unfavorable runtime
complexity. In contrast, Contraction Clustering (RASTER) is a single-pass
algorithm for identifying density-based clusters with linear time complexity.
Due to its favorable runtime and the fact that its memory requirements are
constant, this algorithm is highly suitable for big data applications where the
amount of data to be processed is huge. It consists of two steps: (1) a
contraction step which projects objects onto tiles and (2) an agglomeration
step which groups tiles into clusters. This algorithm is extremely fast in both
sequential and parallel execution. In single-threaded execution on a
contemporary workstation, an implementation in Rust processes a batch of 500
million points with 1 million clusters in less than 50 seconds. The speedup due
to parallelization is significant, amounting to a factor of around 4 on an
8-core machine.
# Qualitative Benchmarking of Deep Learning Hardware and Frameworks: Review and Tutorial 
Previous survey papers offer knowledge of deep learning hardware devices and
software frameworks. This paper introduces benchmarking principles, surveys
machine learning devices including GPUs, FPGAs, and ASICs, and reviews deep
learning software frameworks. It also reviews these technologies with respect
to benchmarking from the angles of our 7-metric approach to frameworks and
12-metric approach to hardware platforms. After reading the paper, the audience
will understand seven benchmarking principles, generally know that differential
characteristics of mainstream AI devices, qualitatively compare deep learning
hardware through our 12-metric approach for benchmarking hardware, and read
benchmarking results of 16 deep learning frameworks via our 7-metric set for
benchmarking frameworks.
# Smart Households Demand Response Management with Micro Grid 
Nowadays the emerging smart grid technology opens up the possibility of
two-way communication between customers and energy utilities. Demand Response
Management (DRM) offers the promise of saving money for commercial customers
and households while helps utilities operate more efficiently. In this paper,
an Incentive-based Demand Response Optimization (IDRO) model is proposed to
efficiently schedule household appliances for minimum usage during peak hours.
The proposed method is a multi-objective optimization technique based on
Nonlinear Auto-Regressive Neural Network (NAR-NN) which considers energy
provided by the utility and rooftop installed photovoltaic (PV) system. The
proposed method is tested and verified using 300 case studies (household). Data
analysis for a period of one year shows a noticeable improvement in power
factor and customers bill.
# FortuneTeller: Predicting Microarchitectural Attacks via Unsupervised Deep Learning 
The growing security threat of microarchitectural attacks underlines the
importance of robust security sensors and detection mechanisms at the hardware
level. While there are studies on runtime detection of cache attacks, a generic
model to consider the broad range of existing and future attacks is missing.
Unfortunately, previous approaches only consider either a single attack
variant, e.g. Prime+Probe, or specific victim applications such as
cryptographic implementations. Furthermore, the state-of-the art anomaly
detection methods are based on coarse-grained statistical models, which are not
successful to detect anomalies in a large-scale real world systems. Thanks to
the memory capability of advanced Recurrent Neural Networks (RNNs) algorithms,
both short and long term dependencies can be learned more accurately.
Therefore, we propose FortuneTeller, which for the first time leverages the
superiority of RNNs to learn complex execution patterns and detects unseen
microarchitectural attacks in real world systems. FortuneTeller models benign
workload pattern from a microarchitectural standpoint in an unsupervised
fashion, and then, it predicts how upcoming benign executions are supposed to
behave. Potential attacks and malicious behaviors will be detected
automatically, when there is a discrepancy between the predicted execution
pattern and the runtime observation. We implement FortuneTeller based on the
available hardware performance counters on Intel processors and it is trained
with 10 million samples obtained from benign applications. For the first time,
the latest attacks such as Meltdown, Spectre, Rowhammer and Zombieload are
detected with one trained model and without observing these attacks during the
training. We show that FortuneTeller achieves F-score of 0.9970.
# Robust Guarantees for Perception-Based Control 
Motivated by vision based control of autonomous vehicles, we consider the
problem of controlling a known linear dynamical system for which partial state
information, such as vehicle position, can only be extracted from
high-dimensional data, such as an image. Our approach is to learn a perception
map from high-dimensional data to partial-state observation and its
corresponding error profile, and then design a robust controller. We show that
under suitable smoothness assumptions on the perception map and generative
model relating state to high-dimensional data, an affine error model is
sufficiently rich to capture all possible error profiles, and can further be
learned via a robust regression problem. We then show how to integrate the
learned perception map and error model into a novel robust control synthesis
procedure, and prove that the resulting perception and control loop has
favorable generalization properties. Finally, we illustrate the usefulness of
our approach on a synthetic example and on the self-driving car simulation
platform CARLA.
# Incorporating Query Term Independence Assumption for Efficient Retrieval and Ranking using Deep Neural Networks 
Classical information retrieval (IR) methods, such as query likelihood and
BM25, score documents independently w.r.t. each query term, and then accumulate
the scores. Assuming query term independence allows precomputing term-document
scores using these models---which can be combined with specialized data
structures, such as inverted index, for efficient retrieval. Deep neural IR
models, in contrast, compare the whole query to the document and are,
therefore, typically employed only for late stage re-ranking. We incorporate
query term independence assumption into three state-of-the-art neural IR
models: BERT, Duet, and CKNRM---and evaluate their performance on a passage
ranking task. Surprisingly, we observe no significant loss in result quality
for Duet and CKNRM---and a small degradation in the case of BERT. However, by
operating on each query term independently, these otherwise computationally
intensive models become amenable to offline precomputation---dramatically
reducing the cost of query evaluations employing state-of-the-art neural
ranking models. This strategy makes it practical to use deep models for
retrieval from large collections---and not restrict their usage to late stage
re-ranking.
# Identifying Missing Component in the Bechdel Test Using Principal Component Analysis Method 
A lot has been said and discussed regarding the rationale and significance of
the Bechdel Score. It became a digital sensation in 2013 when Swedish cinemas
began to showcase the Bechdel test score of a film alongside its rating. The
test has drawn criticism from experts and the film fraternity regarding its use
to rate the female presence in a movie. The pundits believe that the score is
too simplified and the underlying criteria of a film to pass the test must
include 1) at least two women, 2) who have at least one dialogue, 3) about
something other than a man, is egregious. In this research, we have considered
a few more parameters which highlight how we represent females in film, like
the number of female dialogues in a movie, dialogue genre, and part of speech
tags in the dialogue. The parameters were missing in the existing criteria to
calculate the Bechdel score. The research aims to analyze 342 movies scripts to
test a hypothesis if these extra parameters, above with the current Bechdel
criteria, are significant in calculating the female representation score. The
result of the Principal Component Analysis method concludes that the female
dialogue content is a key component and should be considered while measuring
the representation of women in a work of fiction.
# Neural Networks on Groups 
Recent work on neural networks has shown that allowing them to build internal
representations of data not restricted to $\mathbb{R}^n$ can provide
significant improvements in performance. The success of Graph Neural Networks,
Convolutional Kernel Networks, and Fourier Neural Networks among other methods
have demonstrated the clear value of applying abstract mathematics to the
design of neural networks. The theory of neural networks has not kept up
however, and the relevant theoretical results (when they exist at all) have
been proven on a case-by-case basis without a general theory. The process of
deriving new theoretical backing for each new type of network has become a
bottleneck to understanding and validating new approaches.
  In this paper we extend the concept of neural networks to general groups and
prove that neural networks with a single hidden layer and a bounded
non-constant activation function can approximate any $L^p$ function defined
over a locally compact Abelian group. This framework and universal
approximation theorem encompass all of the aforementioned contexts. We also
derive important corollaries and extensions with minor modification, including
the case for approximating continuous functions on a compact subset, neural
networks with ReLU activation functions on a linearly bi-ordered group, and
neural networks with affine transformations on a vector space. Our work also
obtains as special cases the recent theorems of Qi et al. [2017], Sennai et al.
[2019], Keriven and Peyre [2019], and Maron et al. [2019].
