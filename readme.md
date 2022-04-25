# Adverserial Architecture for Source-agnostic Learning

This repository contains implementation of a modified version of the method proposed at (the following paper)[http://sleep.csail.mit.edu/files/rfsleep-paper.pdf]:

Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture
Mingmin Zhao, Shichao Yue, Dina Katabi, Tommi Jaakkola, Matt Bianchi
International Conference on Machine Learning (ICML’17)

Briefly, the paper introduces a model for predicting sleep state (i.e. awake, light sleep, deep sleep, and rapid eye movement) based on the radio frequencies (RFs) that ones body reflects during the sleep. RFs are emitted in the room by a device and data is gathered using sensors that measure the RFs that ones body reflects during the sleep. An AI model then interprets these sensor data and predicts the state of the sleep.

The challenge is that the RFs are highly dependent on the measurment environment (e.g. room) and the sleeping individual. This, then, becomes a multi-domain adaptation probem where each domain consists of the environment and the individual and has a different data distribution. Therefore, the domain-specific extraneous data should be ignored in the input to be able to make robust predictions. This is performed using an adverserial architecutre. Please see the paper for more details.

## My Implementation

Similar to the original architecture proposed in the above paper, the architecture implemented in this repository consists of three components: **cncoder**, **predictor**, and **discriminator**. The main difference between this implementation and the original one is in the encoder compnenet, where the original paper uses a CNN-RNN architecture but I use a CNN architecture. In other words, the model implemented here does not consider the past data when predicting the sleep state at a given time point. Another difference is that a smaller CNN model is used here compared to the original paper, which uses a ResNet with 24 convolutional layers. These changes are made due to lack of data to train large models.

Points I would like to mention about my submission are as follows:
- All functions for building and training the classification model are encapsulated in the class **GAN** in the *model.py* file. The data processing and hyperparameter search codes are in the *main.py* file.
- As in the provided paper, i.e. *"Learning Sleep Stages from Radio Signals: A Conditional Adversarial Architecture"*, I have implemented three components using different functions **Encoder**, **Predictor**, and **Discriminator** using *Tensorflow*. This compartmentalization is useful for learning as described later. The function **FullModel** puts the components together as shown in figure 1(b) in the paper.
- To implement the learning algorithm (Algorithm 1 in the paper), which consisted of three consequitive steps for learning different components, I used Tensorflow **GradientTape** and a custom loss function (equation (3) in the paper). When updating a particular component's weights in each iteration (a batch), other components are freezed using the components' *trainable* attributes.
- The model for the **Encoder** component is a ResNet implementation that I have modified. The important modifications include:
  + Using 9 convulational layers instead of 34 due to three reasons:  1) my computer's memory could not handle such a large network, 2) training time was very large for larger networks, and 3) the given dataset is relatively small and might result in overfitting if used for training a large network. This reduction is done by removing the identity blocks, so that the input and output dimensions are preserved.
  + Given the highly rectangular shape of the input samples (64 x 8192), I used a different striding and padding with larger striding values for horizontal sliding. This helped reducing the number of parameters to a manageable number.
- For the **Predictor** and **Discriminator** components, I implemented three layer feed forward perceptrons.
- Other implementation features:
  + L2 regularization for the **Predictor** and **Discriminator** components
  + Adam optimizier with exponential learning rate decay
  + Tracking the iteration performance based on the train and validation/dev accuracy measured by the loss value and AUC, and selecting the model with the largest validation AUC as the best model
  + A **Save** function that stores all the performance logs (*iter_perf.csv*), the best performace (*best_perf.csv*), hyperparameters ('hyperparameters.csv'), and the best model all in the model folder

Thank you!
