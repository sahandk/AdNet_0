# Huami Take-home Assignment
### *Sahand Khakabimamaghani*

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
