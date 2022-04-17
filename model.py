import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, ZeroPadding2D, \
    BatchNormalization, Activation, Add, Input, Lambda
from tensorflow.keras.initializers import glorot_uniform
import numpy as np
from sklearn import metrics
import pandas as pd
import csv

# The class for generative adversarial learning
class GAN():
    def __init__(self):
        self.E = None # to store the encoder
        self.P = None # to store the predictor
        self.D = None # to store the discriminator
        self.FM = None # to store the full model
        self.iter_perfomance = [] # to store the performance metric in each training iteration
        # to store the model and performance of the best model (i.e. best iteration)
        self.BestModel = dict({'Model': None,
                               'Val AUC': None,
                               'Train AUC': None,
                               'Val Cost': None,
                               'Train Cost': None})
        self.Hyperparams = dict() # to store the training hyperparameters

    # This function constructs the identity block of the encoder CNN
    # Modified from https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/2.Deep%20Convolutional%20Models/Residual%20Networks.ipynb
    def identity_block(self, X, f, filters, stage, block):
        # Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2 = filters

        # Save the input value
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # # Third component of main path
        # X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
        #            kernel_initializer=glorot_uniform(seed=0))(X)
        # X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    # This function constructs the convolutional block of the encoder CNN
    # Modified from https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/2.Deep%20Convolutional%20Models/Residual%20Networks.ipynb
    def convolutional_block(self, X, f, filters, stage, block, s1=2, s2=3):
        # Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(f, f), strides=(s1, s2), padding='same', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # # Third component of main path
        # X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0)(X)(X)
        # X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
        #
        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s1, s2), padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    # The CNN encoder constructor
    def Encoder(self, input_shape):
        if self.E:
            return self.E

        X_input = Input(shape=input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 5), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 3))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64], stage=2, block='a')
        # X = identity_block(X, 3, [64, 64], stage=2, block='b')
        # X = identity_block(X, 3, [64, 64], stage=2, block='b')

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128, 128], stage=3, block='a')
        # X = identity_block(X, 3, [128, 128], stage=3, block='a', s = 2)
        # X = identity_block(X, 3, [128, 128], stage=3, block='b')
        # X = identity_block(X, 3, [128, 128], stage=3, block='c')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256, 256], stage=4, block='a')
        # X = identity_block(X, 3, [256, 256], stage=4, block='a', s = 2)
        # X = identity_block(X, 3, [256, 256], stage=4, block='b')
        # X = identity_block(X, 3, [256, 256], stage=4, block='c')
        # X = identity_block(X, 3, [256, 256], stage=4, block='d')
        # X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        #
        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512, 512], stage=5, block='a')
        # X = identity_block(X, 3, [512, 512], stage=5, block='a', s = 2)
        # X = identity_block(X, 3, [512, 512], stage=5, block='b')

        # AVGPOOL
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        # Output layer
        X = Flatten()(X)
        # X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0)(X)

        # Create model
        self.E = Model(inputs=X_input, outputs=X, name='Encoder')

        # return model
        return self.E

    # The predictor component (a three layer perceptron) constructor
    def Predictor(self, input_shape):
        if self.P:
            return self.P

        X_input = Input(shape=input_shape)

        # Predictor
        X = Dense(np.ceil(input_shape[0] / 2), activation='relu', name='fc1_p',
                  kernel_initializer=glorot_uniform(seed=0), kernel_regularizer='l2')(X_input)
        X = BatchNormalization()(X)
        X = Dense(np.ceil(input_shape[0] / 4), activation='relu', name='fc2_p',
                  kernel_initializer=glorot_uniform(seed=0), kernel_regularizer='l2')(X)
        X = BatchNormalization()(X)
        X = Dense(1, activation='sigmoid', name='fc3_p', kernel_initializer=glorot_uniform(seed=0),
                  kernel_regularizer='l2')(X)

        # Create model
        self.P = Model(inputs=X_input, outputs=X, name='Predictor')

        return self.P

    # The discriminator component (a three layer perceptron) constructor
    def Discriminator(self, input_shape):
        if self.D:
            return self.D

        X_input = Input(shape=input_shape)

        # Predictor
        X = Dense(np.ceil(input_shape[0] / 2), activation='relu', name='fc1_d',
                  kernel_initializer=glorot_uniform(seed=0), kernel_regularizer='l2')(X_input)
        X = BatchNormalization()(X)
        X = Dense(np.ceil(input_shape[0] / 4), activation='relu', name='fc2_d',
                  kernel_initializer=glorot_uniform(seed=0), kernel_regularizer='l2')(X)
        X = BatchNormalization()(X)
        X = Dense(1, activation='sigmoid', name='fc3_d', kernel_initializer=glorot_uniform(seed=0),
                  kernel_regularizer='l2')(X)

        # Create model
        self.D = Model(inputs=X_input, outputs=X, name='Discriminator')

        return self.D

    # The loss function based on equation (3) - returns a tensor
    def Loss(self, targets, pred_p, pred_d, lambd, negative=False, verbose=False):
        y_p = targets['Predictor'].reshape(1, -1)
        y_d = targets['Discriminator'].reshape(1, -1)
        pred_p = tf.transpose(pred_p)
        pred_d = tf.transpose(pred_d)
        if verbose:
            print('Predictor Batch Loss: {0} - Discriminator Batch Loss: {1}'.format( \
                round(float(tf.keras.losses.binary_crossentropy(y_p, pred_p).numpy()[0]), 4),
                round(float(tf.keras.losses.binary_crossentropy(y_d, pred_d).numpy()[0]), 4)))
        if negative:
            return - tf.keras.losses.binary_crossentropy(y_p, pred_p) + \
                   lambd * tf.keras.losses.binary_crossentropy(y_d, pred_d)

        return tf.keras.losses.binary_crossentropy(y_p, pred_p) - \
               lambd * tf.keras.losses.binary_crossentropy(y_d, pred_d)

    # The full model constructor putting the encoder, predictor, and discriminator together as in figure 1(b)
    def FullModel(self, input_shape):
        X = Input(shape=input_shape)
        X_e = self.Encoder(input_shape)(X)
        X_p = self.Predictor(tf.keras.backend.int_shape(X_e)[1:])(X_e)
        I_d = tf.keras.layers.concatenate([X_e, X_p])
        X_d = self.Discriminator(tf.keras.backend.int_shape(I_d)[1:])(I_d)

        model = Model(X, [X_p, X_d])

        return model

    # The training algorithm as in Algorithm 1
    def Train(self, X_train, Y_train_p, Y_train_d, X_val, Y_val_p, Y_val_d, learning_rate, lambd, batch_size,
              num_epochs):
        # Initialization
        tf.keras.backend.clear_session()
        self.E = None
        self.P = None
        self.D = None
        self.FM = None
        self.iter_perfomance = []
        self.BestModel = dict({'Model': None,
                               'Val AUC': None,
                               'Train AUC': None,
                               'Val Cost': None,
                               'Train Cost': None})
        self.Hyperparams = {'lr': learning_rate, 'lambda': lambd, 'batch size': batch_size}

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=400,
            decay_rate=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Constructing the model
        self.FM = self.FullModel((64, 8192, 1))
        self.E.summary()
        self.P.summary()
        self.D.summary()
        self.FM.summary()

        # # too slow
        # train_AUC_p = tf.keras.metrics.AUC()
        # validation_AUC_p = tf.keras.metrics.AUC()
        # train_AUC_d = tf.keras.metrics.AUC()
        # validation_AUC_d = tf.keras.metrics.AUC()

        for epoch in range(num_epochs):
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))
            for bi in range(num_batches):
                # Building the batch
                start_ind = bi * batch_size
                end_ind = (bi + 1) * batch_size
                if bi == num_batches - 1:
                    end_ind = X_train.shape[0]
                inputs = X_train[start_ind:end_ind, ]
                targets = {'Predictor': Y_train_p[start_ind:end_ind],
                           'Discriminator': Y_train_d[start_ind:end_ind]}

                # Upper bound for the discriminator loss
                H_d = -np.mean(targets['Discriminator']) * np.log2(np.mean(targets['Discriminator'])) - \
                      (1 - np.mean(targets['Discriminator'])) * np.log2(1 - np.mean(targets['Discriminator']))

                # ENCODER
                ## Un/Freezing components
                self.FM.get_layer('Encoder').trainable = True
                self.FM.get_layer('Predictor').trainable = False
                self.FM.get_layer('Discriminator').trainable = False
                # Open a GradientTape.
                with tf.GradientTape() as tape:
                    # Forward pass.
                    predictions = self.FM(inputs)
                    # Compute the loss value for this batch.
                    loss_value = self.Loss(targets, predictions[0], predictions[1], lambd)
                    # Get gradients of loss wrt the *trainable* weights.
                gradients = tape.gradient(loss_value, self.FM.trainable_weights)
                # Update the weights of the model.
                # print(len(self.FM.trainable_weights))
                optimizer.apply_gradients(zip(gradients, self.FM.trainable_weights))
                # predictions = self.FM(inputs)
                # _ = self.Loss(targets, predictions, lambd, verbose=True)

                # PREDICTOR
                ## Un/Freezing components
                self.FM.get_layer('Encoder').trainable = False
                self.FM.get_layer('Predictor').trainable = True
                with tf.GradientTape() as tape:
                    # Forward pass.
                    predictions = self.FM(inputs)
                    # Compute the loss value for this batch.
                    loss_value = self.Loss(targets, predictions[0], predictions[1], lambd)
                    # Get gradients of loss wrt the *trainable* weights.
                gradients = tape.gradient(loss_value, self.FM.trainable_weights)
                # Update the weights of the model.
                # print(len(self.FM.trainable_weights))
                optimizer.apply_gradients(zip(gradients, self.FM.trainable_weights))
                # predictions = self.FM(inputs)
                # _ = self.Loss(targets, predictions, lambd, verbose=True)

                # DISCRIMINATOR
                ## Un/Freezing components
                self.FM.get_layer('Predictor').trainable = False
                self.FM.get_layer('Discriminator').trainable = True
                predictions = self.FM(inputs)
                while tf.keras.losses.binary_crossentropy(targets['Discriminator'].reshape(1, -1),
                                                          tf.transpose(predictions[1])).numpy()[0] > H_d:
                    print('within while! ce: ' +
                          str(round(tf.keras.losses.binary_crossentropy(targets['Discriminator'].reshape(1, -1),
                                                                        tf.transpose(predictions[1])).numpy()[0], 3)) +
                          ' > H_d: ' + str(round(H_d, 3)))
                    with tf.GradientTape() as tape:
                        # Forward pass.
                        predictions = self.FM(inputs)
                        # Compute the loss value for this batch.
                        loss_value = self.Loss(targets, predictions[0], predictions[1], lambd,
                                               negative=True)
                        # Get gradients of loss wrt the *trainable* weights.
                    gradients = tape.gradient(loss_value, self.FM.trainable_weights)
                    # Update the weights of the model.
                    # print(len(self.FM.trainable_weights))
                    optimizer.apply_gradients(zip(gradients, self.FM.trainable_weights))
                    predictions = self.FM(inputs)
                    # _ = self.Loss(targets, predictions, lambd, verbose=True)

                ## too slow
                # train_AUC_p.reset_states()
                # validation_AUC_p.reset_states()
                # train_AUC_d.reset_states()
                # validation_AUC_d.reset_states()
                # _ = train_AUC_p.update_state(targets['Predictor'], predictions[0])
                # _ = validation_AUC_p.update_state(Y_val_p, self.FM(X_val)[0])
                # _ = train_AUC_d.update_state(targets['Discriminator'], predictions[1])
                # _ = validation_AUC_d.update_state(Y_val_d, self.FM(X_val)[1])

                # Evaluating and printing the iteration performance
                predictions = self.FM(inputs)
                val_preds = self.FM(X_val)

                fpr, tpr, thresholds = metrics.roc_curve(targets['Predictor'], predictions[0].numpy(), pos_label=1)
                train_AUC_p = metrics.auc(fpr, tpr)
                fpr, tpr, thresholds = metrics.roc_curve(targets['Discriminator'], predictions[1].numpy(), pos_label=1)
                train_AUC_d = metrics.auc(fpr, tpr)
                fpr, tpr, thresholds = metrics.roc_curve(Y_val_p, val_preds[0].numpy(), pos_label=1)
                validation_AUC_p = metrics.auc(fpr, tpr)
                fpr, tpr, thresholds = metrics.roc_curve(Y_val_d, val_preds[1].numpy(), pos_label=1)
                validation_AUC_d = metrics.auc(fpr, tpr)

                print('\nEpoch: {0} - Batch: {1}'.format(epoch, bi))
                c = self.Loss(targets, predictions[0], predictions[1], lambd, verbose=True)
                c_val = self.Loss({'Predictor': Y_val_p,
                                   'Discriminator': Y_val_d}, val_preds[0], val_preds[1], lambd)
                # Update the best model
                if (not self.BestModel['Model']) or validation_AUC_p > self.BestModel['Val AUC']:
                    self.BestModel['Model'] = tf.keras.models.clone_model(self.FM)
                    self.BestModel['Model'].set_weights(self.FM.get_weights())
                    self.BestModel['Val AUC'] = validation_AUC_p
                    self.BestModel['Val Cost'] = c_val.numpy()[0]

                print('Batch Results      -> Total Loss: {0} - Predictor AUC: {1} - Discriminator AUC: {2}'.format(
                    round(float(c.numpy()[0]), 4),
                    round(train_AUC_p, 3),
                    round(train_AUC_d, 3)))
                print('Validation Results -> Total Loss: {0} - Predictor AUC: {1} - Discriminator AUC: {2}'.format(
                    round(float(c_val.numpy()[0]), 4),
                    round(validation_AUC_p, 3),
                    round(validation_AUC_d, 3)))

                # Tracking the iteration performance
                self.iter_perfomance.append({
                    'Batch Cost': float(c.numpy()[0]),
                    'Val Cost': float(c_val.numpy()[0]),
                    'Batch Predictor AUC': train_AUC_p,
                    'Val Predictor AUC': validation_AUC_p,
                    'Batch Discriminator AUC': train_AUC_d,
                    'Val Discriminator AUC': validation_AUC_d
                })

        # Computing the training performance for the best model
        print('Computing the training performance...')
        perf = self.Performance(self.BestModel['Model'], X_train, Y_train_p, Y_train_d, lambd)
        self.BestModel['Train AUC'] = perf[0]
        self.BestModel['Train Cost'] = perf[2]

    # This function computes the model performance on a given dataset
    def Performance(self, model, X, y_p, y_d, lambd):
        assert X.shape[0] == len(y_p)
        preds_p = None
        preds_d = None
        # Memory issue when predicting the whole dataset, so breaking into chunks and concatenating the results later
        for i in range(0, len(y_p), 64):
            print(str(i + 1) + '/' + str(len(y_p)))
            preds = model(X[i:min((i + 64), len(y_p))])
            if preds_p == None:
                preds_p = preds[0]
                preds_d = preds[1]
            else:
                preds_p = tf.concat([preds_p, preds[0]], 0)
                preds_d = tf.concat([preds_d, preds[1]], 0)

        fpr, tpr, thresholds = metrics.roc_curve(y_p, preds_p.numpy(), pos_label=1)
        p_auc = metrics.auc(fpr, tpr)
        fpr, tpr, thresholds = metrics.roc_curve(y_d, preds_d.numpy(), pos_label=1)
        d_auc = metrics.auc(fpr, tpr)

        return [p_auc, d_auc, self.Loss({'Predictor': y_p,
                                         'Discriminator': y_d}, preds_p, preds_d, lambd).numpy()[0]]

    # This function saves the training information (the best model and its performance,
    #                                               iteration performances, and the hyperparameters)
    def Save(self, path):
        self.BestModel['Model'].save(path)

        pd.DataFrame(self.iter_perfomance).to_csv(path + '/iter_perf.csv')

        w = csv.writer(open(path + '/best_perf.csv', "w"))
        for key, val in self.BestModel.items():
            if key == 'Model':
                continue
            w.writerow([key, val])

        w = csv.writer(open(path + '/hyperparameters.csv', "w"))
        for key, val in self.Hyperparams.items():
            w.writerow([key, val])
