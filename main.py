import model as mf
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import tensorflow as tf


# This function puts an RF file and its label and source information into a dictionary
def ConcatenateRecords(i, file, label):
    print(i)
    if not os.path.exists('./data/intermediate/' + file):
        print('Non-existing file!!!')
        return {'RF': None, 'label': None}

    cur_df = pd.read_csv('./data/intermediate/' + file, skiprows=1, header=None)
    if file[0] == 'F':
        source = int(file[3])
    else:
        source = int(file[4])
    return {'RF': cur_df.to_numpy(),
            'label': label,
            'source': source}


# Preprocessing the data
if os.path.exists('./data/all_records.pkl'):
    dataset = pd.read_pickle('./data/all_records.pkl')
else:
    file_labels = pd.read_csv('./data/file_locator.csv', header=None)
    pool = mp.Pool(7)
    dataset_list = list(pool.starmap(ConcatenateRecords, zip(range(file_labels.shape[0]),
                                                             file_labels[0].apply(lambda x: x[21:]),
                                                             file_labels[1])))

    dataset = pd.DataFrame(dataset_list)
    dataset.to_pickle('./data/all_records.pkl')
    del dataset_list

RFs = np.stack(dataset['RF'].tolist(), axis=0) # RF reads as a three dimensional array
labels = dataset['label'].to_numpy() # labels (binary)
sources = dataset['source'].to_numpy() - 1 # subjects/sources (binary)
del dataset

# Hyper-parameter tuning
gan = mf.GAN()
for lr in [0.0005, 0.001]:
    for lambd in [0.2, 0.4, 0.6]:
        gan.Train(RFs[0:800], labels[0:800], sources[0:800],
                  RFs[800:], labels[800:], sources[800:],
                  lr, lambd, 20, 10)
        gan.Save('gan_lr-' + str(lr) + '_lambda-' + str(lambd))

# Usage
best_lr = 0.001
best_lambd = 0.2
best_model = tf.keras.models.load_model('gan_lr-' + str(lr) + '_lambda-' + str(lambd))
print(best_model.predict(RFs[800:805])[0])
print(labels[800:805])
