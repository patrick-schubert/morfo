import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, fbeta_score
from sklearn.model_selection import train_test_split
from opt import RAdam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import efficientnet.keras as efn
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add, BatchNormalization, Input, Dense, MaxPooling2D, Conv2D, Flatten, Concatenate
from keras.layers.core import Activation, Layer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

models = ['Images_MinMaxStandard', 'Images_MinMaxRange', 'Images_Standard']

for model_name in models:
    print(f"Importing Dataset | {model_name}")

    data_dir = '/home/dados2T/Morfo/'
    #model_name = 'Images_MinMaxRange'
    images = np.load(os.path.join(data_dir, model_name + '.npy'))
    target = np.load(os.path.join(data_dir,'Y.npy'))
    target = target[:,-2:]
    #pad = np.zeros((images.shape[0],images.shape[1],images.shape[2],1), dtype="float32")

    X_train, X_test, Y_train, Y_test = train_test_split(images[:,:,:,7:10], target, test_size = 0.10, random_state = 7) # CHANNELS G,R,I
    #X_train_F378, X_test_F378, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,0:1],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F395, X_test_F395, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,1:2],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F410, X_test_F410, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,2:3],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F430, X_test_F430, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,3:4],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F515, X_test_F515, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,4:5],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F660, X_test_F660, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,5:6],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_F861, X_test_F861, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,6:7],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_G, X_test_G, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,7:8],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_I, X_test_I, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,8:9],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_R, X_test_R, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,9:10],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_U, X_test_U, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,10:11],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)
    #X_train_Z, X_test_Z, Y_train, Y_test = train_test_split(np.concatenate([images[:,:,:,11:],pad,pad], axis=-1), target, test_size = 0.10, random_state = 7)


    del images
    print(X_train.shape)
    print(X_test.shape)
    
    print("Building Model")

    inp = Input((320,320,3))
    efn_arc = efn.EfficientNetB2(input_tensor = inp, weights='imagenet')

    y_hat = Dense(2,activation ="sigmoid")(efn_arc.layers[-2].output)

    model = Model(efn_arc.input, y_hat)

    model.compile(loss = "categorical_crossentropy", optimizer=RAdam(),metrics = ['accuracy'])
    
    print("Training Model")

    model_name = model_name + '.hdf5'
    batch_size = 20
    check = ModelCheckpoint(model_name, monitor="val_loss", verbose=1, save_best_only=True)

    gen = ImageDataGenerator(
            rotation_range=180,
            zoom_range=0.20,
            vertical_flip = True,
        horizontal_flip=True,
            fill_mode="nearest")
    """
    def gen_flow_for_three_inputs(X1, X2, X3, y):
        genX1 = gen.flow(X1,y,  batch_size=batch_size,seed=1)
        genX2 = gen.flow(X2, batch_size=batch_size,seed=1)
        genX3 = gen.flow(X3, batch_size=batch_size,seed=1)
        while True:
                X1i = genX1.next()
                X2i = genX2.next()
                X3i = genX3.next()
                #Assert arrays are equal - this was for peace of mind, but slows down training
                #np.testing.assert_array_equal(X1i[0],X2i[0])
                yield [X1i[0], X2i, X3i], X1i[1]

    gen_flow = gen_flow_for_three_inputs(X_train_h, X_train_j, X_train_y, Y_train)
    """

    history = model.fit_generator(gen.flow(X_train, Y_train, batch_size = batch_size), epochs = 20,  
                verbose = 1, validation_data= (X_test, Y_test), callbacks=[check], 
                steps_per_epoch = X_train.shape[0] // batch_size)

    print("Training Statistics")
    pred = model.predict(X_train)

    fig = plt.figure(figsize = (40,20))
    fig.suptitle(model_name[:-5])

    plt.subplot(2,3,1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], thresh = roc_curve(Y_train[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['E','S']
    for i in range(2):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Training ROC')
    plt.legend(loc="lower right")


    print("Test Statistics")

    pred = model.predict(X_test)

    plt.subplot(2,3,2)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], thresh = roc_curve(Y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['E','S']
    for i in range(2):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Test ROC')
    plt.legend(loc="lower right")


    print("Best Model Statistics")

    model.load_weights(model_name)
    pred = model.predict(X_test)

    plt.subplot(2,3,3)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], thresh = roc_curve(Y_test[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    lw = 2
    colors = ['darkblue','darkorange']
    classes = ['E','S']
    for i in range(2):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Best Model ROC')
    plt.legend(loc="lower right")


    print("Else Statistics")

    plt.subplot(2,3,4)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(2,3,5)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')



    plt.savefig(f"{model_name[:-5]}.png")