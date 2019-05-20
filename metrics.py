import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback
from keras.preprocessing import image
from sklearn import metrics
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

PATHVAL="validate.csv"
BATCH=16

class roc_callback(Callback):
    def __init__(self,val_gen):
        self.val_gen = val_gen
        x,y = val_gen.getTotal()
        print(y.ravel())
        self.x = x
        self.y = y
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict_generator(self.val_gen,
                                          steps=self.val_gen.n / BATCH,
                                          verbose=0)

        print(self.y[0])
        print(result[0])
        roc_auc = metrics.roc_auc_score(self.y.ravel(), result.ravel())
        print('\r val_roc_auc: %s' % (str(round(roc_auc,4))), end=100*' '+'\n')

        value = coverage_error(self.y, result)
        print('\r coverage_error: %s' % (str(round(value,4))), end=100*' '+'\n')

        value = label_ranking_loss(self.y, result)
        print('\r label_ranking_loss: %s' % (str(round(value, 4))), end=100 * ' ' + '\n')

        roc_auc = label_ranking_average_precision_score(self.y, result)
        print('\r label_ranking_average_precision_score: %s' % (str(round(roc_auc,4))), end=100*' '+'\n')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class coverage_error_callback(Callback):
    def __init__(self):
        datagen = image.ImageDataGenerator()
        validatedf = pd.read_csv(PATHVAL,
                                 dtype={'A': np.float32, 'B': np.float32,
                                        'C': np.float32,
                                        'D': np.float32, 'E': np.float32,
                                        'F': np.float32,
                                        'G': np.float32, 'H': np.float32,
                                        'I': np.float32,
                                        'J': np.float32, 'K': np.float32,
                                        'M': np.float32,
                                        'N': np.float32, 'L': np.float32,
                                        'Path': str})

        self.val_gen = datagen.flow_from_dataframe(validatedf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(224, 224),
                                                    x_col='Path',
                                                    y_col=['A', 'B', 'C', 'D', 'E', 'F', 'G',
                                                           'H',
                                                           'I',
                                                           'J', 'K', 'M', 'N', 'L'],
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

        self.trueLabels = validatedf[
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M", "N", "L", ]].values

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict_generator(self.val_gen,
                                          steps=self.val_gen.n / BATCH,
                                          verbose=0)

        value = coverage_error(self.trueLabels, result)

        print('\r coverage_error: %s' % (str(round(value,4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class label_ranking_average_precision_score_callback(Callback):
    def __init__(self):
        datagen = image.ImageDataGenerator()
        validatedf = pd.read_csv(PATHVAL,
                                 dtype={'A': np.float32, 'B': np.float32,
                                        'C': np.float32,
                                        'D': np.float32, 'E': np.float32,
                                        'F': np.float32,
                                        'G': np.float32, 'H': np.float32,
                                        'I': np.float32,
                                        'J': np.float32, 'K': np.float32,
                                        'M': np.float32,
                                        'N': np.float32, 'L': np.float32,
                                        'Path': str})

        self.val_gen = datagen.flow_from_dataframe(validatedf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(224, 224),
                                                    x_col='Path',
                                                    y_col=['A', 'B', 'C', 'D', 'E', 'F', 'G',
                                                           'H',
                                                           'I',
                                                           'J', 'K', 'M', 'N', 'L'],
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

        self.trueLabels = validatedf[
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M", "N", "L", ]].values

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict_generator(self.val_gen,
                                          steps=self.val_gen.n / BATCH,
                                          verbose=0)

        roc_auc = label_ranking_average_precision_score(self.trueLabels, result)

        print('\r label_ranking_average_precision_score: %s' % (str(round(roc_auc,4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

class label_ranking_loss_callback(Callback):
    def __init__(self):
        datagen = image.ImageDataGenerator()
        validatedf = pd.read_csv(PATHVAL,
                                 dtype={'A': np.float32, 'B': np.float32,
                                        'C': np.float32,
                                        'D': np.float32, 'E': np.float32,
                                        'F': np.float32,
                                        'G': np.float32, 'H': np.float32,
                                        'I': np.float32,
                                        'J': np.float32, 'K': np.float32,
                                        'M': np.float32,
                                        'N': np.float32, 'L': np.float32,
                                        'Path': str})

        self.val_gen = datagen.flow_from_dataframe(validatedf,
                                                    directory=None,
                                                    color_mode='grayscale',
                                                    target_size=(224, 224),
                                                    x_col='Path',
                                                    y_col=['A', 'B', 'C', 'D', 'E', 'F', 'G',
                                                           'H',
                                                           'I',
                                                           'J', 'K', 'M', 'N', 'L'],
                                                    class_mode="other",
                                                    shuffle=False,
                                                    batch_size=BATCH,
                                                    drop_duplicates=False)

        self.trueLabels = validatedf[
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "M", "N", "L", ]].values

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict_generator(self.val_gen,
                                          steps=self.val_gen.n / BATCH,
                                          verbose=0)

        value = label_ranking_loss(self.trueLabels, result)

        print('\r label_ranking_loss: %s' % (str(round(value,4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def auc_roc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred,summation_method='careful_interpolation')
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auc_roc2(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.streaming_dynamic_auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def auc_roc3(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.auc_with_confidence_intervals(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def precision(y_true, y_pred):
    score, up_opt = tf.metrics.precision(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def recall(y_true, y_pred):
    score, up_opt = tf.metrics.recall(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def f1(y_true, y_pred):
    score, up_opt = tf.contrib.metrics.f1_score(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

