"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import numpy as np
import keras.backend as K
import os
import tensorflow as tf
from keras.optimizers import SGD,Adam
from keras import callbacks
from keras.layers import Layer,Dense,Input
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from emotion_encoder.Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd

from emotion_encoder.TIMNET import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


def npairs_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    samples = []
    for i in range(y_true.shape[-1]):
        samples.append(tf.math.reduce_mean(tf.boolean_mask(y_pred, tf.argmax(y_true, axis=1) == i), axis=0))
    samples = tf.convert_to_tensor(samples)
    # y_pos_samples = tf.matmul(y_true, samples)
    similarity_matrix = tf.einsum('ij,kj->ik', y_pred, samples)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=similarity_matrix, labels=y_true)
    return tf.math.reduce_mean(loss)

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)  
        super(WeightLayer, self).build(input_shape)  
 
    def call(self, x):
        tempx = tf.transpose(x,[0,2,1])
        x = K.dot(tempx,self.kernel)
        x = tf.squeeze(x,axis=-1)
        return  x
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])
    
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        get_custom_objects().update({'npairs_loss': npairs_loss})
        print("TIMNET MODEL SHAPE:",input_shape)
    
    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size, 
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True, 
                                name='TIMNET')(self.inputs, bidirection=self.args.bidirection)

        self.decision = WeightLayer()(self.multi_decision)

        self.decision_norm = self.decision/(tf.norm(self.decision, axis=1, keepdims=True)+1e-5)
        # self.similarity_matrix = tf.matmul(self.decision, self.decision, transpose_a=False, transpose_b=True)
        ###
        # self.predictions = Dense(self.num_classes, activation='softmax')(self.decision_norm)
        self.model = Model(inputs = self.inputs, outputs = self.decision)
        
        # LossFunc = {'tf.math.truediv':'npairs_loss', 'dense':'categorical_crossentropy'}
        # lossWeights = {'tf.math.truediv':0.1, 'dense':1}

        self.model.compile(loss = 'npairs_loss',
                        #    loss_weights=lossWeights,
                           optimizer =Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics=['npairs_loss']
                           )
        print("Temporal create success!")
        
    def train(self, x, y):

        print("Use Tensorboard")
        import datetime
        # Hide GPU from visible devices
        log_dir = f"log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i=1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now,'%Y-%m-%d_%H-%M-%S')
        # kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, stratify=y)
        avg_loss = 0
        self.create_model()
        # ytrain = smooth_labels(ytrain, 0.1)
        folder_address = filepath+self.args.data+"_"+str(self.args.random_seed)+"_"+now_time
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)
        weight_path=folder_address+'/'+"weights_best.hdf5"
        checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_npairs_loss', verbose=1,save_weights_only=True,save_best_only=True,mode='min')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
        # max_acc = 0
        best_eva_list = []
        h = self.model.fit(xtrain, ytrain,validation_data=(xtest,  ytest),batch_size = self.args.batch_size, epochs = self.args.epoch, verbose=1,callbacks=[checkpoint,tensorboard])
        self.model.load_weights(weight_path)
        best_eva_list = self.model.evaluate(xtest,  ytest)
        avg_loss += best_eva_list[0]
        # avg_accuracy += best_eva_list[1]
        # print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100))
        # i+=1
        # y_pred_best = self.model.predict(xtest)

        # self.matrix.append(confusion_matrix(np.argmax(ytest,axis=1),np.argmax(y_pred_best,axis=1)))
        # em = classification_report(np.argmax(ytest,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
        # self.eva_matrix.append(em)
        # print(classification_report(np.argmax(ytest,axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))

        # print("Average ACC:",avg_accuracy)
        # self.acc = avg_accuracy
        # writer = pd.ExcelWriter(resultpath+self.args.data+'_'+str(round(self.acc*10000)/100)+"_"+str(self.args.random_seed)+"_"+now_time+'.xlsx')
        # for i,item in enumerate(self.matrix):
        #     temp = {}
        #     temp[" "] = self.class_label
        #     for j,l in enumerate(item):
        #         temp[self.class_label[j]]=item[j]
        #     data1 = pd.DataFrame(temp)
        #     data1.to_excel(writer,sheet_name=str(i))

        #     df = pd.DataFrame(self.eva_matrix[i]).transpose()
        #     df.to_excel(writer,sheet_name=str(i)+"_evaluate")
        # writer.close()

        K.clear_session()
        # self.matrix = []
        # self.eva_matrix = []
        # self.acc = 0
        self.trained = True
    
    # def test(self, x, y, path):
    #     i=1
    #     kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
    #     avg_accuracy = 0
    #     avg_loss = 0
    #     x_feats = []
    #     y_labels = []
    #     for train, test in kfold.split(x, y):
    #         self.create_model()
    #         weight_path=path+'/'+str(self.args.split_fold)+"-fold_weights_best_"+str(i)+".hdf5"
    #         self.model.fit(x[train], y[train],validation_data=(x[test],  y[test]),batch_size = 64,epochs = 0,verbose=0)
    #         self.model.load_weights(weight_path)#+source_name+'_single_best.hdf5')
    #         best_eva_list = self.model.evaluate(x[test],  y[test])
    #         avg_loss += best_eva_list[0]
    #         avg_accuracy += best_eva_list[1]
    #         print(str(i)+'_Model evaluation: ', best_eva_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
    #         i+=1
    #         y_pred_best = self.model.predict(x[test])
    #         self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))
    #         em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label,output_dict=True)
    #         self.eva_matrix.append(em)
    #         print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=self.class_label))
    #         caps_layer_model = Model(inputs=self.model.input,
    #         outputs=self.model.get_layer(index=-2).output)
    #         feature_source = caps_layer_model.predict(x[test])
    #         x_feats.append(feature_source)
    #         y_labels.append(y[test])
    #     print("Average ACC:",avg_accuracy/self.args.split_fold)
    #     self.acc = avg_accuracy/self.args.split_fold
    #     return x_feats, y_labels
    
    def infer(self, x, model_dir):
        batch_size, feat_dim=x.shape[0],x.shape[2]
        x_feats=np.zeros(shape=(10,batch_size,feat_dim))
        # y_preds =np.zeros(shape=(10,batch_size,4))
        weight_path=os.path.join(model_dir, "weights_best.hdf5")
        self.model.load_weights(weight_path)#+source_name+'_single_best.hdf5')
        # y_pred = self.model.predict(x)
        # caps_layer_model = Model(inputs=self.model.input,
        # outputs=self.model.get_layer(index=-2).output)
        x_feats = self.model.predict(x, batch_size=batch_size, verbose=0)
        # y_preds[i-1]=y_pred
        return x_feats
        # return np.mean(x_feats, axis=0), mode(np.argmax(y_preds, axis=-1), axis=0).mode[0]