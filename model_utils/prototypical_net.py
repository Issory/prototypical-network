import os
from typing import Callable, Union, Optional
import keras
import random
import h5py

from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Dropout,Permute,RepeatVector,Concatenate,MaxPool2D
from keras.layers import Reshape, Conv3DTranspose,Activation, BatchNormalization,Multiply,Add
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau,Callback
from keras import regularizers
from keras import optimizers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from scipy.optimize import curve_fit
from PIL import Image
from scipy.misc import imread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class ProtoNet:

    def __init__(self,
                 ip1 = (105,105,1),
                 n_way=60,
                 n_shot=5,
                 n_query=5,
                 beta=1.0,
                 optimizer: Optional[Union[Callable, str]] = None):
        self.input_shape_image = ip1
        self.beta = beta
        self.num_class = n_way
        self.num_support = n_shot
        self.num_query = n_query
        self.optimizer = "adam"
        if optimizer is not None:
            self.optimizer = optimizer
        
        self.model = self.build_architecture()

    def load_data(self,path,n=0,image_shape=(64,64)):
        X=[]
        y = []
        cat_dict = {}
        lang_dict = {}
        curr_y = n
        #we load every alphabet seperately so we can isolate them later
        for alphabet in os.listdir(path):
            print("loading alphabet: " + alphabet)
            lang_dict[alphabet] = [curr_y,None]
            alphabet_path = os.path.join(path,alphabet)
            #every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                cat_dict[curr_y] = (alphabet, letter)
                category_images=[]
                letter_path = os.path.join(alphabet_path, letter)
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    #image = imread(image_path)
                    image = Image.open(image_path)
                    image = 1-np.array(image.resize(image_shape).convert('L'))
                    category_images.append(image)
                    y.append(curr_y)
                try:
                    X.append(np.stack(category_images))
                #edge case  - last one
                except ValueError as e:
                    print(e)
                    print("error - category_images:", category_images)
                curr_y += 1
                lang_dict[alphabet][1] = curr_y - 1
        y = np.vstack(y)
        X = np.stack(X)
        return X,y,lang_dict

    def generator(self,datagen,support,query,label):
        n_classes, n_support, im_height, im_width, im_channel = support.shape
        n_classes, n_query, im_height, im_width, im_channel = query.shape
        support = np.reshape(support,[-1,im_height,im_width,im_channel])
        query = np.reshape(query,[-1,im_height,im_width,im_channel])
        label = np.reshape(label,[-1])
        generator_support = datagen.flow(support, batch_size=support.shape[0])
        generator_query = datagen.flow(query,label, batch_size=query.shape[0])
        while True:
            batch_support = generator_support.next()
            batch_query,batch_label = generator_query.next()
            batch_support = np.reshape(batch_support,[n_classes,n_support,im_height,im_width,im_channel])
            batch_query = np.reshape(batch_query,[n_classes,n_query,im_height,im_width,im_channel])
            batch_label = np.reshape(batch_label,[n_classes,n_query])
            yield [batch_support,batch_query],batch_label
    
    def train(self,train_data,train_label,
            n_epochs=100,n_episodes=100,
            filename = 'record.csv',
            path = '',
            image_augmentation=False,
            augmentation_num = 1,
            valsplit = 0,
            featurewise_center=False,
            samplewise_center=False, 
            featurewise_std_normalization=False, 
            samplewise_std_normalization=False, 
            zca_whitening=False, 
            zca_epsilon=1e-06, 
            rotation_range=0, 
            width_shift_range=0.0, 
            height_shift_range=0.0, 
            brightness_range=None, 
            shear_range=0.0, 
            zoom_range=0.0, 
            channel_shift_range=0.0, 
            fill_mode='nearest', 
            cval=0.0, 
            horizontal_flip=False, 
            vertical_flip=False, 
            rescale=None, 
            preprocessing_function=None):
        datagen = ImageDataGenerator(
            featurewise_center=featurewise_center,  
            samplewise_center=samplewise_center, 
            featurewise_std_normalization=featurewise_std_normalization, 
            samplewise_std_normalization=samplewise_std_normalization, 
            zca_whitening=zca_whitening, 
            zca_epsilon=zca_epsilon, 
            rotation_range=rotation_range, 
            width_shift_range=width_shift_range, 
            height_shift_range=height_shift_range, 
            brightness_range=brightness_range, 
            shear_range=shear_range, 
            zoom_range=zoom_range, 
            channel_shift_range=channel_shift_range, 
            fill_mode=fill_mode, 
            cval=cval, 
            horizontal_flip=horizontal_flip, 
            vertical_flip=vertical_flip, 
            rescale=rescale, 
            preprocessing_function=preprocessing_function)
        if os.path.exists(filename):
            os.remove(filename)
        file_write_obj = open(filename,'w')
        file_write_obj.write('Batch_num,')
        for var in self.model.metrics_names:
            file_write_obj.write(var)
            file_write_obj.write(',')
        if (valsplit!=0):
            for var in self.model.metrics_names:
                file_write_obj.write('val_'+var)
                file_write_obj.write(',')
        file_write_obj.write('\n')
        n_classes, n_examples, im_height, im_width = train_data.shape
        n_way = self.num_class
        n_shot = self.num_support
        n_query = self.num_query
        num = 0
        val_loss = [np.inf]
        val_classes = int(valsplit*n_classes)
        for ep in range(n_epochs):
            for epi in range(n_episodes):
                epi_classes = np.random.permutation(n_classes-val_classes)[:n_way]
                support = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
                query = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)
                for i, epi_cls in enumerate(epi_classes):
                    selected = np.random.permutation(n_examples)[:n_shot + n_query]
                    support[i] = train_data[epi_cls, selected[:n_shot]]
                    query[i] = train_data[epi_cls, selected[n_shot:]]
                support = np.expand_dims(support, axis=-1)
                query = np.expand_dims(query, axis=-1)
                if (valsplit!=0):
                    epi_classes = np.random.permutation(np.arange(n_classes-val_classes,n_classes))[:n_way]
                    support_val = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
                    query_val = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)
                    for i, epi_cls in enumerate(epi_classes):
                        selected = np.random.permutation(n_examples)[:n_shot + n_query]
                        support_val[i] = train_data[epi_cls, selected[:n_shot]]
                        query_val[i] = train_data[epi_cls, selected[n_shot:]]
                    support_val = np.expand_dims(support_val, axis=-1)
                    query_val = np.expand_dims(query_val, axis=-1)
                    labels_val = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
                    labels_val = keras.utils.to_categorical(labels_val, num_classes=None, dtype='float32')
                if image_augmentation==True:
                    for num_rec in range(augmentation_num):
                        labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
                        [support,query],labels = next(self.generator(datagen,support,query,labels))
                        labels = keras.utils.to_categorical(labels, num_classes=None, dtype='float32')
                        record = self.model.train_on_batch([support,query], labels, sample_weight=None, class_weight=None)
                        num += 1
                        file_write_obj.write('%d,'%(num))
                        for var in record:
                            file_write_obj.write(str(var))
                            file_write_obj.write(',')
                        if (valsplit!=0)&(num%20==0):
                            val_record = self.model.test_on_batch([support_val,query_val], labels_val, sample_weight=None)
                            print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}, val_loss: {:.5f}, val_acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, record[0], record[1],val_record[0], val_record[1]))
                            val_loss.append(val_record[0])
                            for var in val_record:
                                file_write_obj.write(str(var))
                                file_write_obj.write(',')
                            if val_loss[-1]==np.min(val_loss):
                                self.model.save(path+'checkpoint_batch_{}_val_loss_{}.h5'.format(num,val_loss[-1]))
                        file_write_obj.write('\n')
                else:
                    labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
                    labels = keras.utils.to_categorical(labels, num_classes=None, dtype='float32')
                    record = self.model.train_on_batch([support,query], labels, sample_weight=None, class_weight=None)
                    num += 1
                    file_write_obj.write('%d,'%(num))
                    for var in record:
                        file_write_obj.write(str(var))
                        file_write_obj.write(',')
                    if (valsplit!=0)&(num%20==0):
                        val_record = self.model.test_on_batch([support_val,query_val], labels_val, sample_weight=None)
                        print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}, val_loss: {:.5f}, val_acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, record[0], record[1],val_record[0], val_record[1]))
                        val_loss.append(val_record[0])
                        for var in val_record:
                            file_write_obj.write(str(var))
                            file_write_obj.write(',')
                        
                        if val_loss[-1]==np.min(val_loss):
                            self.model.save(path+'checkpoint_batch_{}_val_loss_{}.h5'.format(num,val_loss[-1]))
                    file_write_obj.write('\n')
                if (epi+1) % 50 == 0:
                    print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_episodes, record[0], record[1]))
        file_write_obj.close()

    def test(self,test_data,test_label,
            n_way = None,n_shot = None,
            n_query = None,n_episodes=100):
        n_classes, n_examples, im_height, im_width = test_data.shape
        if n_way is None:
            n_way = self.num_class
        else:
            self.num_class = n_way
        if n_shot is None:
            n_shot = self.num_support
        else:
            self.num_support = n_shot
        if n_query is None:
            n_query = self.num_query
        else:
            self.num_query = n_query
        self.model = self.build_test_model()
        avg_acc = 0
        for epi in range(n_episodes):
            epi_classes = np.random.permutation(n_classes)[:n_way]
            support = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
            query = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(n_examples)[:n_shot + n_query]
                support[i] = test_data[epi_cls, selected[:n_shot]]
                query[i] = test_data[epi_cls, selected[n_shot:]]
            support = np.expand_dims(support, axis=-1)
            query = np.expand_dims(query, axis=-1)
            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
            labels = keras.utils.to_categorical(labels, num_classes=None, dtype='float32')
            record = self.model.test_on_batch([support,query], labels, sample_weight=None)
            avg_acc += record[1]
            if (epi+1) % 50 == 0:
                print('[episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_episodes, record[0], record[1]))
        avg_acc /= n_episodes
        print('Average Test Accuracy: {:.5f}'.format(avg_acc))

    def build_architecture(self):
        encoder = self.build_encoder()
        ### Loss Part ###
        #support_input = Input(batch_shape=(self.num_class,self.num_support)+self.input_shape_image,name='support_input')
        #query_input = Input(batch_shape=(self.num_class,self.num_query)+self.input_shape_image,name='query_input')

        support_input = Input(batch_shape=(None,None)+self.input_shape_image,name='support_input')
        query_input = Input(batch_shape=(None,None)+self.input_shape_image,name='query_input')

        support = Lambda(self.proto_reshape_com_support,name='support_reshape')(support_input)
        query = Lambda(self.proto_reshape_com_query,name='query_reshape')(query_input)
        emb_sup = encoder(support)
        emb_sup = Lambda(self.proto_reshape_sep_support,name='emb_sup_reshape')(emb_sup)
        emb_sup = Lambda(lambda x:K.mean(x,axis=1),name='emb_sup_lambda')(emb_sup)
        emb_q = encoder(query)
        dists = Lambda(self.euclidean_distance,name='euclidean_distance')([emb_q, emb_sup])
        log_p_y = Lambda(self.proto_reshape_sep_query,name='ob_reshape')(dists)
        
        model = Model([support_input,query_input],log_p_y)
        ### Custom Loss Functions ###

        def proto_loss(y_true, y_pred):
            return -K.mean(K.reshape(K.sum(y_true*y_pred,axis=-1),[-1]))
        def proto_acc(y_true,y_pred):
            return K.mean(K.equal(K.argmax(y_pred,axis=-1),K.argmax(y_true,axis=-1)))
        #'categorical_crossentropy'
        model.compile(optimizer=self.optimizer, loss=[proto_loss], metrics=[proto_acc,'categorical_accuracy'])
        print("[*] Model is built and compiled")

        return model
    
    def build_encoder(self):
        ### Encoder Part ###
        image_input = Input(shape=self.input_shape_image,name = 'image_input')
        conv_1 = Conv2D(filters=64, kernel_size=3, padding='same',name = 'encode_conv1')(image_input)# (105-3)/2+1=52
        bn_1 = BatchNormalization(name = 'encode_bn1')(conv_1)
        act_1 = Activation('relu',name = 'encode_act1')(bn_1)
        act_1 = MaxPool2D(name='encode_max1')(act_1)
        conv_2 = Conv2D(filters=64, kernel_size=3, padding='same',name = 'encode_conv2')(act_1)#(52-3)/2+1=25
        bn_2 = BatchNormalization(name = 'encode_bn2')(conv_2)
        act_2 = Activation('relu',name = 'encode_act2')(bn_2)
        act_2 = MaxPool2D(name='encode_max2')(act_2)
        conv_3 = Conv2D(filters=64, kernel_size=3, padding='same',name = 'encode_conv3')(act_2)#(25-3)/2+1=12
        bn_3 = BatchNormalization(name = 'encode_bn3')(conv_3)
        act_3 = Activation('relu',name = 'encode_act3')(bn_3)
        act_3 = MaxPool2D(name='encode_max3')(act_3)
        conv_4 = Conv2D(filters=64, kernel_size=3, padding='same',name = 'encode_conv4')(act_3)#(12-3)/2+1=5
        bn_4 = BatchNormalization(name = 'encode_bn4')(conv_4)
        act_4 = Activation('relu',name = 'encode_act4')(bn_4)
        act_4 = MaxPool2D(name='encode_max4')(act_4)

        conv_4_flat = Flatten(name = 'encode_fattlen')(act_4)

        return Model(image_input,conv_4_flat,name='encoder')
    def proto_reshape_com_support(self,args):
        a = args
        #return K.reshape(a,[self.num_class*self.num_support,self.input_shape_image[0],self.input_shape_image[1],self.input_shape_image[2]])
        return K.reshape(a,[-1,self.input_shape_image[0],self.input_shape_image[1],self.input_shape_image[2]])
    def proto_reshape_com_query(self,args):
        a = args
        #return K.reshape(a,[self.num_class*self.num_query,self.input_shape_image[0],self.input_shape_image[1],self.input_shape_image[2]])
        return K.reshape(a,[-1,self.input_shape_image[0],self.input_shape_image[1],self.input_shape_image[2]])
    def proto_reshape_sep_support(self,args):
        a = args
        return K.reshape(a,[self.num_class,self.num_support,-1])
    def proto_reshape_sep_query(self,args):
        a = args
        return K.reshape(a,[self.num_class,self.num_query,-1])
    def euclidean_distance(self,args):
        a,b = args
        N, D = K.shape(a)[0], K.shape(a)[1]
        M = K.shape(b)[0]
        a = K.expand_dims(a,axis=1)
        b = K.expand_dims(b,axis=0)
        a = K.tile(a, [1,M,1])
        b = K.tile(b, [N,1,1]) 
        dist = K.mean(K.square(a-b),axis=2) 
        return -dist-K.logsumexp(-dist)#tf.nn.log_softmax(-dist)

    def build_test_model(self):
        ### Loss Part ###
        #support_input = Input(batch_shape=(self.num_class,self.num_support)+self.input_shape_image,name='support_input')
        #query_input = Input(batch_shape=(self.num_class,self.num_query)+self.input_shape_image,name='query_input')

        support_input = Input(batch_shape=(None,None)+self.input_shape_image,name='support_input')
        query_input = Input(batch_shape=(None,None)+self.input_shape_image,name='query_input')

        support = Lambda(self.proto_reshape_com_support,name='support_reshape')(support_input)
        query = Lambda(self.proto_reshape_com_query,name='query_reshape')(query_input)
        emb_sup = self.model.get_layer('encoder')(support)
        emb_sup = Lambda(self.proto_reshape_sep_support,name='emb_sup_reshape')(emb_sup)
        emb_sup = Lambda(lambda x:K.mean(x,axis=1),name='emb_sup_lambda')(emb_sup)
        emb_q = self.model.get_layer('encoder')(query)
        dists = Lambda(self.euclidean_distance,name='euclidean_distance')([emb_q, emb_sup])
        log_p_y = Lambda(self.proto_reshape_sep_query,name='ob_reshape')(dists)
        
        model = Model([support_input,query_input],log_p_y)
        ### Custom Loss Functions ###

        def proto_loss(y_true, y_pred):
            return -K.mean(K.reshape(K.sum(y_true*y_pred,axis=-1),[-1]))
        def proto_acc(y_true,y_pred):
            return K.mean(K.equal(K.argmax(y_pred,axis=-1),K.argmax(y_true,axis=-1)))
        
        model.compile(optimizer=self.optimizer, loss=[proto_loss], metrics=[proto_acc,'categorical_accuracy'])
        print("[*] Test Model is rebuilt and compiled")
        return model

    def record_plot(self,path):
        df_record = pd.read_csv(path,sep=',')
        plt.figure()
        plt.plot(df_record['Batch_num'],df_record[self.model.metrics_names[0]],'r',label='train loss')
        plt.grid(True)
        plt.xlabel('Number of Batch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig('batch_loss.png')


   
