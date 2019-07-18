from model_utils.prototypical_net import ProtoNet
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from keras import optimizers
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
KTF.set_session(session)
#Inital net
lr = 1e-03
SGD = optimizers.SGD(lr=lr, momentum=0.9, decay=3e-04, nesterov=False)
pnet = ProtoNet(ip1 = (28,28,1),
                n_way=60,
                n_shot=5,
                n_query=5,
                optimizer='adam')
pnet.model.summary()
#train
train_data,train_label,lang_dict = pnet.load_data(path='K:\GitHub\omniglot\python\images_background',image_shape=(28,28))
pnet.train(train_data,train_label,
            valsplit = 0.2,
            filename = 'K:\GitHub\prototypical-network\Results\log_record.csv',
            path = 'Results\\', 
            image_augmentation=False,
            augmentation_num = 2,
            rotation_range=180,
            n_epochs=20,n_episodes=100)
pnet.model.save('K:\GitHub\prototypical-network\Results\model.h5')
pnet.record_plot(path = 'K:\GitHub\prototypical-network\Results\log_record.csv')
#test
#pnet.model.load_weights('K:\GitHub\prototypical-network\Results\weights.h5', by_name=True)
test_data,test_label,lang_dict = pnet.load_data(path='K:\GitHub\omniglot\python\images_evaluation',image_shape=(28,28))            
pnet.test(test_data,test_label,n_way=20,n_shot=5,n_query=15,n_episodes=1000)