# This module generates encoders for feature extracting
from tensorflow import keras
from tensorflow.keras.layers import Input,Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from models.blocks import conv_block,dcp
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance,Cosine_Distance
from models.stn import stn
from tensorflow.keras.applications import DenseNet121
from models.senet import Senet
from tensorflow.keras.applications import  MobileNetV2

def create_encoder(gfenet_model):
    inp = Input((64,64,3))
    x = gfenet_model(inp)
    x = conv_block(x,kernel_size=(2,2),n_filters=64,strides=(1,1))
    x = conv_block(x,kernel_size=(3,3),n_filters=64,strides=(1,1))
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(units = 100,kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('linear')(x)
    return keras.Model(inp,x,name='stu_encoder')

def create_model_st( teacher_model, input_shape = (64,64,3)):
    #teacher_model.get_layer('encoder').get_layer('model').summary()
    GFENet = keras.Model(
        inputs=teacher_model.get_layer('encoder').get_layer('model').inputs,
        outputs = teacher_model.get_layer('encoder').get_layer('model').get_layer('pool2_pool').output
        #outputs = teacher_model.get_layer('encoder').get_layer('model').outputs
    )
    GFENet.trainable = False
    support = Input(input_shape)
    query = Input(input_shape)
    encoder = create_encoder(GFENet)
    #encoder.summary()
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    #out = Activation("softmax")(dist)
    return Senet(inputs = [support,query],outputs=dist)
