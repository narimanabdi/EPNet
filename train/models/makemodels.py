from models import densenet,resnet,mobilenet,densenetmini,student
from tensorflow.keras.applications import DenseNet121
from tensorflow import keras
import tensorflow as tf
#from models.conv3b import create_conv3b
def make_proto_model(backbone,input_shape):
    if backbone == 'resnet':
        return resnet.create_model(input_shape = input_shape)
    if backbone == 'densenet':
        return student.create_model_st(input_shape = input_shape)
    if backbone == 'mobilenet':
        return mobilenet.create_model(input_shape = input_shape)
    if backbone == 'densenetmini':
        return densenetmini.create_model(input_shape = input_shape)


def make_base_model(backbone,n_class,dim):
    inp = keras.layers.Input((dim,dim,3))
    if backbone == 'densenet':
        encoder = DenseNet121(
            input_shape=(dim,dim,3),weights=None,include_top=False)
        x = encoder(inp)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(2048,activation='relu')(x)
        x = keras.layers.Dense(1024,activation='relu')(x)
        x = keras.layers.Dense(n_class,activation='softmax')(x)
    # if backbone == 'conv3b':
    #     encoder = create_conv3b(input_shape=(dim,dim,3))
        
        x = encoder(inp)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1024,activation='relu')(x)
        x = keras.layers.Dense(512,activation='relu')(x)
        x = keras.layers.Dense(n_class,activation='softmax')(x)
    
    return keras.Model(inp,x),encoder

