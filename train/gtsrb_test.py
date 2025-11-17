from time import time
from tensorflow import keras
from models.densenet import create_densenet
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import CategoricalAccuracy
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.models import load_model
from models.stn import BilinearInterpolation,Localization
from argparse import ArgumentParser

from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from prettytable import PrettyTable
from utils import count_params
from models.stn2 import depthwise_stn
from models.stn2 import DepthBilinearInterpolation,DepthLocalization
from models.distances import Cosine_Distance, Euclidean_Distance

parser = ArgumentParser('SENet GTSRB Benchmark')
parser.add_argument('--mode',type=str,default='test')
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--model',type=str,default='normal')

args = parser.parse_args()

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img-mean)/std



def make_gtsrb_model(original_encoder_file):
    input_shape = (64,64,3)
    encoder = load_model(original_encoder_file)
    #encoder.trainable = True
    inp = keras.layers.Input(input_shape)
    x = depthwise_stn(inp,[16,32,24],kernel_size=(5,5),stage=1)
    x = encoder(x)
    x = Dense(43,kernel_initializer='he_normal',name='dense_final')(x)
    x = BatchNormalization(name='batch_final')(x)
    x = Activation('softmax',name='act_final')(x)
    return keras.Model(inputs=inp,outputs = x)

def generate_data(subset='train'):
    if subset == 'train':
        datapath = 'datasets/GTSRB/all'
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            rotation_range=30,
            horizontal_flip=False,
            shear_range=0.1,
            preprocessing_function=standardize, validation_split=0.2)
        train_gen = datagen.flow_from_directory(
            datapath,batch_size=256,class_mode='categorical',
            target_size=(64,64),subset='training',shuffle=True,seed=42)
        val_gen = datagen.flow_from_directory(
            datapath,batch_size=64,class_mode='categorical',
            target_size=(64,64),subset='validation',shuffle=True,seed=42)
        return train_gen, val_gen
    if subset == 'test':
        test_path = 'datasets/GTSRB/GTSRB_Test'
        X = []
        print(f'loading GTSRB Test data ...')
        files = os.listdir(test_path)
        files.sort()
        for f in files:
            file_path = os.path.join(test_path,f)
            X += [standardize(
                img_to_array(
                    load_img(
                        file_path,target_size=(64,64),
                        interpolation='bilinear')))]
        X = tf.constant(np.array(X))
        df = pd.read_csv('datasets/GTSRB/GT-final_test.csv',sep=';')
        y_true = tf.constant(
            to_categorical(np.array(df['ClassId']),num_classes=43))
        return X,y_true

def train(epochs):
    senet_gtsrb = make_gtsrb_model(original_encoder_file= f'model_files/student_gtsrb_mse_encoder.h5')
    metric = CategoricalAccuracy()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4,epsilon=1e-8)
    #optimizer = keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
    loss_fun = keras.losses.CategoricalCrossentropy()
    senet_gtsrb.compile(optimizer=optimizer,loss=loss_fun,metrics=metric)
    senet_gtsrb.summary()
    train_gen,val_gen = generate_data(subset ='train')
    #X,y_true = generate_data(subset = 'test')

    best_test_acc = 0
    for e in range(epochs):
        print(f'========epoch {e+1}=========')
        senet_gtsrb.fit(train_gen,epochs=1)
        _,test_acc = senet_gtsrb.evaluate(val_gen,verbose=1)
        
        if test_acc > best_test_acc:
            senet_gtsrb.save('model_files/senet_gtsrb_bench_diseq_stn.h5')
            best_test_acc = test_acc
        print(f'test accuracy{test_acc:.4f} best accuracy{best_test_acc:.4f}')
        
@tf.function
def make_inference(model,data):
    return model(data)         
def test():
    from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score
    if args.model == 'normal':
        final_h5 = 'model_files/senet_gtsrb_bench_diseq.h5'
    elif args.model == 'stn':
        final_h5 = 'model_files/senet_gtsrb_bench_diseq_stn.h5'
    senet_gtsrb = load_model(
        final_h5,compile=True,        custom_objects={
            'DepthBilinearInterpolation':DepthBilinearInterpolation,
            'DepthLocalization':DepthLocalization})
    metric = CategoricalAccuracy()
    senet_gtsrb.compile(metrics=metric)
    X,y = generate_data(subset = 'test')
    #senet_gtsrb.evaluate(X,y_true,batch_size=128,verbose=1)
    tval = []
    pb = tf.keras.utils.Progbar(len(y),verbose=1)
    p = 0
    y_true = []
    y_pred = []
    #predictions = original_encoder.predict(test_generator)
    for i,y_test in enumerate(y):
        s = time()
        p = make_inference(senet_gtsrb,tf.expand_dims(X[i],axis=0))
        tval.append(time() - s)
        predicted_calss = np.argmax(p)
        actual_class = np.argmax(y_test)
        y_true.append(predicted_calss)
        y_pred.append(actual_class)
        pb.add(1)
    end_time = time()
    precision_score_value = precision_score(y_true,y_pred,average='macro')
    recall_score_value = recall_score(y_true,y_pred,average='macro')
    accuracy_score_value = accuracy_score(y_true,y_pred)
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Top-1 Accuracy", f'{accuracy_score_value*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(senet_gtsrb)*1e6:.1f}'])
    myTable.add_row(["Top-1 Recall", f'{recall_score_value*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{precision_score_value*100.0:.2f}'])
    myTable.add_row(["Average of Inference Time", f'{np.mean(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{np.std(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{accuracy_score_value*100.0/count_params(senet_gtsrb):.2f}ms'])
    #myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

@tf.function
def nn(model,inp,ztemplates,distance):
    z = model(tf.expand_dims(inp,axis=0))
    if distance == 'l2':
        return Euclidean_Distance()([ztemplates,z])
    elif distance == 'cosine':
        return Cosine_Distance()([ztemplates,z])

def test_nn():
    from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score
    from data_loader import get_loader
    # final_h5 = 'model_files/senet_gtsrb_bench_diseq.h5'
    # senet_gtsrb = load_model(
    #     final_h5,compile=True,        custom_objects={
    #         'BilinearInterpolation':BilinearInterpolation,
    #         'Localization':Localization})
    original_encoder = load_model('model_files/student_gtsrb_mse_encoder.h5')
    loader = get_loader('gtsrb2tt100k') 
    metric = CategoricalAccuracy()
    # senet_gtsrb.compile(metrics=metric)
    test_generator,_,_ = loader.get_generator(batch=32,dim=64,shuffle=False)
    #test_generator = tt100k_generator
    #extract template image to fit nearest neighbor
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _Xq,_y,t
    #number of classes
    n_cls = 43
    #embed Xs to latent space
    Zs = original_encoder(Xs)
    #fit nearest neighbor to latent space
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    #start inference and generate report
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    X,y = generate_data(subset = 'test')
    #senet_gtsrb.evaluate(X,y_true,batch_size=128,verbose=1)
    tval = []
    pb = tf.keras.utils.Progbar(len(y),verbose=1)
    p = 0
    y_true = []
    y_pred = []
    #predictions = original_encoder.predict(test_generator)
    for i,y_test in enumerate(y):
        s = time()
        #p = make_inference(senet_gtsrb,tf.expand_dims(X[i],axis=0))
        p = nn(original_encoder,X[i],Zs,distance='cosine')
        tval.append(time() - s)
        predicted_calss = np.argmax(p)
        actual_class = np.argmax(y_test)
        y_true.append(predicted_calss)
        y_pred.append(actual_class)
        pb.add(1)
    end_time = time()
    precision_score_value = precision_score(y_true,y_pred,average='macro')
    recall_score_value = recall_score(y_true,y_pred,average='macro')
    accuracy_score_value = accuracy_score(y_true,y_pred)
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Top-1 Accuracy", f'{accuracy_score_value*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder)*1e6:.1f}'])
    myTable.add_row(["Top-1 Recall", f'{recall_score_value*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{precision_score_value*100.0:.2f}'])
    myTable.add_row(["Average of Inference Time", f'{np.mean(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{np.std(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{accuracy_score_value*100.0/count_params(original_encoder):.2f}ms'])
    #myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')
    

if __name__ == '__main__':
    if args.mode == 'train':
        train(args.epochs)
    if args.mode == ('test'):
        test()
    if args.mode == ('nn'):
        test_nn()