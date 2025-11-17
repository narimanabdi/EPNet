import argparse
from time import time

import optuna
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from tensorflow.keras.models import load_model

from models.metrics import loss_mse,accuracy,loss_new, loss_ce
from models.makemodels import make_proto_model
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance, Cosine_Distance
from models.stn import BilinearInterpolation,Localization
from models.senet import Senet
from models.distill import Distiller
from models.student import create_model_st


parser = argparse.ArgumentParser('SENet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-3)
parser.add_argument('--dloss',default='mse')
args = parser.parse_args()
#trainng parameters
dim = args.dim
batch = args.batch
epochs = args.epochs
lr = args.lr
#metric trackers
train_acc_tracker = Mean('train_accuracy')
train_loss_tracker = Mean('train_loss')
test_acc_tracker = Mean('train_accuracy')

def make_data_generator(test_mode):
    '''
    return train & test data loaders
    '''
    loader = get_loader(test_mode) 
    if test_mode== 'belga2flick' or \
        test_mode == 'belga2toplogo' or test_mode == 'gtsrb2tt100k' or test_mode == 'gtsrb':
        train_gen, val_gen,test_gen = loader.get_generator(
            batch=batch,dim=dim)
    else:
        train_gen, test_gen = loader.get_generator(
            batch=batch,dim=dim)
        val_gen = []

    return train_gen,val_gen ,test_gen

def make_teacher_encoder():
    encoder = keras.models.load_model(
    'model_files/best_encoders/densenet_' + args.test+ '_encoder.h5',
    custom_objects={
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization},compile=False)
    support = keras.layers.Input((64,64,3))
    query = keras.layers.Input((64,64,3))
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    return Senet(inputs = [support,query],outputs=dist)
    

def objective(trial):
    '''
    meta-training function
    ep: the number of epochs
    '''
  
    te = make_teacher_encoder()

    alpha = trial.suggest_float('alpha',0.1,0.9,step=0.1)
    temp = trial.suggest_int('temp',2,20,step=1)
    lr = trial.suggest_categorical('lr',[0.001,0.0009,1e-4,1e-5,8e-4,2e-3])
   
    teacher = keras.models.load_model(
    'model_files/best_models/densenet_' + args.test+ '_whole.h5',
    custom_objects={
        'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization,'Senet':Senet},compile=False)
    
    teacher_clone = keras.models.load_model(
    'model_files/best_models/densenet_' + args.test+ '_whole.h5',
    custom_objects={
        'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization,'Senet':Senet},compile=False)
    
    teacher_model = keras.Model(
        inputs=teacher.inputs,
        outputs = teacher.get_layer('weighted__euclidean__distance').output
        #outputs = teacher_model.get_layer('encoder').get_layer('model').outputs
    )
 
   
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    teacher.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    #teacher.trainable = False
    student = create_model_st(teacher_model=teacher_clone, input_shape = (dim,dim,3))
    student.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    best_test_acc = 0.0
    #define distiller for knowledge distillation

    distiller = Distiller(student=student, teacher=teacher_model)
    if args.dloss == 'mse':
        distill_loss = loss_mse
    elif args.dloss == 'kl':
        distill_loss = keras.losses.KLDivergence()
    distiller.compile(
        optimizer=optimizer_fn,
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=loss_mse,
        distillation_loss_fn=distill_loss,#distillation_loss_fn=keras.losses.KLDivergence()
        alpha=alpha,
        temperature=temp,
    )
    train_datagen, val_dattagen, test_datagen = make_data_generator(args.test)
    test_acc_tracker = Mean('train_accuracy')
    # for e in range(1):
    #     distiller.fit(train_datagen,verbose=0)
    #     te_acc,_ = distiller.evaluate(test_datagen, verbose=0)
    #     test_acc_tracker.update_state(te_acc)
    distiller.fit(train_datagen,verbose=0)
    te_acc,_ = distiller.evaluate(test_datagen, verbose=0)
    
    return 1 - te_acc


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective,n_trials=30)
    print(study.best_params)

    