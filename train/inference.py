##############################
#                            #
# Nearest Neighbor Inference #
#                            #
##############################

from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from prettytable import PrettyTable
from utils import count_params
import argparse
from models.distances import Cosine_Distance, Euclidean_Distance

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Precision, Recall,TopKCategoricalAccuracy
from models.metrics import loss_mse,accuracy,loss_new, loss_ce
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.senet import Senet

from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score

rec_tracker = Mean(name='Nearest_Neighbor_Recall')
pre_tracker = Mean(name='Nearest_Neighbor_Precision')
_recall = Recall()
_precision = Precision()

parser = argparse.ArgumentParser('Nearest Neighbor Test')
parser.add_argument('--data',type = str,default='gtsrb2tt100k',help = 'Test type')
parser.add_argument('--mode',type = str,default='normal')
parser.add_argument('--batch',type = int,default=128)
parser.add_argument('--metric',default='l2')
parser.add_argument('--lite',help='TFLite Encoder File name')
parser.add_argument('--dloss',default='mse')
args = parser.parse_args()

#get data loader
loader = get_loader(args.data) 
batch = args.batch
#test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
#tracker for benckmarking
acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
time_tracker = Mean(name='Time')

def compute_confidence_interval(accuracy_value:float,number_samples:int,confidence:int):
    standard_error = np.sqrt(accuracy_value*(1-accuracy_value)/number_samples)
    if confidence == 95:
        const = 1.96
    margin_error = const * standard_error
    low_bound = accuracy_value - margin_error
    high_bound = accuracy_value + margin_error
    return low_bound, high_bound
@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    if args.metric == 'l2':
        return Euclidean_Distance()([ztemplates,z])
    elif args.metric == 'cosine':
        return Cosine_Distance()([ztemplates,z])

@tf.function
def nn_lite(inp,ztemplates):
    z = tf.expand_dims(inp,axis=0)
    if args.metric == 'l2':
        return Euclidean_Distance()([ztemplates,z])
    elif args.metric == 'cosine':
        return Cosine_Distance()([ztemplates,z])

def inference_lite(interpreter,img,input_details,output_details):
    '''
    This function genererate output for a single input image 
    '''
    img = tf.expand_dims(img,axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def run_model(original_encoder_file,test='gtsrb2tt100k'):
    '''
    Run a standard test
    original_encoder_file: Encoder h5 file
    '''
    #generate data loader
    original_encoder = load_model(original_encoder_file)
    loader = get_loader(test) 
    #test_generator = loader.get_test_generator(batch=batch,dim=32,shuffle=False)
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    #test_generator = tt100k_generator
    #extract template image to fit nearest neighbor
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _Xq,_y,t
    #number of classes
    n_cls = len(Xs)
    #embed Xs to latent space
    Zs = original_encoder(Xs)
    #fit nearest neighbor to latent space
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    #start inference and generate report
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    start_time = time()
    batches = 0
    Zq = np.empty((args.batch,300))
    tval = []
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    p = 0
    y_true = []
    y_pred = []
    number_of_samples = 0
    for data,y_test in test_generator:
        for i,x in enumerate(data[1]):
            s = time()
            p = nn(original_encoder,x,Zs)
            tval.append(time() - s)
            predicted_calss = np.argmax(p)
            actual_class = np.argmax(y_test[i])
            y_true.append(predicted_calss)
            y_pred.append(actual_class)
            number_of_samples += 1

        #acc_tracker.update_state(nn.score(Zq,y_test))
        batches = batches + 1
        #break
        pb.add(1)
    end_time = time()
    precision_score_value = precision_score(y_true,y_pred,average='macro')
    recall_score_value = recall_score(y_true,y_pred,average='macro')
    f1_score = (2*precision_score_value*recall_score_value)/(precision_score_value+recall_score_value)
    accuracy_score_value = accuracy_score(y_true,y_pred)
    low_conf,high_conf = compute_confidence_interval(accuracy_score_value,number_of_samples,95)
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Loss", args.dloss])
    myTable.add_row(["Distance", args.metric])
    myTable.add_row(["Top-1 Accuracy", f'{accuracy_score_value*100.0:.2f}'])
    myTable.add_row(["95 low conf", f'{low_conf*100.0:.2f}'])
    myTable.add_row(["95 high conf", f'{high_conf*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder)*1e6:.1f}'])
    myTable.add_row(["Top-1 Recall", f'{recall_score_value*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{precision_score_value*100.0:.2f}'])
    myTable.add_row(["F1 Score", f'{f1_score*100.0:.2f}'])
    myTable.add_row(["Average of Inference Time", f'{np.mean(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{np.std(tval)*1000.0:.2f}ms'])
    #myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

def run_model_lite(original_encoder_file,lite_encoder_file,test='gtsrb2tt100k'):
    '''
    Run a standard test
    original_encoder_file: Encoder h5 file
    lite_encoder_file: tflite encoder file
    '''
    #generate data loader
    original_encoder = load_model(original_encoder_file)
    loader = get_loader(test) 
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    #extract template image to fit nearest neighbor
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _Xq,_y,t
    #number of classes
    n_cls = len(Xs)
    #embed Xs to latent space
    Zs = original_encoder(Xs)
    #fit nearest neighbor to latent space
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    #start inference and generate report
    interpreter = tf.lite.Interpreter(lite_encoder_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    start_time = time()
    batches = 0
    Zq = np.empty((args.batch,100))
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    p = 0
    y_true = []
    y_pred = []
    tval = []
    number_of_samples = 0
    for data,y_test in test_generator:
        for i,x in enumerate(data[1]):
            s = time()
            Zq[i] = inference_lite(
                interpreter = interpreter,
                img= x,
                input_details=input_details,
                output_details=output_details)
            #acc_tracker.update_state(nn.score(tf.expand_dims(Zq[i],axis=0),tf.expand_dims(y_test[i],axis=0)))
            #p = nn.predict(tf.expand_dims(Zq[i],axis=0))
            p = nn_lite(inp=Zq[i],ztemplates=Zs)
            tval.append(time() - s)
            predicted_calss = np.argmax(p)
            actual_class = np.argmax(y_test[i])
            y_true.append(predicted_calss)
            y_pred.append(actual_class)
            number_of_samples += 1

        #acc_tracker.update_state(nn.score(Zq,y_test))
        batches = batches + 1
        #break
        pb.add(1)
    end_time = time()
    precision_score_value = precision_score(y_true,y_pred,average='macro')
    recall_score_value = recall_score(y_true,y_pred,average='macro')
    f1_score = (2*precision_score_value*recall_score_value)/(precision_score_value+recall_score_value)
    accuracy_score_value = accuracy_score(y_true,y_pred)
    low_conf,high_conf = compute_confidence_interval(accuracy_score_value,number_of_samples,95)
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Loss", args.dloss])
    myTable.add_row(["Distance", args.metric])
    myTable.add_row(["Top-1 Accuracy", f'{accuracy_score_value*100.0:.2f}'])
    myTable.add_row(["95 low conf", f'{low_conf*100.0:.2f}'])
    myTable.add_row(["95 high conf", f'{high_conf*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder)*1e6:.1f}'])
    myTable.add_row(["Top-1 Recall", f'{recall_score_value*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{precision_score_value*100.0:.2f}'])
    myTable.add_row(["F1 Score", f'{f1_score*100.0:.2f}'])
    myTable.add_row(["Average of Inference Time", f'{np.mean(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{np.std(tval)*1000.0:.2f}ms'])
    #myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

@tf.function
def make_inference(model,data):
    logits = model(data)
    return tf.nn.softmax(logits)
def run_whole_model(original_encoder_file,test='gtsrb2tt100k'):
    '''
    Run a standard test
    original_encoder_file: Encoder h5 file
    '''
    #generate data loader
    #original_encoder = load_model(original_encoder_file)
    loader = get_loader(test) 
    #test_generator = loader.get_test_generator(batch=batch,dim=32,shuffle=False)
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)

    original_encoder = load_model(
        original_encoder_file,
        custom_objects={
            'Euclidean_Distance':Euclidean_Distance,
            'Senet':Senet},compile=False)

    optimizer_fn = keras.optimizers.Adam(learning_rate=0.01,epsilon=1.0e-8)
    original_encoder.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=[TopKCategoricalAccuracy(k=1,name = 'Top5accuracy')])
    #te_acc = original_encoder.evaluate(test_generator)
    #print(te_acc)
    start_time = time()
    batches = 0
    tval = []
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    p = 0
    y_true = []
    y_pred = []
    #predictions = original_encoder.predict(test_generator)
    for data,y_test in test_generator:
        s = time()
        p = make_inference(original_encoder,data)
        tval.append(time() - s)
        for i in range(len(y_test)):
            predicted_calss = np.argmax(p[i])
            actual_class = np.argmax(y_test[i])
            y_true.append(predicted_calss)
            y_pred.append(actual_class)
        pb.add(1)
    precision_score_value = precision_score(y_true,y_pred,average='macro')
    recall_score_value = recall_score(y_true,y_pred,average='macro')
    f1_score = (2*precision_score_value*recall_score_value)/(precision_score_value+recall_score_value)
    accuracy_score_value = accuracy_score(y_true,y_pred)
    myTable = PrettyTable([" Softmax Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Top-1 Accuracy", f'{accuracy_score_value*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder)*1e6:.1f}'])
    myTable.add_row(["Top-1 Recall", f'{recall_score_value*100.0:.2f}'])
    myTable.add_row(["Top-1 Precision", f'{precision_score_value*100.0:.2f}'])
    myTable.add_row(["F1 Score", f'{f1_score*100.0:.2f}'])
    myTable.add_row(["Average of Inference Time", f'{np.mean(tval)*1000.0:.2f}ms'])
    myTable.add_row(["STD of Inference Time", f'{np.std(tval)*1000.0:.2f}ms'])
    #myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

if __name__ == '__main__':
    if args.mode == 'lite':
        #run_model_lite(
                #original_encoder_file='model_files/best_encoders/student_' + args.data + '_encoder.h5',
                #lite_encoder_file='model_files/best_encoders/student_' + args.data + '_encoder.tflite',
                #test=args.data)
        run_model_lite(
                original_encoder_file='model_files/student_gtsrb2tt100k_mse_encoder.h5',
                lite_encoder_file='model_files/student_gtsrb2tt100k_mse_encoder_lite.tflite',
                test=args.data)
    elif args.mode == 'whole':
        run_whole_model(original_encoder_file= f'model_files/student_{args.data}_{args.dloss}_whole.h5',
            test=args.data)
    elif args.mode == '1nn':
        run_model(
            original_encoder_file= f'model_files/student_{args.data}_{args.dloss}_encoder.h5',
            test=args.data)