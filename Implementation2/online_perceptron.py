import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUMBER_OF_ITERATION = 15
def get_labels(k):
   label = ['label']
   for i in range(1,k+1):
       label.append(str(i))
   return label
def add_bias(data):
   data['785'] = np.array([1]*data.shape[0])
   return data
def kp(x,y,p=1):
   value = 1 + np.matmul(x,np.transpose(y))
   value = np.power(value,p)
   return value
def get_Y_and_X(data):
   y = data['label'].apply(lambda x: 1 if x==3 else -1)
   x = data.drop(['label'],1)
   return x,y

def online_train():
    train_df,Y = get_Y_and_X(add_bias(pd.read_csv('pa2_train.csv',names=get_labels(784))))
    valid_df, VY = get_Y_and_X(add_bias(pd.read_csv('pa2_valid.csv',names=get_labels(784))))

    (n, features) = train_df.shape

    (vn, vfeatures) = valid_df.shape

    W = [0 for x in range(0, features)]
    W = np.array(W)
    iters = 15; _iter = 0;
    training_error = []
    validation_error = []
    train_accuracy = []
    validation_accuracy = []
    weightsMap = {}
    maxAccuracy = 0
    maxAccuracyIndex = 0
    while _iter < iters:
        error = 0;
        v_error = 0;
        for i in range(0, n):
            x = train_df.iloc[i]
            u = W.dot(x)
            yi = Y.iloc[i]
            if yi*u <= 0:
                W = np.add(W,np.multiply(yi,x))
        for i in range(0, n):
            x = train_df.iloc[i]
            u = W.dot(x)
            yi = Y.iloc[i]
            if yi*u <= 0:
                error+=1 
        weightsMap[_iter] = W
        training_error.append(error)
        t_accuracy = 1-(error/n);
        train_accuracy.append(t_accuracy)
        for i in range(0, vn):
            vx = valid_df.iloc[i]
            vu = np.array([W]).dot(vx)
            vyi = VY.iloc[i]
            if vyi*vu <= 0:
                v_error += 1;
        validation_error.append(v_error)
        v_accuracy = 1-(v_error/vn);
        validation_accuracy.append(v_accuracy)
        _iter+=1
    return weightsMap

    # t, = plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], train_accuracy, label="train")
    # v, = plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], validation_accuracy, label="validation")
    # plt.xlabel('iterations')
    # plt.ylabel('accuracy')
    # plt.legend(handles=[t,v], loc='best')
    # print(validation_accuracy)

def online_predict(weightsMap):
    accurate_weights=weightsMap[13]
    predict_df = add_bias(pd.read_csv('pa2_test_no_label.csv',names=get_labels(784)[1:]))
    a = (np.array([accurate_weights]))
    y_ = a.dot(np.transpose(predict_df))
    op = pd.DataFrame(np.sign(y_)).T
    op.to_csv("oplabel___.csv", index=False, header=False)

online_predict(online_train())