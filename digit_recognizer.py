import pandas as pd
import numpy as np

# convert csv dataset files into dataframe
df_labeled = pd.read_csv('train.csv')
df_unlabeled = pd.read_csv('test.csv')


# shuffle training data just in case
from sklearn.utils import shuffle 

df_labeled = shuffle(df_labeled)

# split data between input and output numpy arrays
X = df_labeled.iloc[:,1:].values
y = df_labeled.iloc[:,0].values

X_unlabeled = df_unlabeled.values

# set to false to train using full dataset

X_train, y_train = (X, y)
split_data = False

if (split_data):
    # split labeled data into train, val and test
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4 )
#    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5 )


# train neural network  
from sklearn.neural_network import MLPClassifier

layers = (400) # set hidden layer shape

mlp = MLPClassifier(hidden_layer_sizes = layers, verbose = True, \
                    learning_rate='invscaling', solver='lbfgs')

mlp.fit(X_train,y_train)

pred = mlp.predict(X_unlabeled)

if (split_data):
    from sklearn.metrics import classification_report, confusion_matrix
    print confusion_matrix(y_test, pred) , '\n'
    print classification_report(y_test, pred)

else:
    df_pred = pd.DataFrame(pred)
    df_pred.to_csv('kaggle_submit.csv')


print 'holla'


