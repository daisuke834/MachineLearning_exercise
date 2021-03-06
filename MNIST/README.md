Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
# MNIST
## Summary
* Convolutional Neural Network (LeNet-5)
  * time = 310min (GPU)
  * Test Error Rate = 0.017
* Neural Network (3 layer)
  * time = 48min (GPU)
  * Test Error Rate = 0.0356428571429
* Random Forest
  * time = 13min
  * Test Error Rate = 0.0294285714286
* xgboost
  * time = 188min
  * Test Error Rate = 0.0250714285714
* SVM Gaussian Kernel (RBF) (Batch Analysis)
  * time = 55min
  * Test Error Rate = 0.0320161290323
* SVM Linear Kernel (Batch Analysis)
  * time = 2.0min
  * Test Error Rate = 0.131366666667
* SVM Linear Kernel (Stochastic Gradient Descent)
  * time = 2.6min
  * Test Error Rate = 0.105357142857
* Logistic Regression (Batch Analysis)
  * time = 5.9min
  * Test Error Rate = 0.107933333333

## Convolutional Neural Network: LeNet-5 (Tensor Flow)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 56000
* number of validation sets: 7000
* number of test sets: 7000
* dropout: 0.5
* Grid Search
  * Learning Rate: 7 patterns
  * Lambda: 9 patterns

### Results
* Validation Results
  * Time of training(GPU) = 18614sec (310min)
  * Best Score=0.988285714286
  * Best Parm={'alpha':0.1, 'lambda':0.000316227766017}
* Test Results
  * Accuracy=0.983
  * Error Rate=0.017

![Accuracy CNN Lenet-5](TensorFlow_CNN_LeNet5_accuracy.png)

Actual Value/Prediction
![Prediction CNN Lenet-5](TensorFlow_CNN_LeNet5_prediction.png)

## 3-layer Neural Network (Tensor Flow)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 42000
* number of validation sets: 14000
* number of test sets: 14000
* number of hidden node: 1024
* dropout: 0.5
* Grid Search
  * Learning Rate: 7 patterns
  * Lambda: 11 patterns

### Results
* Validation Results
  * Time of training(GPU) = 2908sec (48min)
  * Best Score=0.965928571429
  * Best Parm={'alpha':0.1,	'lambda':0.01}
* Test Results
  * Accuracy=0.964357142857
  * Error Rate=0.0356428571429

![Accuracy 3-layer NN](TensorFlow_NN_3_accuracy.png)

Actual Value/Prediction
![Prediction 3-layer NN](TensorFlow_NN_3_prediction.png)

## Random Forest
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 56000
* number of test sets: 14000
* 3-fold cross validation
* Grid Search
  * n_estimators: [1, 3, 10, 30, 100, 300, 1000]

### Results
* Validation Results
  * Time of training = 765sec (12.7min)
  * Best Score=0.967303571429
  * Best Parm={'n_estimators': 1000}
* Test Results
  * Accuracy=0.970571428571
  * Error Rate=0.0294285714286

![Accuracy Random Forest](RandomForest_accuracy.png)

Actual Value/Prediction
![Prediction Random Forest](RandomForest_prediction.png)

## xgboost
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 56000
* number of test sets: 14000
* 3-fold cross validation
* Grid Search
  * n_estimators: [1, 3, 10, 30, 100, 300, 1000]

### Results
* Validation Results
  * Time of training = 11305sec (188min)
  * Best Score=0.973375
  * Best Parm={'n_estimators': 1000}
* Test Results
  * Accuracy=0.974928571429
  * Error Rate=0.0250714285714

![Accuracy xgboost](xgboost_learn_accuracy.png)

Actual Value/Prediction
![Prediction xgboost](xgboost_learn_prediction.png)

## SVM Gaussian Kernel (RBF) (Batch Analysis)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 8000
* number of test sets: 62000
* 3-fold cross validation
* Grid Search
  * C: 10 patterns
  * Gamma: 10 patterns

### Results
* Training Results
  * Time of training = 3317sec (55min)
  * Best Score=0.960875
  * Best Parm={'C': 10.0, 'gamma': 0.021544346900318822}
* Test Results
  * Accuracy=0.967983870968
  * Error Rate=0.0320161290323

![Accuracy SVM RBF](SVM_RBF_accuracy.png)

Actual Value/Prediction
![Prediction SVM RBF](SVM_RBF_prediction.png)

## SVM Linear Kernel (Batch Analysis)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 20000
* number of validation sets: 20000
* number of test sets: 30000
* Grid Search
  * C: 15 patterns

### Results
* Validation Results
  * Time of training = 118sec (2.0min)
  * Best Score=0.91225
  * Best Parm={'C': 0.01}
* Test Results
  * Accuracy=0.868633333333
  * Error Rate=0.131366666667

![Accuracy SVM Linear](SVM_Linear_accuracy.png)

Actual Value/Prediction
![Prediction SVM Linear](SVM_Linear_prediction.png)

## SVM Linear Kernel (Stochastic Gradient Descent)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 42000
* number of validation sets: 14000
* number of test sets: 14000
* Grid Search
  * C: 15 patterns

### Results
* Training Results
  * Time of training = 154sec (2.6min)
  * Best Score=0.915571428571
  * Best Parm={'C': 3162.27766017}
* Test Results
  * Accuracy=0.894642857143
  * Error Rate=0.105357142857

![Accuracy SVM Linear SGD](SVM_Linear_SGD_accuracy.png)

Actual Value/Prediction
![Prediction SVM Linear SGD](SVM_Linear_SGD_prediction.png)


## Logistic Regression (Batch Analysis)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 20000
* number of validation sets: 20000
* number of test sets: 30000
* Grid Search
  * C: 15 patterns

### Results
* Training Results
  * Time of training = 356sec (5.9min)
* Validation Results
  * Best Score=0.9098
  * Best Parm={'C': 0.316227766017}
* Test Results
  * Accuracy=0.892066666667
  * Error Rate=0.107933333333

![Accuracy LogisticRegression](LogisticRegression_accuracy.png)

Actual Value/Prediction
![Prediction LogisticRegression](LogisticRegression_prediction.png)

