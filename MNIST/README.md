# MNIST
## SVM, Gaussian Kernel (RBF)
### Lerning Condition
* number of features: 28x28 = 784
* number of training sets: 8000
* number of test sets: 62000
* 3 fold cross validation
* Grid Search
  * C: 15 patterns
  * Gamma: 7 patterns
### Results
* Training Results
  * Fitting 3 folds for each of 105 candidates, totalling 315 fits
  * Time of training = 4854sec (81min)
  * Best Score=0.935375
  * Best Parm={'kernel': 'rbf', 'C': 3.1622776601683791, 'gamma': 0.001}
* Test Results
  * Accuracy=0.942403225806
  * Error Rate=0.0575967741935

![Accuracy SVM RBF](SVM_RBF_accuracy.png)

Actual Value/Prediction
![Prediction SVM RBF](SVM_RBF.png)
