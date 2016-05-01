# MNIST
## SVM, Gaussian Kernel (RBF)
### Condition
* number of features: 28x28 = 784
* number of training sets: 6000
* number of test sets: 64000
* 3 fold cross validation
* Grid Search
  * C: 8 patterns
  * Gamma: 4 patterns

### Result
* Training Set
  * Fitting 3 folds for each of 32 candidates, totalling 96 fits
  * Time of training = 900 sec
  * Best Score=0.934666666667
  * Best Parm={'kernel': 'rbf', 'C': 100, 'gamma': 0.001}
* Test Set
  * Accuracy=0.93946875

![SVM_RBF](SVM_RBF.png)
