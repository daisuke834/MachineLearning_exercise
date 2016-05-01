# sklearn.datasets.load_iris()
* number of data sets: 150
* number of features: 2
* 10 fold cross validation
* Grid Search
  * C: 10 patterns
  * Gamma: 8 patterns (when Gaussian Kernel)

## Result
* SVM with Linear Kernel
  * Time = 2.0sec
  * Accuracy = 0.93
* SVM with Gaussian Kernel (RBF)
  * Time = 2.7sec
  * Accuracy = 0.96

![Linear Kernel](SVM_Linear_single.png)
![Gaussian Kernel](SVM_RBF.png)
