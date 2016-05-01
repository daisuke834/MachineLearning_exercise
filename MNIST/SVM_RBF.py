import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
import time

def Xshaping(_X):
	_X_mean = np.mean(_X, axis=0).reshape(1, -1)
	_X_std = np.std(_X, axis=0).reshape(1, -1)
	_X_std[_X_std == 0] = 1
	_Xnorm = ( _X - _X_mean ) / _X_std
	return _Xnorm, _X_mean, _X_std

if __name__ == '__main__':
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")

	_time_start = time.time()
	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = _mnist.data
	_y = _mnist.target
	_X = _X[_array_index_rand]
	_y = _y[_array_index_rand]
	_X_train = _X[:6000]
	_y_train = _y[:6000]
	_X_test = _X[6000:]
	_y_test = _y[6000:]
	_X_train_norm, _X_mean, _X_std = Xshaping(_X_train)
	_X_test_norm = ( _X_test - _X_mean ) / _X_std
	#_X_train, _X_test, _y_train, _y_test = cross_validation.train_test_split(_X, _y, test_size=0.2, random_state=0)
	print 'X:', _X.shape
	print 'y:', _y.shape
	print 'X(train):', _X_train.shape
	print 'y(train):', _y_train.shape
	print 'X(test):', _X_test.shape
	print 'y(test):', _y_test.shape

	_C_list = [10**_i for _i in range(-2, 6)]
	_gamma_list = [10**_i for _i in range(-4, 0)]
	#_C_list = [10000]
	#_gamma_list = [0.01]
	_param_grid = {'C':_C_list, 'gamma':_gamma_list, 'kernel':['rbf']}
	_grid = grid_search.GridSearchCV(svm.SVC(), param_grid=_param_grid, verbose=2, n_jobs=4)
	_grid.fit(_X_train_norm, _y_train)
	print 'Best Score='+str(_grid.best_score_)+', Best Parm='+str(_grid.best_params_)
	_C_maxAccuracy = _grid.best_params_['C']
	_gamma_maxAccuracy = _grid.best_params_['gamma']
	_model = _grid.best_estimator_ 

	_time_end = time.time()
	print 'time for learning', (_time_end-_time_start)

	_accuracy = metrics.accuracy_score(_model.predict(_X_test_norm), _y_test)
	print "Test Set: Accuracy="+str(_accuracy)
	
	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples = np.array(list(zip(_X_test,_y_test)))[_p]
	for _index, (_data, _label) in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data.reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model.predict( ((_data - _X_mean)/_X_std).reshape(1,-1))
		plt.title(str(int(_label))+'/'+str(int(_predict)), color='red')
	plt.show()


