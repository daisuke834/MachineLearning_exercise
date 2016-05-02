#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
from sklearn import preprocessing
import time

if __name__ == '__main__':
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")

	_time_start = time.time()
	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = np.array(_mnist.data, dtype=float)
	_y = _mnist.target
	_X = _X[_array_index_rand]
	_y = _y[_array_index_rand]
	_X_train, _X_test, _y_train, _y_test = cross_validation.train_test_split(_X, _y, test_size=0.2, random_state=0)
	_scaler = preprocessing.StandardScaler()
	_scaler.fit(_X_train)
	_X_train_norm = _scaler.transform(_X_train)
	_X_test_norm = _scaler.transform(_X_test)
	print 'X:', _X.shape
	print 'y:', _y.shape
	print 'X(train):', _X_train.shape
	print 'y(train):', _y_train.shape
	print 'X(test):', _X_test.shape
	print 'y(test):', _y_test.shape

	_C_list = list(np.logspace(-2,5,15))
	_param_grid = {'C':_C_list}
	_grid = grid_search.GridSearchCV(linear_model.LogisticRegression(), param_grid=_param_grid, verbose=2, n_jobs=4)
	_grid.fit(_X_train_norm, _y_train)
	_time_end = time.time()
	print 'time for learning', (_time_end-_time_start)
	
	print 'Best Score='+str(_grid.best_score_)+', Best Parm='+str(_grid.best_params_)
	_C_maxAccuracy = _grid.best_params_['C']
	_model = _grid.best_estimator_ 

	_scores = np.ndarray( (len(_C_list)), dtype=float)
	for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
		print '\t',_parameters, '\t', _mean_validation_score
		_scores[_C_list.index(_parameters['C'])] = _mean_validation_score

	_accuracy = metrics.accuracy_score(_model.predict(_X_test_norm), _y_test)
	print "Test Set: Accuracy="+str(_accuracy)
	print "Test Set: Error Rate="+str(1.0 - _accuracy)
	
	plt.plot(_C_list, _scores)
	plt.title('Accuracy at Test Set: Logistic Regression')
	plt.xscale('log')
	plt.xlabel('C')
	plt.ylabel('Accuracy at Test Set')
	plt.show()

	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples = np.array(list(zip(_X_test,_y_test)))[_p]
	for _index, (_data, _label) in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data.reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model.predict( _scaler.transform(_data.reshape(1,-1)) )
		plt.title(str(int(_label))+'/'+str(int(_predict)), color='red')
	plt.show()


