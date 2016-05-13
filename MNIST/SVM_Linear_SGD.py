#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import linear_model
import multiprocessing as mp
from sklearn import preprocessing
import time

def learn( (_X_train, _y_train, _X_valid, _y_valid, _C, _C_index) ):
	_lambda = 1.0/_C
	_model = linear_model.SGDClassifier(alpha=_lambda, n_iter=50)
	_model.fit(_X_train, _y_train)
	_score = _model.score(_X_valid, _y_valid)
	return _score, _model, _C_index

if __name__ == '__main__':
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")

	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = np.array(_mnist.data, dtype=float)
	_y = _mnist.target
	_X = _X[_array_index_rand]
	_y = _y[_array_index_rand]
	_num_dataset = len(_X)
	_rand_index = np.random.permutation(_num_dataset)
	_valid_start_index = int(_num_dataset*0.6)
	_test_start_index = int(_num_dataset*0.8)
	#_valid_start_index = int(20000)
	#_test_start_index  = int(40000)
	_X_train = _X[_rand_index[:_valid_start_index]]
	_y_train = _y[_rand_index[:_valid_start_index]]
	_X_valid = _X[_rand_index[_valid_start_index:_test_start_index]]
	_y_valid = _y[_rand_index[_valid_start_index:_test_start_index]]
	_X_test = _X[_rand_index[_test_start_index:]]
	_y_test = _y[_rand_index[_test_start_index:]]
	_pixel_depth = 255.0
	_X_train_norm = _X_train/_pixel_depth
	_X_valid_norm = _X_valid/_pixel_depth
	_X_test_norm = _X_test/_pixel_depth
	print 'X:', _X.shape
	print 'y:', _y.shape
	print 'X(train):', _X_train.shape
	print 'y(train):', _y_train.shape
	print 'X(valid):', _X_valid.shape
	print 'y(valid):', _y_valid.shape
	print 'X(test):', _X_test.shape
	print 'y(test):', _y_test.shape

	_time_start = time.time()
	_num_of_C_grids = 15
	_C_list = np.logspace(-2,5,_num_of_C_grids)
	_scores = np.ndarray(_num_of_C_grids)
	_parameters = []
	print 'C:', len(_C_list), _C_list
	for _index, _C in enumerate(_C_list):
		_parameters.append((_X_train_norm, _y_train, _X_valid_norm, _y_valid, _C, _index))
	_pool = mp.Pool(4)
	_callback = _pool.map(learn, _parameters)
	_best_accuracy = None
	for _score, _model, _C_index in _callback:		
		_scores[_C_index] = _score
		if _best_accuracy is None or _score>_best_accuracy:
			_best_accuracy = _score
			_best_C = _C_list[_C_index]
			_best_model = _model
	_time_end = time.time()
	print 'time for learning:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
	print 'Best Score (Valid Set)='+str(_best_accuracy)+', C='+str(_best_C)

	_test_accuracy = _model.score(_X_test_norm, _y_test)
	print "Test Set: Accuracy="+str(_test_accuracy)
	print "Test Set: Error Rate="+str(1.0 - _test_accuracy)
	
	plt.plot(_C_list, _scores)
	plt.title('Accuracy at Validation Set: SVM Linear SGD')
	plt.xscale('log')
	plt.xlabel('C')
	plt.ylabel('Accuracy at Validation Set')
	plt.show()

	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples = np.array(list(zip(_X_test,_y_test)))[_p]
	for _index, (_data, _label) in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data.reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model.predict(_data.reshape(1,-1)/_pixel_depth)
		plt.title(str(int(_label))+'/'+str(int(_predict)), color='red')
	plt.show()


