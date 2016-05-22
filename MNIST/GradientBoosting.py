#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import grid_search
import time

if __name__ == '__main__':
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")

	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = np.array(_mnist.data, dtype=float)
	_y = _mnist.target
	_X = _X[_array_index_rand]
	_y = _y[_array_index_rand]
	_num_dataset = len(_X)
	_rand_index = np.random.permutation(_num_dataset)
	_test_start_index = int(_num_dataset*0.8)
	_X_train = _X[_rand_index[:_test_start_index]]
	_y_train = _y[_rand_index[:_test_start_index]]
	_X_test = _X[_rand_index[_test_start_index:]]
	_y_test = _y[_rand_index[_test_start_index:]]
	_pixel_depth = 255.0
	_X_train_norm = _X_train/_pixel_depth
	_X_test_norm = _X_test/_pixel_depth
	print 'X:', _X.shape
	print 'y:', _y.shape
	print 'X(train):', _X_train.shape
	print 'y(train):', _y_train.shape
	print 'X(test):', _X_test.shape
	print 'y(test):', _y_test.shape

	_time_start = time.time()
	_num_of_tree_list = [1,3,10,30,100]
	print 'n_estimators:', len(_num_of_tree_list), _num_of_tree_list
	_param_grid = {'n_estimators':_num_of_tree_list}
	_grid = grid_search.GridSearchCV(ensemble.GradientBoostingClassifier(), param_grid=_param_grid, verbose=2, n_jobs=4)
	_grid.fit(_X_train_norm, _y_train)
	_time_end = time.time()
	print 'time for learning:'+str(_time_end-_time_start)+'sec ('+str((_time_end-_time_start)/60.0)+'min)'
	print 'Best Score (Valid Set)='+str(_grid.best_score_)+', Best Param='+str(_grid.best_params_)
	_num_of_tree_maxAccuracy = _grid.best_params_['n_estimators']
	_model = _grid.best_estimator_

	_scores = np.ndarray(len(_num_of_tree_list), dtype=float)
	for _parameters, _mean_validation_score, _cv_validation_scores in _grid.grid_scores_:
		print '\t',_parameters, '\t', _mean_validation_score
		_scores[_num_of_tree_list.index(_parameters['n_estimators'])] = _mean_validation_score

	_test_accuracy = _model.score(_X_test_norm, _y_test)
	print "Test Set: Accuracy="+str(_test_accuracy)
	print "Test Set: Error Rate="+str(1.0 - _test_accuracy)

	plt.plot(_num_of_tree_list, _scores)
	plt.title('Accuracy at Validation Set: Gradient Boosting')
	plt.xscale('log')
	plt.xlabel('Number of trees')
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
