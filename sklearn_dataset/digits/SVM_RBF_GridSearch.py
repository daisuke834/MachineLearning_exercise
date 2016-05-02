#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
import time

if __name__ == '__main__':
	_time_start = time.time()

	_digits = datasets.load_digits()
	_X = _digits.data
	_y = _digits.target

	print 'X:', _X.shape
	print 'y:', _y.shape

	print 'number of points of dataset:', _X.shape[0]
	print 'number of dimension:', _X.shape[1]

	_C_list = [1000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001]
	_gamma_list = [2**_i for _i in range(-12, -5)]
	_param_grid = {'C':_C_list, 'gamma':_gamma_list}
	_kfold = cross_validation.KFold(len(_X), n_folds=10)
	_grid = grid_search.GridSearchCV(svm.SVC(), param_grid=_param_grid, cv=_kfold, verbose=2, n_jobs=4)
	_grid.fit(_X, _y)
	print 'Best Score='+str(_grid.best_score_)+', Best Parm='+str(_grid.best_params_)
	_C_maxAccuracy = _grid.best_params_['C']
	_gamma_maxAccuracy = _grid.best_params_['gamma']
	_model = _grid.best_estimator_ 

	_time_end = time.time()
	print 'time for learning', (_time_end-_time_start)

	_p = np.random.random_integers(0, len(_X), 25)
	_samples = np.array(list(zip(_X,_y)))[_p]
	for _index, (_data, _label) in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data.reshape(8,8), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model.predict(_data.reshape(1,-1))
		plt.title(str(_label)+'/'+str(_predict), color='red')
	plt.show()
