#Copyright (C) 2016 Daisuke Hashimoto. All Rights Reserved.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import datasets
import time

if __name__ == '__main__':
	_data = datasets.load_iris()
	_X = _data.data[:,2:4]
	_y = _data.target
	_X_names = _data.feature_names[2:4]
	_y_names = _data.target_names
	print 'X:', _X.shape
	print 'y:', _y.shape
	print 'number of points of dataset:', _X.shape[0]
	print 'number of dimension:', _X.shape[1]
	
	_time_start = time.time()
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

	_num_of_graph = len(_X_names) * (len(_X_names) -1) /2
	_num_of_col_graph = 3
	_num_of_row_graph = (_num_of_graph+_num_of_col_graph-1)/_num_of_col_graph
	_index = 1
	_colors = ['red', 'blue', 'green']
	for _y_axis in range(1,len(_X_names)):
		for _x_axis in range(_y_axis):
			#plt.subplot(_num_of_row_graph, _num_of_col_graph, _index)
			for _label_index in range(len(_y_names)):
				plt.scatter(_X[_y==_label_index, _x_axis], _X[_y==_label_index, _y_axis], label=_y_names[_label_index], color=_colors[_label_index])
			_xp = np.linspace(min(_X[:, _x_axis]), max(_X[:, _x_axis])*1.2,500)
			_yp = np.linspace(min(_X[:, _y_axis]), max(_X[:, _y_axis])*1.2,500)
			_Xmg, _Ymg = np.meshgrid(_xp, _yp)
			_XX = np.hstack( ( _Xmg.ravel().reshape(-1,1), _Ymg.ravel().reshape(-1,1) ) )
			_Z = _model.predict(_XX)
			_Z = _Z.reshape(_Xmg.shape)
			plt.contour(_Xmg, _Ymg, _Z, label='DB')
			plt.xlabel(_X_names[_x_axis])
			plt.ylabel(_X_names[_y_axis])
			plt.title('SVM Gaussian Kernel(RBF), C='+str(_C_maxAccuracy)+', gamma='+str(_gamma_maxAccuracy))
			plt.legend()
			_index = _index + 1
	plt.show()