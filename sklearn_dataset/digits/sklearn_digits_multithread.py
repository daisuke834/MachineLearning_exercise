import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
import time
import multiprocessing as mp

def learn( (_X, _y, _kfold, _C, _gamma) ):
	_scores = []
	for _train_index, _test_index in _kfold:
		_model = svm.SVC(C=_C, gamma=_gamma)
		_model.fit(_X[_train_index],_y[_train_index])
		_score = metrics.accuracy_score(_model.predict(_X[_test_index]), _y[_test_index])
		_scores.append(_score)
	_accuracy = (sum(_scores) / len(_scores)) * 100.0
	return _C, _gamma, _accuracy, _model
	
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
	_kfold = cross_validation.KFold(len(_X), n_folds=10)
	_parameter = []
	for _gamma in _gamma_list:
		for _C in _C_list:
			_parameter.append((_X, _y, _kfold, _C, _gamma))

	_pool = mp.Pool(4)
	_callback = _pool.map(learn, _parameter)

	_maxAccuracy = None
	_C_maxAccuracy = None
	_gamma_maxAccuracy = None
	_model_maxAccuracy = None
	for _C, _gamma, _accuracy, _model in _callback:		
		print 'gamma:'+str(_gamma)+'\tC:'+str(_C)+'\taccuracy: '+str(round(_accuracy,2))+'%'
		if _maxAccuracy is None or _accuracy>_maxAccuracy:
			_maxAccuracy = _accuracy
			_C_maxAccuracy = _C
			_gamma_maxAccuracy = _gamma
			_model_maxAccuracy = _model

	print 'Best Score: gamma='+str(_gamma_maxAccuracy)+'\tC='+str(_C_maxAccuracy)+'\tAccuracy='+str(round(_maxAccuracy,2))

	_time_end = time.time()
	print 'time for learning', (_time_end-_time_start)

	_p = np.random.random_integers(0, len(_X), 25)
	_samples = np.array(list(zip(_X,_y)))[_p]
	for _index, (_data, _label) in enumerate(_samples):
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data.reshape(8,8), cmap=cm.gray_r, interpolation='nearest')
		_predict = _model_maxAccuracy.predict(_data.reshape(1,-1))
		plt.title(str(_label)+'/'+str(_predict), color='red')
	plt.show()


