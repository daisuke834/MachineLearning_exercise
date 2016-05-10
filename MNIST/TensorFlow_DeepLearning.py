import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
import time


if __name__ == '__main__':
	_num_labels=10
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")
	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = np.array(_mnist.data, dtype=float)
	_y = _mnist.target
	_y = (np.arange(_num_labels) == _y[:,None]).astype(np.float32)

	print 'X:', _X.shape
	print 'y:', _y.shape

	print 'number of points of dataset:', _X.shape[0]
	print 'number of dimension:', _X.shape[1]
	_num_of_features = _X.shape[1]

	_num_dataset = len(_X)
	_rand_index = np.random.permutation(_num_dataset)
	_valid_start_index = int(_num_dataset*0.6)
	_test_start_index = int(_num_dataset*0.8)
	#_valid_start_index = int(1000)
	#_test_start_index = int(2000)
	_X_train = _X[_rand_index[:_valid_start_index]]
	_y_train = _y[_rand_index[:_valid_start_index]]
	_X_valid = _X[_rand_index[_valid_start_index:_test_start_index]]
	_y_valid = _y[_rand_index[_valid_start_index:_test_start_index]]
	_X_test = _X[_rand_index[_test_start_index:]]
	_y_test = _y[_rand_index[_test_start_index:]]
	_scaler = preprocessing.StandardScaler()
	_scaler.fit(_X_train)
	_X_train = _scaler.transform(_X_train)
	_X_valid = _scaler.transform(_X_valid)
	_X_test = _scaler.transform(_X_test)
	_num_train_dataset = len(_X_train)
	_num_valid_dataset = len(_X_valid)
	_num_test_dataset = len(_X_test)
	print 'number of train dataset:', _num_train_dataset
	print 'number of valid dataset:', _num_valid_dataset
	print 'number of test dataset:', _num_test_dataset
	del(_rand_index)

	#*********Definition ********************
	_n_of_hidden_nodes = 1024
	_num_steps = 3001
	_batch_size = 128
	_graph = tf.Graph()
	with _graph.as_default():
		_tf_X_train = tf.placeholder(tf.float32, shape=(_batch_size, _num_of_features))
		_tf_y_train = tf.placeholder(tf.float32, shape=(_batch_size, _num_labels))
		
		_tf_X_valid = tf.constant(_X_valid, dtype=tf.float32)
		_tf_X_test  = tf.constant(_X_test, dtype=tf.float32)
		
		_weights1 = tf.Variable(tf.truncated_normal([_num_of_features, _n_of_hidden_nodes]), dtype=tf.float32)
		_biases1 = tf.Variable(tf.zeros([_n_of_hidden_nodes]), dtype=tf.float32)
		_weights2 = tf.Variable(tf.truncated_normal([_n_of_hidden_nodes, _num_labels]), dtype=tf.float32)
		_biases2 = tf.Variable(tf.zeros([_num_labels]), dtype=tf.float32)
		_tf_lambda = tf.placeholder(tf.float32)
		_tf_alpha = tf.placeholder(tf.float32)

		_hidden_hypo = tf.nn.relu(tf.matmul(_tf_X_train, _weights1) + _biases1)
		_logits = tf.matmul(_hidden_hypo, _weights2) + _biases2
		_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_logits, _tf_y_train))+ 0.5 * _tf_lambda * (tf.nn.l2_loss(_weights1) + tf.nn.l2_loss(_weights2))

		_optimizer = tf.train.AdagradOptimizer(_tf_alpha).minimize(_loss)

		_train_prediction = tf.nn.softmax(_logits)
		_valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(_tf_X_valid, _weights1) + _biases1), _weights2) + _biases2)
		_test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(_tf_X_test, _weights1) + _biases1), _weights2) + _biases2)

	def accuracy(_predictions, _labels):
		return ( np.sum(np.argmax(_predictions, 1) == np.argmax(_labels, 1)) / float(_predictions.shape[0]) )

	#***Learning****************************
	print '*****************************'
	print 'Start Learning'
	_time_start = time.time()
	_alpha_list = np.logspace(-2,1,7)
	_lambda_list = np.logspace(-3,1,9)
	_scores = np.ndarray( (len(_alpha_list), len(_lambda_list) ), dtype=float)
	with tf.Session(graph=_graph) as _session:
		_best_accuracy_valid = None
		_best_alpha = None
		_best_lambda = None
		for _al_index, _alpha in enumerate(_alpha_list):
			for _lam_index, _lambda in enumerate(_lambda_list):
				tf.initialize_all_variables().run()
				for _step in range(_num_steps):
					_delta=100.0
					_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
					_batch_data		= _X_train[_offset:(_offset + _batch_size), :]
					_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
					_feed_dict = {_tf_X_train : _batch_data, _tf_y_train : _batch_labels, _tf_lambda: _lambda, _tf_alpha:_alpha}
					_, _l, _predictions = _session.run([_optimizer, _loss, _train_prediction], feed_dict=_feed_dict)
					if (_step % 500 == 0):
						_accuracy_valid = accuracy(_valid_prediction.eval(), _y_valid)
						if _best_accuracy_valid is None or _accuracy_valid>_best_accuracy_valid:
							if _best_accuracy_valid is not None: _delta = _accuracy_valid - _best_accuracy_valid;
							_best_accuracy_valid = _accuracy_valid
							_best_alpha = _alpha
							_best_lambda = _lambda
							if _delta < 0.001: break;
				_accuracy_valid = accuracy(_valid_prediction.eval(), _y_valid)
				_scores[_al_index, _lam_index] = _accuracy_valid
				print 'alpha='+str(_alpha)+',\tlambda='+str(_lambda)+',\tValidAccuracy='+str(_accuracy_valid)

		tf.initialize_all_variables().run()
		for _step in range(_num_steps):
			_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
			_batch_data		= _X_train[_offset:(_offset + _batch_size), :]
			_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
			_feed_dict = {_tf_X_train : _batch_data, _tf_y_train : _batch_labels, _tf_lambda: _best_lambda, _tf_alpha:_best_alpha}
			_, _l, _predictions = _session.run([_optimizer, _loss, _train_prediction], feed_dict=_feed_dict)
		print 'End Learning'
		print 'Validation Best Accuracy:', _best_accuracy_valid
		print 'Validation Best Error Rate:', (1.0-_best_accuracy_valid)
		print 'Validation Best Alpha:', _best_alpha
		print 'Validation Best Lambda:', _best_lambda
		print '*****************************'
		print 'Test Accuracy:', accuracy(_test_prediction.eval(), _y_test)
		print 'Test Error Rate:', (1.0-accuracy(_test_prediction.eval(), _y_test))
		_test_predict_final = _test_prediction.eval().copy()

		_time_end = time.time()
		print 'time for learning:', str(_time_end-_time_start) + 'sec\t(' + str((_time_end-_time_start)/60.0) + 'min)'

	plt.subplot(1, 2, 1)
	for _al_index, _alpha in enumerate(_alpha_list):
		plt.plot(_lambda_list, _scores[_al_index, :], label='Alpha='+str(_alpha))
	plt.title('Accuracy at Valid Set')
	plt.xscale('log')
	plt.xlabel('lambda')
	plt.ylabel('Accuracy at Valid Set')
	plt.legend(loc='lower right')

	plt.subplot(1, 2, 2)
	for _lam_index, _lambda in enumerate(_lambda_list):
		plt.plot(_alpha_list, _scores[:, _lam_index], label='lambda='+str(_lambda))
	plt.title('Accuracy at Valid Set')
	plt.xscale('log')
	plt.xlabel('alpha')
	plt.ylabel('Accuracy at Valid Set')
	plt.legend(loc='lower right')
	plt.show()

