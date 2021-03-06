import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import datasets
import time


if __name__ == '__main__':
	_num_labels=10
	_image_size = 28
	_pixel_depth = 255.0
	_mnist = datasets.fetch_mldata('MNIST original', data_home=".")
	_array_index_rand = np.random.permutation(range(len(_mnist.data)))
	_X = np.array(_mnist.data, dtype=float).reshape(-1, _image_size,_image_size, 1)
	_y = _mnist.target
	_y = (np.arange(_num_labels) == _y[:,None]).astype(np.float32)

	print 'X:', _X.shape
	print 'y:', _y.shape

	print 'number of points of dataset:', _X.shape[0]

	_num_dataset = len(_X)
	_rand_index = np.random.permutation(_num_dataset)
	_valid_start_index = int(_num_dataset*0.8)
	_test_start_index = int(_num_dataset*0.9)
	#_valid_start_index = int(1024)
	#_test_start_index = int(2048)
	_X_train = _X[_rand_index[:_valid_start_index]]
	_y_train = _y[_rand_index[:_valid_start_index]]
	_X_valid = _X[_rand_index[_valid_start_index:_test_start_index]]
	_y_valid = _y[_rand_index[_valid_start_index:_test_start_index]]
	_X_test = _X[_rand_index[_test_start_index:]]
	_y_test = _y[_rand_index[_test_start_index:]]
	_X_train_norm = _X_train/_pixel_depth
	_X_valid_norm = _X_valid/_pixel_depth
	_X_test_norm = _X_test/_pixel_depth
	_num_train_dataset = len(_X_train)
	_num_valid_dataset = len(_X_valid)
	_num_test_dataset = len(_X_test)
	print 'number of train dataset:', _num_train_dataset
	print 'number of valid dataset:', _num_valid_dataset
	print 'number of test dataset:', _num_test_dataset
	del(_rand_index)

	#*********Definition ********************
	_num_steps = 3001
	#_num_steps = 11
	_batch_size = 128
	_dummy_batch_labeldata = np.zeros((_batch_size,_num_labels))
	_patch_size = 5
	_depth = 16
	_num_hidden = 64
	_graph = tf.Graph()
	with _graph.as_default():
		_tf_lambda = tf.placeholder(tf.float32)
		_tf_alpha = tf.placeholder(tf.float32)
		_tf_keep_prob = tf.placeholder(tf.float32)
		_tf_X = tf.placeholder(tf.float32, shape=(_batch_size, _image_size, _image_size, 1))
		_tf_y = tf.placeholder(tf.float32, shape=(_batch_size, _num_labels))

		_weights1 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, 1, _depth], stddev=0.1), dtype=tf.float32)
		_biases1 = tf.Variable(tf.zeros([_depth]), dtype=tf.float32)
		_weights2 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, _depth, _depth], stddev=0.1), dtype=tf.float32)
		_biases2 = tf.Variable(tf.constant(1.0, shape=[_depth]), dtype=tf.float32)
		_weights3 = tf.Variable(tf.truncated_normal([_image_size//4 * _image_size//4 * _depth, _num_hidden], stddev=0.1), dtype=tf.float32)
		_biases3 = tf.Variable(tf.constant(1.0, shape=[_num_hidden]), dtype=tf.float32)
		_weights4 = tf.Variable(tf.truncated_normal([_num_hidden, _num_labels], stddev=0.1), dtype=tf.float32)
		_biases4 = tf.Variable(tf.constant(1.0, shape=[_num_labels]), dtype=tf.float32)

		def model():
			_conv1 = tf.nn.relu(tf.nn.conv2d(_tf_X,  _weights1, strides=[1,1,1,1], padding='SAME') + _biases1)
			_pool1 = tf.nn.max_pool(_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			_conv2 = tf.nn.relu(tf.nn.conv2d(_pool1, _weights2, strides=[1,1,1,1], padding='SAME') + _biases2)
			_pool2 = tf.nn.max_pool(_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			_shape = _pool2.get_shape().as_list()
			_pool2_flat = tf.reshape(_pool2, [_shape[0], _shape[1]*_shape[2]*_shape[3]])
			_fully_connect1 = tf.nn.dropout(tf.nn.relu(tf.matmul(_pool2_flat, _weights3)+_biases3), _tf_keep_prob)
			_read_out = tf.matmul(_fully_connect1, _weights4)+_biases4
			return _read_out
		def predict():
			return tf.nn.softmax(model())
		def num_of_errors(_num_of_data):
			_predicts = tf.slice(tf.argmax(tf.nn.softmax(model()),1), [0], [_num_of_data])
			_labels = tf.slice(tf.argmax(_tf_y, 1), [0], [_num_of_data])
			return tf.reduce_sum(tf.cast(tf.not_equal(_predicts, _labels), tf.int32))

		_tf_logits = model()
		_tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_tf_logits, _tf_y))
		_tf_loss_reg = _tf_loss + 0.5 * _tf_lambda * (tf.nn.l2_loss(_weights1) + tf.nn.l2_loss(_weights2) + tf.nn.l2_loss(_weights3) + tf.nn.l2_loss(_weights4))
		_tf_optimizer = tf.train.AdagradOptimizer(_tf_alpha).minimize(_tf_loss_reg)
		_tf_prediction = tf.nn.softmax(_tf_logits)

	#***Learning****************************
	print '*****************************'
	print 'Start Learning'
	_time_start = time.time()
	#_alpha_list = np.array([0.05, 5])
	#_lambda_list = np.array([0.0000001, 10])
	_alpha_list = np.logspace(-2,1,7)
	_lambda_list = np.logspace(-5,-1,9)
	_scores = np.ndarray( (len(_alpha_list), len(_lambda_list) ), dtype=float)
	for _al_index, _alpha in enumerate(_alpha_list):
		for _lam_index, _lambda in enumerate(_lambda_list):
			with tf.Session(graph=_graph) as _session:
				_session.run(tf.initialize_all_variables())
				for _step in range(_num_steps):
					_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
					_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
					_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
					_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_lambda:_lambda, _tf_alpha:_alpha, _tf_keep_prob:0.5}
					_, _l, _predictions = _session.run([_tf_optimizer, _tf_loss_reg, _tf_prediction], feed_dict=_feed_dict)
				_num_of_errors = 0
				for _step in range(int((_num_valid_dataset+_batch_size-1)/_batch_size)):
					_offset = _step * _batch_size
					_batch_data		= _X_valid_norm[_offset:(_offset + _batch_size), :]
					_batch_labels	= _y_valid[_offset:(_offset + _batch_size)]
					_real_batch_size = len(_batch_data)
					if _real_batch_size != _batch_size:
						_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
						_batch_data = np.vstack((_batch_data, _padding))
						_padding = np.zeros((_batch_size-_real_batch_size, _num_labels))
						_batch_labels = np.vstack((_batch_labels, _padding))
					_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_keep_prob:1}
					_current_num_of_errors = _session.run(num_of_errors(_real_batch_size), feed_dict=_feed_dict)
					_num_of_errors = _num_of_errors + _current_num_of_errors
				_accuracy_valid = float(_num_valid_dataset - _num_of_errors) / _num_valid_dataset
				_scores[_al_index, _lam_index] = _accuracy_valid
			print 'alpha='+str(_alpha)+',\tlambda='+str(_lambda)+',\tValidAccuracy='+str(_accuracy_valid)+',\tNum_of_errors='+str(_num_of_errors)+',\tNum_of_points='+str(_num_valid_dataset)

	_best_accuracy_valid = None
	_best_alpha = None
	_best_lambda = None
	for _al_index, _alpha in enumerate(_alpha_list):
		for _lam_index, _lambda in enumerate(_lambda_list):
			if _best_accuracy_valid is None or _scores[_al_index, _lam_index]>_best_accuracy_valid:
				_best_accuracy_valid = _scores[_al_index, _lam_index]
				_best_alpha = _alpha_list[_al_index]
				_best_lambda = _lambda_list[_lam_index]

	with tf.Session(graph=_graph) as _session:
		_session.run(tf.initialize_all_variables())
		for _step in range(_num_steps):
			_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
			_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
			_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
			_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_lambda:_best_lambda, _tf_alpha:_best_alpha, _tf_keep_prob:0.5}
			_, _l, _predictions = _session.run([_tf_optimizer, _tf_loss_reg, _tf_prediction], feed_dict=_feed_dict)
		_num_of_errors = 0
		for _step in range(int((_num_test_dataset+_batch_size-1)/_batch_size)):
			_offset = _step * _batch_size
			_batch_data		= _X_test_norm[_offset:(_offset + _batch_size), :]
			_batch_labels	= _y_test[_offset:(_offset + _batch_size)]
			_real_batch_size = len(_batch_data)
			if _real_batch_size != _batch_size:
				_padding = np.zeros((_batch_size-_real_batch_size, _image_size,_image_size, 1))
				_batch_data = np.vstack((_batch_data, _padding))
				_padding = np.zeros((_batch_size-_real_batch_size, _num_labels))
				_batch_labels = np.vstack((_batch_labels, _padding))
			_feed_dict = {_tf_X:_batch_data, _tf_y:_batch_labels, _tf_keep_prob:1}
			_current_num_of_errors = _session.run(num_of_errors(_real_batch_size), feed_dict=_feed_dict)
			_num_of_errors = _num_of_errors + _current_num_of_errors
		_accuracy_test = float(_num_test_dataset - _num_of_errors) / _num_test_dataset
		_p = np.random.random_integers(0, len(_X_test), _batch_size)
		_batch_data = _X_test_norm[_p]
		_feed_dict = {_tf_X:_batch_data, _tf_keep_prob:1}
		_batch_predict = _session.run(predict(), feed_dict=_feed_dict)
	_time_end = time.time()
	print 'End Learning'
	print '*****************************'
	print 'time for learning:', str(_time_end-_time_start) + 'sec\t(' + str((_time_end-_time_start)/60.0) + 'min)'
	print 'Validation Best Accuracy:', _best_accuracy_valid
	print 'Validation Best Error Rate:', (1.0-_best_accuracy_valid)
	print 'Validation Best Alpha:', _best_alpha
	print 'Test Accuracy:', _accuracy_test
	print 'Test Error Rate:', (1.0-_accuracy_test)

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

	_batch_predict		= _batch_predict[:25]
	_batch_data_origin	= _X_test[_p[:25]]
	_batch_labels		= _y_test[_p[:25]]
	for _index in range(25):
		_label				= np.argmax(_batch_labels[_index])
		_predicted_label	= np.argmax(_batch_predict[_index])
		_data = _batch_data_origin[_index].ravel().reshape(_image_size,_image_size)
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data, cmap=cm.gray_r, interpolation='nearest')
		plt.title(str(int(_label))+'/'+str(int(_predicted_label)), color='red')
	plt.show()
