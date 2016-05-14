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
	print 'number of dimension:', _X.shape[1],'x', _X.shape[2]

	_num_dataset = len(_X)
	_rand_index = np.random.permutation(_num_dataset)
	_valid_start_index = int(_num_dataset*0.6)
	_test_start_index = int(_num_dataset*0.8)
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
	#_num_steps = 3001
	_num_steps = 11
	_batch_size = 128
	_patch_size = 5
	_depth = 16
	_num_hidden = 64
	_graph = tf.Graph()
	with _graph.as_default():
		_tf_X_train = tf.placeholder(tf.float32, shape=(_batch_size, _image_size, _image_size, 1))
		_tf_y_train = tf.placeholder(tf.float32, shape=(_batch_size, _num_labels))
		
		_tf_X_valid = tf.constant(_X_valid_norm, dtype=tf.float32)
		_tf_X_test  = tf.constant(_X_test_norm, dtype=tf.float32)
		
		_weights1 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, 1, _depth], stddev=0.1), dtype=tf.float32)
		_biases1 = tf.Variable(tf.zeros([_depth]), dtype=tf.float32)
		_weights2 = tf.Variable(tf.truncated_normal([_patch_size, _patch_size, _depth, _depth], stddev=0.1), dtype=tf.float32)
		_biases2 = tf.Variable(tf.constant(1.0, shape=[_depth]), dtype=tf.float32)
		_weights3 = tf.Variable(tf.truncated_normal([_image_size//4 * _image_size//4 * _depth, _num_hidden], stddev=0.1), dtype=tf.float32)
		_biases3 = tf.Variable(tf.constant(1.0, shape=[_num_hidden]), dtype=tf.float32)
		_weights4 = tf.Variable(tf.truncated_normal([_num_hidden, _num_labels], stddev=0.1), dtype=tf.float32)
		_biases4 = tf.Variable(tf.constant(1.0, shape=[_num_labels]), dtype=tf.float32)
		_tf_alpha = tf.placeholder(tf.float32)

		def model_drop(_data):
			_conv1 = tf.nn.relu(tf.nn.conv2d(_data,  _weights1, strides=[1,1,1,1], padding='SAME') + _biases1)
			_pool1 = tf.nn.max_pool(_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			_conv2 = tf.nn.relu(tf.nn.conv2d(_pool1, _weights2, strides=[1,1,1,1], padding='SAME') + _biases2)
			_pool2 = tf.nn.max_pool(_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
			_shape = _pool2.get_shape().as_list()
			_pool2_flat = tf.reshape(_pool2, [_shape[0], _shape[1]*_shape[2]*_shape[3]])
			_fully_connect1 = tf.nn.dropout(tf.nn.relu(tf.matmul(_pool2_flat, _weights3)+_biases3), 0.5)
			_read_out = tf.matmul(_fully_connect1, _weights4)+_biases4
			return _read_out

		_logits = model_drop(_tf_X_train)
		_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_logits, _tf_y_train))

		_optimizer = tf.train.AdagradOptimizer(_tf_alpha).minimize(_loss)

		def predict(_data):
			_predict=None
			for _index in range((len(_data.copy())+1023)/1024):
				_tmp_data = _data[_index*1024:(_index+1)*1024]
				_conv1 = tf.nn.relu(tf.nn.conv2d(_tmp_data,  _weights1, strides=[1,1,1,1], padding='SAME') + _biases1)
				_pool1 = tf.nn.max_pool(_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
				_conv2 = tf.nn.relu(tf.nn.conv2d(_pool1, _weights2, strides=[1,1,1,1], padding='SAME') + _biases2)
				_pool2 = tf.nn.max_pool(_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
				_shape = _pool2.get_shape().as_list()
				_pool2_flat = tf.reshape(_pool2, [_shape[0], _shape[1]*_shape[2]*_shape[3]])
				_fully_connect1 = tf.nn.relu(tf.matmul(_pool2_flat, _weights3)+_biases3)
				_read_out = tf.matmul(_fully_connect1, _weights4)+_biases4
				if _predict is None:
					_predict = tf.nn.softmax(_read_out).copy()
				else:
					_predict = np.vstack((_predict, tf.nn.softmax(_read_out).copy()))
			return _predict

		_train_prediction = tf.nn.softmax(_logits)
		_valid_prediction = predict(_tf_X_valid)
		_test_prediction = predict(_tf_X_test)

	def accuracy(_predictions, _labels):
		return ( np.sum(np.argmax(_predictions, 1) == np.argmax(_labels, 1)) / float(_predictions.shape[0]) )

	#***Learning****************************
	print '*****************************'
	print 'Start Learning'
	_time_start = time.time()
	_alpha_list = np.logspace(-2,1,7)
	_scores = np.ndarray(len(_alpha_list), dtype=float)
	with tf.Session(graph=_graph) as _session:
		_best_accuracy_valid = None
		_best_alpha = None
		for _al_index, _alpha in enumerate(_alpha_list):
			tf.initialize_all_variables().run()
			for _step in range(_num_steps):
				#_delta=100.0
				_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
				_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
				_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
				_feed_dict = {_tf_X_train : _batch_data, _tf_y_train : _batch_labels, _tf_alpha:_alpha}
				_, _l, _predictions = _session.run([_optimizer, _loss, _train_prediction], feed_dict=_feed_dict)
				if (_step % 500 == 0):
					_accuracy_valid = accuracy(_valid_prediction.eval(), _y_valid)
					if _best_accuracy_valid is None or _accuracy_valid>_best_accuracy_valid:
						#if _best_accuracy_valid is not None: _delta = _accuracy_valid - _best_accuracy_valid;
						_best_accuracy_valid = _accuracy_valid
						_best_alpha = _alpha
						#if _delta < 0.001: break;
			_accuracy_valid = accuracy(_valid_prediction.eval(), _y_valid)
			_scores[_al_index] = _accuracy_valid
			print 'alpha='+str(_alpha)+',\tValidAccuracy='+str(_accuracy_valid)

		tf.initialize_all_variables().run()
		for _step in range(_num_steps):
			_offset = (_step * _batch_size) % (_num_train_dataset - _batch_size)
			_batch_data		= _X_train_norm[_offset:(_offset + _batch_size), :]
			_batch_labels	= _y_train[_offset:(_offset + _batch_size)]
			_feed_dict = {_tf_X_train : _batch_data, _tf_y_train : _batch_labels, _tf_alpha:_best_alpha}
			_, _l, _predictions = _session.run([_optimizer, _loss, _train_prediction], feed_dict=_feed_dict)
		print 'End Learning'
		print 'Validation Best Accuracy:', _best_accuracy_valid
		print 'Validation Best Error Rate:', (1.0-_best_accuracy_valid)
		print 'Validation Best Alpha:', _best_alpha
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

	_p = np.random.random_integers(0, len(_X_test), 25)
	_samples = np.array(list(zip(_X_test,_y_test,_test_predict_final)))[_p]
	for _index, (_data, _y_val_test, _test_predict_final_val) in enumerate(_samples):
		_label = np.argmax(_y_val_test)
		_predicted_label = np.argmax(_test_predict_final_val)
		plt.subplot(5,5,_index+1)
		plt.axis('off')
		plt.imshow(_data, cmap=cm.gray_r, interpolation='nearest')
		#plt.imshow(_data.reshape(28,28), cmap=cm.gray_r, interpolation='nearest')
		plt.title(str(int(_label))+'/'+str(int(_predicted_label)), color='red')
	plt.show()

