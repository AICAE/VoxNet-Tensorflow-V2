import math, datetime, os
from voxnet import *
from volumetric_data import ShapeNet40Vox30

dataset = ShapeNet40Vox30()
voxnet = VoxNet()

p = dict() # placeholders

p['labels'] = tf.placeholder(tf.float32, [None, 40])

p['loss'] = tf.nn.softmax_cross_entropy_with_logits(logits=voxnet[-2], labels=p['labels'])
p['loss'] = tf.reduce_mean(p['loss'])
p['l2_loss'] = tf.add_n([tf.nn.l2_loss(w) for w in voxnet.kernels])

p['correct_prediction'] = tf.equal(tf.argmax(voxnet[-1], 1), tf.argmax(p['labels'], 1))
p['accuracy'] = tf.reduce_mean(tf.cast(p['correct_prediction'], tf.float32))

p['learning_rate'] = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
	p['train'] = tf.train.AdamOptimizer(p['learning_rate'], epsilon=1e-3).minimize(p['loss'])
p['weights_decay'] = tf.train.GradientDescentOptimizer(p['learning_rate']).minimize(p['l2_loss'])

# Hyperparameters

epoch_count = 10
batch_size = 64

initial_learning_rate = 0.001
min_learning_rate = 0.000001
learning_rate_decay_limit = 0.0001

num_batches_per_epoch = int(len(dataset.train) / float(batch_size))
#num_batches = 2147483647   # why, this is so big
num_batches = num_batches_per_epoch * epoch_count
print("num_batches_per_epoch = ", num_batches_per_epoch)
learning_decay = 10 * num_batches_per_epoch
weights_decay_after = 5 * num_batches_per_epoch

checkpoint_num = 0
learning_step = 0
min_loss = 1e308

def print_total_trainable_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(shape)
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print("total_trainable_parameters = ", total_parameters)

if not os.path.isdir('checkpoints'):
	os.mkdir('checkpoints')

with open('checkpoints/accuracies.txt', 'w') as f:
	f.write('')

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	print_total_trainable_parameters()

	for batch_index in range(num_batches):

		learning_rate = max(min_learning_rate,
			initial_learning_rate * 0.5**(learning_step / learning_decay))
		learning_step += 1

		if batch_index > weights_decay_after and batch_index % 256 == 0:
			session.run(p['weights_decay'], feed_dict=feed_dict)

		voxs, labels = dataset.train.get_batch(batch_size)
		feed_dict = {voxnet[0]: voxs, p['labels']: labels,
			p['learning_rate']: learning_rate, voxnet.training: True}

		session.run(p['train'], feed_dict=feed_dict)
		print("session run started for batch index = ", batch_index)
		if batch_index and batch_index % 512 == 0:

			print("{} batch: {}".format(datetime.datetime.now(), batch_index))
			print('learning rate: {}'.format(learning_rate))

			feed_dict[voxnet.training] = False
			loss = session.run(p['loss'], feed_dict=feed_dict)
			print('loss: {}'.format(loss))

			if (batch_index and loss > 1.5 * min_loss and
				learning_rate > learning_rate_decay_limit):
				min_loss = loss
				learning_step *= 1.2
				print("decreasing learning rate...")
			min_loss = min(loss, min_loss)

		if batch_index and batch_index % 2048 == 0:
			num_accuracy_batches = 30
			total_accuracy = 0
			for x in range(num_accuracy_batches):
				voxs, labels = dataset.train.get_batch(batch_size)
				feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
				total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
			training_accuracy = total_accuracy / num_accuracy_batches
			print('training accuracy: {}'.format(training_accuracy))

			num_accuracy_batches = 90
			total_accuracy = 0
			for x in range(num_accuracy_batches):
				voxs, labels = dataset.test.get_batch(batch_size)
				feed_dict = {voxnet[0]: voxs, p['labels']: labels, voxnet.training: False}
				total_accuracy += session.run(p['accuracy'], feed_dict=feed_dict)
			test_accuracy = total_accuracy / num_accuracy_batches
			print('test accuracy: {}'.format(test_accuracy))

			print('saving checkpoint {}...'.format(checkpoint_num))
			voxnet.npz_saver.save(session, 'checkpoints/c-{}.npz'.format(checkpoint_num))
			with open('checkpoints/accuracies.txt', 'a') as f:
				f.write(' '.join(map(str, (checkpoint_num, training_accuracy, test_accuracy)))+'\n')
			print('checkpoint saved!')

			checkpoint_num += 1