import tensorflow as tf
from tensorflow.contrib import rnn
from utils import DataLoader

from __future__ import division

max_iter = 1000000
display_iter = 20
HORIZON = 10
BATCH_SIZE = 8
NUM_LSTM_CELL = 300
INPUT_DIM = 2
ENCODE_LEN = 8
SEQ_LEN = 20
LR = 0.001

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, 
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def conv_layer_1d(inputs, kernel_size, stride, num_features, idx, linear = False):
    """
    data is stored in the order of [batch, in_width, in_channels]
    """
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[2]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,input_channels,num_features],stddev=0.01, wd=True)
    biases = tf.get_variable('biases', [num_features], 
                        initializer=tf.constant_initializer(0.01))

    conv = tf.nn.conv1d(inputs, weights, stride=1, padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def network(inputs, hiddens):
    activations = []

    for kernel_size in range(1, HORIZON):
        with tf.name_scope("conv-{}".format(kernel_size)):
            activations.append(conv_layer_1d(inputs, kernel_size, 1, 128, "conv-{}".format(kernel_size)))

    # total_map = tf.concat(activations, 2)

    lstm_cell = rnn.BasicLSTMCell(NUM_LSTM_CELL, forget_bias=1.0, state_is_tuple=True)

    for i in range(len(hiddens)):
        if hiddens[i] is None:
            hiddens[i] = (tf.zeros([BATCH_SIZE,NUM_LSTM_CELL]),)*2
        activations[i], hiddens[i] = lstm_cell(activations[i], hiddens[i])

        # regression
        activations[i] = tf.layers.dense(activations[i], units=INPUT_DIM)

    return activations, hiddens

print 'making network template...'
network_template = tf.make_template('network', network)

with tf.Graph().as_default():
    X = tf.placeholder('float', [None, SEQ_LEN, HORIZON, INPUT_DIM])
    PEOPLE_IN = tf.placeholder(tf.int32)

    hiddens = [None]*HORIZON

    predictions = []
    # encode stage
    for i in range(ENCODE_LEN):
        activations, hiddens = network_template(X[:,i,:,:], hiddens)

    """
    TODO: try adding start signal for decoding
    """

    # decode stage
    for i in range(ENCODE_LEN, SEQ_LEN):
        activations, hiddens = network_template(X[:,i-1,:,:], hiddens)
        # stack activations into a tensor of shape [HORIZON, BATCH_SIZE, INPUT_DIM]
        activations = tf.stack(activations)
        predictions.append(activations)

    # stack predictions into a tensor of shape [DECODE_LEN, HORIZON, BATCH_SIZE, INPUT_DIM]
    predictions = tf.stack(predictions)
    # transpose predictions into a tensor of shape [BATCH_SIZE, DECODE_LEN, HORIZON, INPUT_DIM]
    predictions = tf.transpose(predictions, [2,0,1,3])

    # could still improve loss operation
    # loss_op = tf.nn.l2_loss(X[:,ENCODE_LEN:SEQ_LEN,:,:]-predictions[:,:,:,:])
    loss_op = tf.nn.l2_loss(X[:,ENCODE_LEN:SEQ_LEN,0,:]-predictions[:,:,0,:])

    # condition for while loop
    cond = lambda i, _: tf.less(i, PEOPLE_IN)

    # body for while loop
    def build_loss(i, loss_op):
        return [tf.add(i, 1), tf.add(loss_op, tf.nn.l2_loss(X[:,ENCODE_LEN:SEQ_LEN,i,:]-predictions[:,:,i,:]))]

    _, loss_op = tf.while_loop(cond, build_loss, [tf.constant(0), loss_op])

    avg_loss = loss_op / PEOPLE_IN
    tf.summary.scalar('avg loss', avg_loss)


    # training
    train_op = tf.train.AdamOptimizer(LR).minimize(loss_op)

    # summary
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initiallizer()

    """
    TODO: write data loader
    """
    dataloader = DataLoader(BATCH_SIZE, 21, [1,2,3,4], forcePreProcess=True)
    # loader = utils.DataLoader_ETH('/data/chiehent/data/ETH/ewap_dataset/seq_hotel/', batch_size=BATCH_SIZE)
    # loader.load_train_val_from_file('/data/chiehent/data/ETH/ewap_dataset/seq_hotel/obsmat_8_12.pkl')

    with tf.Session() as sess:
        print 'initializing network...'
        sess.run(init)

        merged_log = tf.summary.merge_all()
        log_writer = tf.summary.FileWriter('./training_log', sess.graph)

        print 'start training...'
        for step in range(1, max_iter+1):
            batch_x = loader.next_train_batch()

            """
            TODO: for each batch, do proximity reordering
            """

            sess.run(train_op, feed_dict={X:batch_x, PEOPLE_IN:num_people})

            if step % display_iter == 0 or step == 1:
                # report loss
                summary, loss = sess.run([merged_log, loss_iop], feed_dict={X: batch_x, PEOPLE_IN:num_people})
                log_writer.add_summary(summary, step)

                print('Iter {}, Minibatch Loss= {:.4f}'.format(step, loss))
    
            print("Optimization Finished!")
