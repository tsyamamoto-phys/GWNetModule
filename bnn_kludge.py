import numpy as np
from scipy.stats import norm
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as tfk 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def reset_session():
    """Creates a new global, interactive session in Graph-mode."""
    global sess
    try:
      tf.reset_default_graph()
      sess.close()
    except:
      pass
    sess = tf.InteractiveSession()

reset_session()


def model_function(x):
    return x[...,0] + 2*x[...,1]

def generate_1d_data(num_training_points, observation_noise_variance):
    """Generate noisy sinusoidal observations at a random set of points.

    Returns: observation_index_points, observations
    """
    index_points_ = np.random.uniform(-10.0, 10.0, (num_training_points, 2))
    # y = f(x) + noise
    observations_ = model_function(index_points_) + np.random.normal(loc = 0.0,
                                                                     scale = np.sqrt(observation_noise_variance),
                                                                     size = (num_training_points))
    return index_points_.astype(np.float32), observations_.astype(np.float32).reshape(-1,1)


# Generate training data with a known noise level (we'll later try to recover
# this value from the data).
NUM_TRAINING_POINTS = 8192 
observation_index_points_, observations_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=0.1)

NUM_VALIDATE_POINTS = 2048 
val_x, val_y = generate_1d_data(
    num_training_points=NUM_VALIDATE_POINTS,
    observation_noise_variance=0.1)

NUM_TEST_POINTS = 8 
test_x, test_y = generate_1d_data(
    num_training_points=NUM_TEST_POINTS,
    observation_noise_variance=0.1)


# build iterator
def build_input_pipeline(train_x, train_y, test_bs,
                         val_x, val_y, val_bs,
                         test_x, test_y):

    training_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    training_batches = training_dataset.shuffle(
        8192, reshuffle_each_iteration=True).repeat().batch(test_bs)
    training_iterator = tf.data.make_one_shot_iterator(training_batches)


    validate_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    validate_batches = validate_dataset.shuffle(
        512, reshuffle_each_iteration=True).repeat().batch(val_bs)
    validate_iterator = tf.data.make_one_shot_iterator(validate_batches)


    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_frozen = (test_dataset.take(8).repeat().batch(8))    
    test_iterator = tf.data.make_one_shot_iterator(test_frozen)


    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(
        handle, training_batches.output_types, training_batches.output_shapes)
    inputs, labels = feedable_iterator.get_next()

    return inputs, labels, handle, training_iterator, validate_iterator, test_iterator



"""
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(observation_index_points_[:,0],
        observation_index_points_[:,1],
        observations_[:,0],
        'o', markersize=3, alpha=0.4)
plt.show()
"""


(inputs, labels, handle,
 training_iterator, validate_iterator, test_iterator) = build_input_pipeline(
     observation_index_points_, observations_, 128,
     val_x, val_y, 512,
     test_x, test_y)




with tf.name_scope("bayesian_neural_net", values=[inputs]):
    net = tf.keras.Sequential([
        tfp.layers.DenseFlipout(20, activation=tf.nn.relu),
        tfp.layers.DenseFlipout(1)])

    predictions = net(inputs)
    pred_distribution = tfd.Normal(loc=predictions, scale=0.001)
# compute ELBO

neg_log_likelihood = tf.losses.mean_squared_error(labels, predictions)
kl = sum(net.losses)
elbo_loss = neg_log_likelihood + kl

# define the metrics
accuracy, accuracy_update_op = tf.metrics.mean_squared_error(
    labels=labels, predictions=predictions)


# set the train algorithm
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(elbo_loss)



# initialize all variables
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())



with tf.Session() as sess:

    sess.run(init_op)

    train_handle = sess.run(training_iterator.string_handle())
    val_handle = sess.run(validate_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    for step in range(50000):
        _ = sess.run([train_op, accuracy_update_op],
                     feed_dict = {handle: train_handle})

        if step % 100 == 0:
            loss_value, accuracy_value = sess.run(
                [elbo_loss, accuracy], feed_dict={handle: train_handle})
            print("Step: {:>3d} ELBO: {:.5f} MSE: {:.5f}".format(step, loss_value, accuracy_value))

            val_loss, val_acc = sess.run(
                [elbo_loss, accuracy], feed_dict={handle: val_handle})
            print("val_ELBO: {:.5f} val_MSE: {:.5f}".format(val_loss, val_acc))

    # test
    probs = np.asarray([sess.run((predictions), feed_dict={handle: test_handle})
                        for _ in range(1000)])
    mean_probs = np.mean(probs, axis=0)
    

for i in range(8):

    param = norm.fit(probs[:,i,0])
    x_min = param[0] - param[1]*3.0
    x_max = param[0] + param[1]*3.0
    x = np.linspace(x_min, x_max, 100)
    pdf_fitted = norm.pdf(x, loc=param[0], scale=param[1])
    pdf = norm.pdf(x)

    plt.figure()
    plt.hist(probs[:,i], bins=20, normed=True, alpha=0.7)
    plt.plot(x, pdf_fitted, c='r')
    plt.axvline(test_y[i], c='k')
    plt.title('%d th data' % (i+1))
plt.show()


