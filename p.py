import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from pdb import set_trace
import tensorflow_datasets as tfds

class CNNet(object):
  def __init__(self, learning_rate=0.001, input_dim = 28, num_class=10):
    # Make hyperparameters instance variables. 
    self.learning_rate = learning_rate
    self.num_class = num_class
    self.input_dim = input_dim
 
    self.initializer = tf.keras.initializers.glorot_uniform()
    self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
    
    # Set Random seed for Tensorflow.
    self.random_seed = 42
    tf.random.set_seed(self.random_seed)

  def network(self, X, activations=False):
    with tf.variable_scope('network', initializer=self.initializer):
        # Define the layers.
        self.layers = [
            tf.layers.Conv2D(filters=16, kernel_size=3,
                                     strides=(1, 1), activation='relu',padding='SAME'),
            tf.layers.Conv2D(filters=32, kernel_size=3,
                                     strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                                     strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.num_class)
        ]
        
        # Store activations for investigation later.
        activations_list = []
        
        # Forward pass loop, store intermediate activations.
        out = X
        for layer in self.layers:
          out = layer(out)
          activations_list.append(out)
        
        if activations:
          return out, activations_list
        else:
          return out, tf.nn.softmax(out)

  def model(self, X, y):
    set_trace()
    # Get the logits from the network.
    out_logits, _ = self.network(X)
   
    # Calculate Cross Entropy loss.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=out_logits))
    
    # Perform backprop wrt loss and update network variables.
    # Instead of doing optimizer.minimize(loss), explicitly defining
    # which variables are trained.
    grads = self.optimizer.compute_gradients(loss)
    
    vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                  scope="network")
    grad_list = [(g, v) for g, v in grads if v in vars_list]
    optimize_op = self.optimizer.apply_gradients(grad_list)
    
    return loss, optimize_op, out_logits
  
  def metrics(self, y, logits, verbose=True):
    # Get prediction values and flatten.
    y = np.argmax(y, axis=1).flatten()
    y_ = np.argmax(logits, axis=1).flatten()

    confusion = confusion_matrix(y_true=y, y_pred=y_)
    accuracy = accuracy_score(y_true=y, y_pred=y_)
    
    if verbose:
      print ("accuracy score: ", accuracy) 
      
    return accuracy
  
  def train(self, train_X, train_y, test_X, test_y, 
            batch_size=256, epochs=100):
    # Clear deafult graph stack and reset global graph definition.
    set_trace()
    tf.reset_default_graph()
    
    # GPU config.  
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    # Placeholders for tensors.
    X = tf.placeholder(shape=[None, self.input_dim, self.input_dim, 1], dtype=tf.float32)
    y = tf.placeholder(shape=[None, self.num_class], dtype=tf.float32)
    
    # Get the ops for training the model.
    loss, optimize, out_logits = self.model(X, y)
     
    self.saver = tf.train.Saver()
    
    # Initialize session.
    with tf.Session(config=config) as sess:
      # Initialize the variables in the graph.
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      
      # Stochastic Gradient Descent loop.
      for step in range(epochs):
        # Total number of batch and start index.
        num_train_batches, start = int(train_X.shape[0]/batch_size), 0

        for _ in range(num_train_batches):
            # Indexes for batch selection.
            end = start + batch_size         
            limit = end if end < train_X.shape[0] else train_X.shape[0]
            idx = np.arange(start, limit)
            
            # Run optimization op with batch.
            _, step_loss = sess.run([optimize, loss], 
                                    {X: train_X[idx], y: train_y[idx]})
            start = end
        
        print('='*80+'\nEpoch: {0} Training Loss: {1}'.format(step, step_loss))
        
        # Get probabilities and report metrics.
        probs = sess.run(tf.nn.softmax(out_logits), {X: test_X, y: test_y})
        acc = self.metrics(test_y, probs)
        
        self.saver.save(sess, "model.ckpt")
        
      # Get and save representation space for training set.
      probs = sess.run(out_logits, {X: train_X})
      np.save('representations.npy', probs)
      
      return step_loss, acc

  def predict(self, X_test, logits=False, reps=False):
    tf.reset_default_graph()
    tf.set_random_seed(42)

    X = tf.placeholder(shape=[None, self.input_dim, self.input_dim, 1], dtype=tf.float32)
    
    # Get the ops for running inference on the model.
    out_logits, out_probs = self.network(X)
    
    saver = tf.train.Saver()
    # Initialize a new session on the graph.
    with tf.Session() as sess:
      
        # Load the trained model into the session to run inference.
        saver.restore(sess, "model.ckpt")
        # Get 
        rep_logits, probs = sess.run([out_logits, out_probs], {X: X_test})
    
    preds = np.argmax(probs, axis=1).flatten()
    if logits:
      return preds, probs
    elif reps:
      return preds, rep_logits
    else:
      return preds

set_trace()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

set_trace()

cnn = CNNet()
