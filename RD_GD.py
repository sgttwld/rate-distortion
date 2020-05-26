"""
Rate-distortion algorithm using gradient descent
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/rate-distortion
"""
import numpy as np
import tensorflow as tf
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def d(x,y):
    return (tf.expand_dims(x,1)-tf.expand_dims(y,0))**2           

def GD(X,beta,N):
    epochs = 10000
    precision = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=.1)

    def softmax(q):
        return tf.exp(q)/tf.reduce_sum(tf.exp(q))

    # UNIFORM DISCRETIZATION OF ran(X)
    Xhat = tf.constant(np.linspace(0,1,N))   # => q below is a dist. over xhat=[0,.01,...,.99,1] 

    # tf variable
    q = tf.Variable(np.random.uniform(0,1,size=(N)))

    # optimize free energy directly
    @tf.function 
    def opt(q,X):
        with tf.GradientTape() as g:
            qexp = tf.expand_dims(softmax(q),0)*tf.exp(-beta*d(X,Xhat))     # shape = (size(X),N)
            logsumexp = tf.math.log(tf.reduce_sum(qexp,axis=1))             # shape = (size(X),)
            obj = -1/beta * tf.reduce_mean(logsumexp)                       # shape = (1,)
        gradients = g.gradient(obj, [q])
        optimizer.apply_gradients(zip(gradients, [q]))

    t0 = time.time()
    for i in range(epochs):
        q0 = softmax(q).numpy()
        opt(q,X)
        if (np.linalg.norm(softmax(q).numpy()-q0) < precision):
            break
    t1 = time.time()

    qexp = tf.expand_dims(softmax(q),0)*tf.exp(-beta*d(X,Xhat))
    Z = tf.reduce_sum(qexp,axis=1)
    D = tf.reduce_mean(tf.reduce_sum(qexp*d(X,Xhat),axis=1)/Z)
    R = -beta*D-tf.reduce_mean(tf.math.log(Z))

    return {
        'xhat': Xhat.numpy(),
        'distortion': D.numpy(),
        'rate': R.numpy()/np.log(2),
        'q': softmax(q).numpy(), 
        'episodes': i, 
        'elapsed': t1-t0,
        'beta': beta,
        }
