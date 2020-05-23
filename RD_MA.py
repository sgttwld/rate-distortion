import numpy as np
import tensorflow as tf
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def d(x,y,k=1):
    return (tf.expand_dims(x,1)-tf.expand_dims(y,0))**2           


def MA(X,beta,N):
    epochs = 10000
    precision = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
    # objective and gradients
    @tf.function 
    def opt(y,X):
        with tf.GradientTape() as g:
            expdist = tf.exp(-beta*d(X,y))                           
            logsumexp = tf.math.log(tf.reduce_mean(expdist,axis=1))     
            obj = -1/beta * tf.reduce_mean(logsumexp)                  
        gradients = g.gradient(obj, [y])
        optimizer.apply_gradients(zip(gradients, [y]))

    # tf variable
    y = tf.Variable(np.random.uniform(0,1,size=(N)))

    t0 = time.time()
    for i in range(epochs):
        y0 = y.numpy()
        opt(y,X)
        if (np.linalg.norm(y.numpy()-y0) < precision):
            break
    t1 = time.time()

    Z = tf.reduce_mean(tf.exp(-beta*d(X,y)),axis=1)
    D = tf.reduce_mean(tf.reduce_mean(tf.exp(-beta*d(X,y))*d(X,y),axis=1)/Z)
    R = -beta*D-tf.reduce_mean(tf.math.log(Z))
    return {
        'Xhat': y.numpy(), 
        'Distortion': D.numpy(), 
        'Rate': R.numpy(),
        'episodes': i, 
        'elapsed': t1-t0,
        'beta': beta,
        }


def MA_iter(X,beta,N):
    epochs = 10000
    precision = 1e-4
    
    # tf variable
    y = tf.Variable(np.random.uniform(0,1,size=(N)))

    t0 = time.time()
    for i in range(epochs):
        y0 = y.numpy()
        # 2  boltzmann dist
        expdist = tf.exp(-beta*(tf.expand_dims(y,0)-tf.expand_dims(X,1))**2)
        post = expdist/tf.expand_dims(tf.reduce_sum(expdist,axis=1),1)
        # 1  conditional expectation
        y.assign(tf.reduce_mean(tf.expand_dims(X,1)*post,axis=0)/tf.reduce_mean(post,axis=0))
        if (np.linalg.norm(y.numpy()-y0) < precision):
            break
    t1 = time.time()

    Z = tf.reduce_mean(tf.exp(-beta*d(X,y)),axis=1)
    D = tf.reduce_mean(tf.reduce_mean(tf.exp(-beta*d(X,y))*d(X,y),axis=1)/Z)
    R = -beta*D-tf.reduce_mean(tf.math.log(Z))
    return {
        'Xhat': y.numpy(), 
        'Distortion': D.numpy(), 
        'Rate': R.numpy(),
        'episodes': i, 
        'elapsed': t1-t0,
        'beta': beta,
        }
