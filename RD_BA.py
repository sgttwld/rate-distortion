"""
Blahut-Arimoto rate-distortion algorithm
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/rate-distortion
"""
import numpy as np
import tensorflow as tf
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def d(x,y):
    return (tf.expand_dims(x,1)-tf.expand_dims(y,0))**2           

def BA(X,beta,N):    
    epochs = 10000
    precision = 1e-4

    # UNIFORM DISCRETIZATION OF ran(X)
    Xhat = tf.constant(np.linspace(0,1,N))   # => q below is a dist. over xhat=[0,.01,...,.99,1] 

    # tf variable
    init = np.random.uniform(0,1,size=(N))
    q = tf.Variable(init/np.sum(init))

    # distortion(Xhat,X)
    dist = d(X,Xhat)

    t0 = time.time()
    # iterate
    for i in range(epochs):
        expdist = tf.expand_dims(q,0)*tf.exp(-beta*dist)
        post = expdist/tf.expand_dims(tf.reduce_sum(expdist,axis=1),1)
        q1 = tf.reduce_mean(post,0)
        if np.linalg.norm(q1.numpy()-q.numpy()) < precision:
            break
        q = q1
    t1 = time.time()

    return {
        'xhat': Xhat.numpy(),
        'q': q.numpy(), 
        'episodes': i, 
        'elapsed': t1-t0,
        'beta': beta,
        }
