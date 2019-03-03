import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Discriminator Net
'''
First declare the input placeholder
In our case, we know that mnist is 28px by 28px with only one channel(grayscale)
So our input becomes a column vector
'''
X = tf.placeholder(tf.float32, [28*28, None], name="X")

D_W1 = tf.Variable(xavier_init([64, 28*28]), name="D_W1")
D_b1 = tf.Variable(tf.zeros([64, 1]), name="D_b1")
D_W2 = tf.Variable(xavier_init([1, 64]), name="D_W2")
D_b2 = tf.Variable(tf.zeros([1, 1]), name="D_b2")

theta_D = [D_W1, D_b1, D_W2, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, [100, None], name="Z")

G_W1 = tf.Variable(xavier_init([128, 100]), name="G_W1")
G_b1 = tf.Variable(tf.zeros([128, 1]), name="G_b1")
G_W2 = tf.Variable(xavier_init([28*28, 128]), name="G_W2")
G_b2 = tf.Variable(tf.zeros([28*28, 1]), name="G_b2")

theta_G = [G_W1, G_b1, G_W2, G_b2]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(D_W1, x) + D_b1)
    D_logit = tf.matmul(D_W2, D_h1) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(G_W1, z) + G_b1)
    G_log_prob = tf.matmul(G_W2, G_h1) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))


D_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

minibatch_size = 128
for iteration in range(100000):
    if iteration % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(100, 16)})

        fig = plot(samples.T)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_minibatch, _ = mnist.train.next_batch(minibatch_size)
    X_minibatch = X_minibatch.T
    if iteration % 2 == 0:
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_minibatch, Z:sample_Z(100, minibatch_size)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(100, minibatch_size)})

    if iteration % 1000 == 0:
        print('Iter: {}'.format(iteration))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
