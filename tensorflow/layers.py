from __future__ import print_function

# Import MNIST data
import tensorflow as tf
import numpy as np

class retrieval_layer:
    def __init__(self, pre_node, bn_node, n_classes):
        self.pre_node = pre_node
        self.bn_node = bn_node
        self.n_classes = n_classes

        self.h3 = tf.get_variable("retrieval_W3", shape=[pre_node, bn_node],
            initializer=tf.contrib.layers.xavier_initializer())
        self.out = tf.get_variable("retrieval_out", shape=[bn_node, n_classes],
            initializer=tf.contrib.layers.xavier_initializer())

        self.b3 =  tf.Variable(tf.zeros([bn_node]))
        self.bout = tf.Variable(tf.zeros([n_classes]))
        self.var_dict = {
                'retrieval_w': self.h3,
                'retrieval_b': self.b3,
                'classifier_w': self.out,
                'classifier_b': self.bout,
                }

    def get_output(self, pre_layer, alpha = 1.0, no_norm = False, l2_loss = False):
        layer_3 = tf.add(tf.matmul(pre_layer, self.h3), self.b3)
        if not no_norm:
            layer_3 = tf.nn.l2_normalize(layer_3, dim = 1)
            norm_out = tf.nn.l2_normalize(self.out,dim = 0)
            #corr_mat = tf.matmul(tf.transpose(norm_out),norm_out)
            out_layer = tf.matmul(layer_3, norm_out)
            if l2_loss:
                out_layer = -1*tf.sqrt(2-2*out_layer)
            out_layer = alpha*out_layer

        else:
            if l2_loss:
                feature_norm = tf.reduce_sum(layer_3*layer_3,1)
                weights_norm = tf.reduce_sum(self.out*self.out,0)
                feature_norm_matrix = tf.tile(tf.expand_dims(feature_norm,1),[1,tf.size(weights_norm)])
                weights_norm_matrix = tf.tile(tf.expand_dims(weights_norm,0),[tf.size(feature_norm),1])
                cos_dis = tf.matmul(layer_3, self.out)
                out_layer = -1*tf.sqrt(feature_norm_matrix+weights_norm_matrix-2*cos_dis)
            else:
                out_layer = tf.matmul(layer_3, self.out) + self.bout
            #corr_mat = tf.matmul(tf.transpose(self.out),self.out)
        return out_layer, layer_3#, corr_mat# Construct model

def nca_loss(distance_matrix, one_hot_label):
    pos_dis = tf.reduce_sum(tf.exp(distance_matrix)*one_hot_label, axis = 1)
    neg_dis = tf.reduce_sum(tf.exp(distance_matrix)*(1.0-one_hot_label), axis = 1)
    loss = -1.0*tf.reduce_mean(tf.log(pos_dis/neg_dis))
    return loss

