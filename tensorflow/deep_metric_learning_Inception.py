import h5py
import numpy as np
import tensorflow as tf
import scipy.io
import collections
import random
import sys
import os
import GoogleNet_Model
import layers
from tqdm import tqdm
import argparse
from Loggers import Logger, FileLogger
import inception_v1
from sklearn.cluster import KMeans
import sklearn

slim = tf.contrib.slim
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', default='/home/xuzhang/project/Medifor/data/car196/', help='folder to image data')
parser.add_argument('--log_dir', default='../tensorflow_log/', help='folder to output log')
parser.add_argument("--no_norm", action="store_true",
                    help="Norm or not")
parser.add_argument("--l2_loss", action="store_true",
                    help="l2 or inner product")
parser.add_argument("--nca", action="store_true",
                    help="l2 or inner product")
parser.add_argument("--data_augment", action="store_true",
                    help="l2 or inner product")
parser.add_argument("--normed_test", action="store_true",
                    help="l2 or inner product")
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--embedding_dim', default=64, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--alpha', default=1.0, type=float, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--training_batch_size', default=100, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--test_batch_size', default=100, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--display_step', default=20, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--nb_epoch', default=20, type=int, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

embedding_dim = args.embedding_dim

training_batch_size = args.training_batch_size

test_batch_size = args.test_batch_size

default_image_size = 224
offset = (256-default_image_size)/2

def get_feature(img_data):
    ptr = 0
    num_sample = img_data.shape[0]
    embedding = np.zeros((num_sample,args.embedding_dim))
    for step in tqdm(range(int(num_sample / args.test_batch_size)+1)):
        real_size = min(args.test_batch_size, num_sample-ptr)
        if real_size <= 0:
            break
        if args.data_augment:
            img = np.zeros((real_size, default_image_size, default_image_size, 3),\
                    dtype = np.float32)
            img[:,:,:,:] = img_data[ptr:ptr+real_size,\
                    offset:(offset+default_image_size),\
                    offset:(offset+default_image_size),:]
        else:
            img = img_data[ptr:ptr + real_size, :, :, :]
        feature = sess.run([test_bottleneck],feed_dict={img_place_holder: img})
        embedding[ptr:ptr + real_size, :] = feature[0]
        ptr += real_size
    return embedding

def eval_nmi(embedding, label, num_category):
    if args.normed_test:
        for i in range(embedding.shape[0]):
            embedding[i,:] = embedding[i,:]/np.sqrt(np.sum(embedding[i,:] ** 2)+1e-4)

    kmeans = KMeans(n_clusters=num_category)
    kmeans.fit(embedding)
    y_kmeans_pred = kmeans.predict(embedding)
    nmi = sklearn.metrics.normalized_mutual_info_score(label, y_kmeans_pred)
    return nmi

def read_data(name, image_mean):
    # reading matlab v7.3 file using h5py. it has struct with img as a member
    with h5py.File(args.data_dir+"{}_cars196_256resized.mat".format(name)) as f:
        original_img_data = [f[element[0]][:] for element in f['{}_images/img'.format(name)]]
        class_id = [f[element[0]][:] for element in f['{}_images/class_id'.format(name)]]
    #original_img_data = np.zeros((100,3,256,256))
    #class_id = np.array(range(100))
    original_img_data = np.asarray(original_img_data)
    original_img_data = np.transpose(original_img_data[:, [2, 1, 0], :, :], (0, 1, 3, 2))
    original_img_data = np.transpose(original_img_data, (0, 2, 3, 1))
    original_img_data = np.float32(original_img_data)
    class_id = np.asarray(class_id)
    class_id = class_id[:, 0, 0]
    class_id = class_id-np.amin(class_id)
    
    num_training_sample = original_img_data.shape[0]
    num_training_category = np.unique(class_id).shape[0]
    
    one_hot_label = np.zeros((num_training_sample, num_training_category))
    img_data = np.zeros((num_training_sample,default_image_size,default_image_size,3))
    
    for i in tqdm(range(num_training_sample)):
        one_hot_label[i, int(class_id[i])] = 1.0
        original_img_data[i,:,:,:] = np.float32(original_img_data[i,:,:,:])/255.0 # - np.float32(image_mean)
        img_data[i,:,:,:] = original_img_data[i, offset:(offset+default_image_size),\
                    offset:(offset+default_image_size), :]
    print('{} dataset shape: {}'.format(name, img_data.shape))

    return img_data, original_img_data, class_id, one_hot_label

suffix = 'car196'


if args.no_norm:
    suffix = suffix + '_no_norm'
    if args.normed_test:
        suffix = suffix + '_normed_test'
else:
    suffix = suffix + '_alpha_{:1.1f}'.format(args.alpha)

if args.l2_loss:
    suffix = suffix + '_l2'

if args.nca:
    suffix = suffix + '_nca'

if args.data_augment:
    suffix = suffix+'_da'

image_mean = scipy.io.loadmat('../data/imagenet_mean.mat')
image_mean = image_mean['image_mean']
image_mean = np.transpose(image_mean, (1, 2, 0))

print 'reading training data'
if args.data_augment:
    _, img_data, class_id, one_hot_label = read_data('training', image_mean)
    _, valid_img_data, valid_class_id, _ = read_data('validation', image_mean)
else:
    img_data, _, class_id, one_hot_label = read_data('training', image_mean)
    valid_img_data, _, valid_class_id, _ = read_data('validation', image_mean)

num_training_sample = img_data.shape[0]
num_training_category = np.unique(class_id).shape[0]
num_valid_category = np.unique(valid_class_id).shape[0]

print('constructing model')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

img_place_holder = tf.placeholder(tf.float32, [None, default_image_size, default_image_size, 3])
label_place_holder = tf.placeholder("float", [None, num_training_category])
#google_net_model= GoogleNet_Model.GoogleNet_Model()
with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    net_output = inception_v1.inception_v1(img_place_holder)
    test_net_output = inception_v1.inception_v1(img_place_holder,\
            reuse = True, is_training = False) 

saver = tf.train.Saver()

with tf.variable_scope('retrieval'):
    retrieval_layer = layers.retrieval_layer(1024, embedding_dim, num_training_category)

out_layer, bottleneck = retrieval_layer.get_output(net_output,\
        alpha = args.alpha, no_norm = args.no_norm, l2_loss = args.l2_loss)
test_out_layer, test_bottleneck = retrieval_layer.get_output(test_net_output,\
        alpha = args.alpha, no_norm = args.no_norm, l2_loss = args.l2_loss)

prediction = tf.nn.softmax(out_layer)

if args.nca:
    loss_op = layers.nca_loss(out_layer, label_place_holder)
else:
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	    logits=out_layer, labels=label_place_holder))

train_op1 = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.94).minimize(loss_op,
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV1'))
train_op2 = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.94).minimize(loss_op, var_list=retrieval_layer.var_dict.values())
train_op = tf.group(train_op1, train_op2)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(label_place_holder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('done constructing model')

print('initialize variables')
sess.run(tf.global_variables_initializer())
saver.restore(sess, '../data/inception_v1.ckpt')
print('done initializing')

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)
logger = Logger(args.log_dir + suffix)

global_step = 0
log_step = 0
for epoch in range(args.nb_epoch):
    ptr = 0
    for step in tqdm(range(int(num_training_sample / training_batch_size)+1)):
        real_size = min(training_batch_size, num_training_sample-ptr)
        if real_size <= 0:
            break

        if args.data_augment:
            img = np.zeros((real_size, default_image_size,\
                    default_image_size, 3), dtype = np.float32)
            flip_flag = np.random.randint(2, size=real_size)
            x_offset = np.random.randint(offset*2, size=real_size)
            y_offset = np.random.randint(offset*2, size=real_size)
            for i in range(real_size):
                img[i,:,:,:] = img_data[ptr+i,y_offset[i]:(y_offset[i]+default_image_size),\
                            x_offset[i]:(x_offset[i]+default_image_size),:]
                if flip_flag[i]:
                    img[i,:,:,:] = img[i,:,::-1,:]
        else:
            img = img_data[ptr:ptr + real_size, :, :, :] 

        label = one_hot_label[ptr:ptr + real_size, :]
        _, cost, accu = sess.run([train_op, loss_op, accuracy],\
                        feed_dict={img_place_holder: img, label_place_holder: label})

        global_step += 1
        ptr += training_batch_size
        if (step+1)%args.display_step == 0:
            logger.log_value('loss', cost, step = log_step)
            logger.log_value('acc', accu, step = log_step)
            log_step = log_step + 1

    training_embedding = get_feature(img_data) 
    training_nmi = eval_nmi(training_embedding, class_id, num_training_category)
    logger.log_value('Training NMI', training_nmi, step = log_step)

    valid_embedding = get_feature(valid_img_data) 
    validation_nmi = eval_nmi(valid_embedding, valid_class_id, num_valid_category)
    logger.log_value('Test NMI', validation_nmi, step = log_step)

    
#saver = tf.train.Saver()
#saver.save(self.sess, '../tensorflow_model/model.ckpt')
#print('extracting feature from training data')
#results = np.zeros((8000, 64), dtype=np.float32)
#
    
# print results.shape
