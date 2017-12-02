import h5py
import numpy as np
import tensorflow as tf
import scipy.io
import collections
import random
import sys

def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return np.divide(scoreMatExp,np.tile(scoreMatExp.sum(1).reshape((113,1)),(1,11318)))

class CNN_Triplet_Metric(object):
    def __init__(self,sess):
        np.set_printoptions(threshold=np.nan)
        self.var_dict,self.fc_var_dict = self.Variables_Dict()
        img = tf.placeholder(tf.float32, [None, 227, 227, 3])
        y = tf.placeholder(tf.float32, [None, 11318])
        self.sess = sess

        print 'loding image mean file'
        image_mean = scipy.io.loadmat('imagenet_mean.mat')
        image_mean = image_mean['image_mean']
        print 'image mean shape:' + str(image_mean.shape)
        image_mean = np.transpose(image_mean,(1,2,0))
        image_mean = np.expand_dims(image_mean,axis=0)
        print 'transformed image mean shape:' + str(image_mean.shape)
        print 'done loading image mean file'

        # reading matlab v7.3 file using h5py. it has struct with img as a member
        print 'reading training data'
        img_data = np.memmap('train_set_for_tf.dat',dtype=np.uint8,mode='r',shape=(59551,3,256,256))
 
        class_id = np.memmap('train_label_for_tf.dat',dtype=np.uint8,mode='r',shape=(59551,1))
        print 'done reading training data'


        print 'constructing model'
        logits,tt1 = self.CNN_Metric_Model(img)
        prediction = tf.nn.softmax(logits)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=y))
        tf.summary.scalar('cost', loss_op)
        train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_op,var_list=self.var_dict.values())
        train_op2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op,var_list=self.fc_var_dict.values())
        train_op = tf.group(train_op1,train_op2)
        # train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_op)


        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        print 'done constructing model'

        print 'initialize variables'
        self.sess.run(tf.global_variables_initializer())
        print 'done initializing'


        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train', self.sess.graph)
        global_step = 0
        batch_size = 113
        total_no_train = 59551
        imean = np.tile(image_mean, (batch_size, 1, 1, 1))
        for epoch in range(200):
            ptr = 0
            for step in range(int(total_no_train/batch_size)):
                inp,opt = img_data[ptr:ptr+batch_size,:,:,:],class_id[ptr:ptr+batch_size,:]

                inp = np.transpose(inp[:, [2, 1, 0], :, :], (0, 1, 3, 2))
                inp = np.transpose(inp, (0, 2, 3, 1))
                class_label = np.zeros((batch_size, 11318))
                for idx, val in enumerate(opt):
                    class_label[idx, int(val) - 1] = 1.0


                inp = np.float32(inp) - np.float32(imean)
                inp = inp[:,14:241,14:241,:]

                cost,_,m,accu,pred,_logits = self.sess.run([loss_op,train_op,merged,accuracy,prediction,logits], feed_dict={img: inp,y:class_label})
                train_writer.add_summary(m,global_step)
                global_step +=1
                ptr += batch_size
                print("Epoch:", '%04d' % (epoch),"Step:", '%04d' % (global_step), "cost=", "{:.9f}".format(cost), "accu=", "{:.9f}".format(accu))
                if global_step%100   == 0:
                    break

        saver = tf.train.Saver()
        saver.save(self.sess,'/tmp/model.ckpt')
        sys.exit()
        print 'read test data'
        with h5py.File("validation_images_crop15_square256.mat") as f:
            img_data = [f[element[0]][:] for element in f['validation_images/img']]
        print 'done reading'
        img_data = np.asarray(img_data)
        img_data = np.transpose(img_data[:, [2, 1, 0], :, :], (0, 1, 3, 2))
        img_data = np.transpose(img_data, (0, 2, 3, 1))

        # print 'the imgag data size is: ' + str(img_data.shape)

        results = np.zeros((60500, 64),dtype=np.float32)

        ptr = 0
        no_of_batches_test = int(img_data.shape[0] / 100)
        for k in range(no_of_batches_test):
            print ptr
            inp = img_data[ptr:ptr + 100,:,:,:]
            imean = np.tile(image_mean, (100, 1, 1, 1))
            inp = np.float32(inp) - np.float32(imean)
            inp = inp[:, 14:241, 14:241, :]

            embeded_feat = self.sess.run([tt1], feed_dict={img: inp})
            results[ptr:ptr+100,:] = embeded_feat[0]
            ptr += 100

        # test_idty = np.ones((1,256,256,3),dtype='f8')
        # test_idty -= image_mean
        # test_idty = test_idty[:, 14:241, 14:241, :]
        # #test_idty = np.random.rand(1,227,227,3)
        # with h5py.File('test_idty.h5','w') as H:
        #     H.create_dataset('img', data=np.transpose(test_idty,(0,3,1,2)))
        # with open('test_h5_idty_list.txt','w') as L:
        #     L.write( '/home/wei/deep_metric/test_idty.h5' )
        # embeded_feat = self.sess.run([a_output, tt1], feed_dict={img_a: test_idty})
        # scipy.io.savemat('test_tf.mat',mdict={'test':embeded_feat[1]})
        # print embeded_feat[1].shape
        # print embeded_feat[1][0, :, :,5]
        np.savetxt("results_tf_64_after_training.csv", results, delimiter=",")
        # print results.shape

    def Variables_Dict(self):
        print 'Loading MAT file for pretrained'
        pretrained_weights = scipy.io.loadmat('tf_ckpt_from_caffe.mat')

        Conv2d_1a_7x7 = tf.constant(np.transpose(pretrained_weights['conv1/7x7_s2'],(2,3,1,0)))
        Conv2d_2b_1x1 = tf.constant(np.transpose(pretrained_weights['conv2/3x3_reduce'],(2,3,1,0)))
        Conv2d_2c_3x3 = tf.constant(np.transpose(pretrained_weights['conv2/3x3'],(2,3,1,0)))

        Conv2d_1a_7x7_bias = tf.constant(pretrained_weights['conv1/7x7_s2_bias'].flatten())
        Conv2d_2b_1x1_bias = tf.constant(pretrained_weights['conv2/3x3_reduce_bias'].flatten())
        Conv2d_2c_3x3_bias = tf.constant(pretrained_weights['conv2/3x3_bias'].flatten())
        # first inception
        Mixed_3b_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3a/1x1'],(2,3,1,0)))
        Mixed_3b_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3a/3x3_reduce'], (2,3, 1, 0)))
        Mixed_3b_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_3a/3x3'], (2,3, 1, 0)))
        Mixed_3b_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3a/5x5_reduce'], (2,3, 1, 0)))
        Mixed_3b_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_3a/5x5'], (2,3, 1, 0)))
        Mixed_3b_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3a/pool_proj'], (2,3, 1, 0)))
        # first inception bias
        Mixed_3b_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3a/1x1_bias'].flatten())
        Mixed_3b_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3a/3x3_reduce_bias'].flatten())
        Mixed_3b_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_3a/3x3_bias'].flatten())
        Mixed_3b_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3a/5x5_reduce_bias'].flatten())
        Mixed_3b_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_3a/5x5_bias'].flatten())
        Mixed_3b_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_3a/pool_proj_bias'].flatten())
        # second inception
        Mixed_3c_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3b/1x1'], (2,3, 1, 0)))
        Mixed_3c_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3b/3x3_reduce'], (2,3, 1, 0)))
        Mixed_3c_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_3b/3x3'], (2,3, 1, 0)))
        Mixed_3c_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3b/5x5_reduce'], (2,3, 1, 0)))
        Mixed_3c_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_3b/5x5'], (2,3, 1, 0)))
        Mixed_3c_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_3b/pool_proj'], (2,3, 1, 0)))
        # second inception bias
        Mixed_3c_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3b/1x1_bias'].flatten())
        Mixed_3c_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3b/3x3_reduce_bias'].flatten())
        Mixed_3c_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_3b/3x3_bias'].flatten())
        Mixed_3c_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_3b/5x5_reduce_bias'].flatten())
        Mixed_3c_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_3b/5x5_bias'].flatten())
        Mixed_3c_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_3b/pool_proj_bias'].flatten())
        # third inception
        Mixed_4b_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4a/1x1'], (2,3, 1, 0)))
        Mixed_4b_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4a/3x3_reduce'], (2,3, 1, 0)))
        Mixed_4b_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_4a/3x3'], (2,3, 1, 0)))
        Mixed_4b_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4a/5x5_reduce'], (2,3, 1, 0)))
        Mixed_4b_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_4a/5x5'], (2,3, 1, 0)))
        Mixed_4b_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4a/pool_proj'], (2,3, 1, 0)))
        # third inception bias
        Mixed_4b_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4a/1x1_bias'].flatten())
        Mixed_4b_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4a/3x3_reduce_bias'].flatten())
        Mixed_4b_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_4a/3x3_bias'].flatten())
        Mixed_4b_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4a/5x5_reduce_bias'].flatten())
        Mixed_4b_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_4a/5x5_bias'].flatten())
        Mixed_4b_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_4a/pool_proj_bias'].flatten())
        # fourth inception
        Mixed_4c_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4b/1x1'], (2,3, 1, 0)))
        Mixed_4c_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4b/3x3_reduce'], (2,3, 1, 0)))
        Mixed_4c_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_4b/3x3'], (2,3, 1, 0)))
        Mixed_4c_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4b/5x5_reduce'], (2,3, 1, 0)))
        Mixed_4c_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_4b/5x5'], (2,3, 1, 0)))
        Mixed_4c_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4b/pool_proj'], (2,3, 1, 0)))
        # fourth inception bias
        Mixed_4c_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4b/1x1_bias'].flatten())
        Mixed_4c_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4b/3x3_reduce_bias'].flatten())
        Mixed_4c_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_4b/3x3_bias'].flatten())
        Mixed_4c_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4b/5x5_reduce_bias'].flatten())
        Mixed_4c_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_4b/5x5_bias'].flatten())
        Mixed_4c_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_4b/pool_proj_bias'].flatten())
        # fifth inception
        Mixed_4d_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4c/1x1'], (2,3, 1, 0)))
        Mixed_4d_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4c/3x3_reduce'], (2,3, 1, 0)))
        Mixed_4d_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_4c/3x3'], (2,3, 1, 0)))
        Mixed_4d_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4c/5x5_reduce'], (2,3, 1, 0)))
        Mixed_4d_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_4c/5x5'], (2,3, 1, 0)))
        Mixed_4d_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4c/pool_proj'], (2,3, 1, 0)))
        # fifth inception bias
        Mixed_4d_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4c/1x1_bias'].flatten())
        Mixed_4d_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4c/3x3_reduce_bias'].flatten())
        Mixed_4d_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_4c/3x3_bias'].flatten())
        Mixed_4d_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4c/5x5_reduce_bias'].flatten())
        Mixed_4d_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_4c/5x5_bias'].flatten())
        Mixed_4d_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_4c/pool_proj_bias'].flatten())
        # sixth inception
        Mixed_4e_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4d/1x1'], (2,3, 1, 0)))
        Mixed_4e_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4d/3x3_reduce'], (2,3, 1, 0)))
        Mixed_4e_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_4d/3x3'], (2,3, 1, 0)))
        Mixed_4e_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4d/5x5_reduce'], (2,3, 1, 0)))
        Mixed_4e_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_4d/5x5'], (2,3, 1, 0)))
        Mixed_4e_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4d/pool_proj'], (2,3, 1, 0)))
        # sixth inception bias
        Mixed_4e_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4d/1x1_bias'].flatten())
        Mixed_4e_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4d/3x3_reduce_bias'].flatten())
        Mixed_4e_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_4d/3x3_bias'].flatten())
        Mixed_4e_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4d/5x5_reduce_bias'].flatten())
        Mixed_4e_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_4d/5x5_bias'].flatten())
        Mixed_4e_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_4d/pool_proj_bias'].flatten())
        # seventh inception
        Mixed_4f_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4e/1x1'], (2,3, 1, 0)))
        Mixed_4f_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4e/3x3_reduce'], (2,3, 1, 0)))
        Mixed_4f_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_4e/3x3'], (2,3, 1, 0)))
        Mixed_4f_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4e/5x5_reduce'], (2,3, 1, 0)))
        Mixed_4f_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_4e/5x5'], (2,3, 1, 0)))
        Mixed_4f_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_4e/pool_proj'], (2,3, 1, 0)))
        # seventh inception bias
        Mixed_4f_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4e/1x1_bias'].flatten())
        Mixed_4f_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4e/3x3_reduce_bias'].flatten())
        Mixed_4f_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_4e/3x3_bias'].flatten())
        Mixed_4f_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_4e/5x5_reduce_bias'].flatten())
        Mixed_4f_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_4e/5x5_bias'].flatten())
        Mixed_4f_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_4e/pool_proj_bias'].flatten())
        # eighth inception
        Mixed_5b_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5a/1x1'], (2,3, 1, 0)))
        Mixed_5b_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5a/3x3_reduce'], (2,3, 1, 0)))
        Mixed_5b_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_5a/3x3'], (2,3, 1, 0)))
        Mixed_5b_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5a/5x5_reduce'], (2,3, 1, 0)))
        Mixed_5b_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_5a/5x5'], (2,3, 1, 0)))
        Mixed_5b_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5a/pool_proj'], (2,3, 1, 0)))
        # eighth inception bias
        Mixed_5b_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5a/1x1_bias'].flatten())
        Mixed_5b_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5a/3x3_reduce_bias'].flatten())
        Mixed_5b_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_5a/3x3_bias'].flatten())
        Mixed_5b_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5a/5x5_reduce_bias'].flatten())
        Mixed_5b_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_5a/5x5_bias'].flatten())
        Mixed_5b_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_5a/pool_proj_bias'].flatten())
        #ninth inception
        Mixed_5c_Branch_0_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5b/1x1'], (2,3, 1, 0)))
        Mixed_5c_Branch_1_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5b/3x3_reduce'], (2,3, 1, 0)))
        Mixed_5c_Branch_1_Conv2d_0b_3x3 = tf.constant(np.transpose(pretrained_weights['inception_5b/3x3'], (2,3, 1, 0)))
        Mixed_5c_Branch_2_Conv2d_0a_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5b/5x5_reduce'], (2,3, 1, 0)))
        Mixed_5c_Branch_2_Conv2d_0b_5x5 = tf.constant(np.transpose(pretrained_weights['inception_5b/5x5'], (2,3, 1, 0)))
        Mixed_5c_Branch_3_Conv2d_0b_1x1 = tf.constant(np.transpose(pretrained_weights['inception_5b/pool_proj'], (2,3, 1, 0)))
        # ninth inception bias
        Mixed_5c_Branch_0_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5b/1x1_bias'].flatten())
        Mixed_5c_Branch_1_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5b/3x3_reduce_bias'].flatten())
        Mixed_5c_Branch_1_Conv2d_0b_3x3_bias = tf.constant(pretrained_weights['inception_5b/3x3_bias'].flatten())
        Mixed_5c_Branch_2_Conv2d_0a_1x1_bias = tf.constant(pretrained_weights['inception_5b/5x5_reduce_bias'].flatten())
        Mixed_5c_Branch_2_Conv2d_0b_5x5_bias = tf.constant(pretrained_weights['inception_5b/5x5_bias'].flatten())
        Mixed_5c_Branch_3_Conv2d_0b_1x1_bias = tf.constant(pretrained_weights['inception_5b/pool_proj_bias'].flatten())
        print 'Finished loading'
        variables = {
            'InceptionV1/Conv2d_1a_7x7/weights':tf.get_variable(name='InceptionV1/Conv2d_1a_7x7/weights',initializer=Conv2d_1a_7x7),
            'InceptionV1/Conv2d_2b_1x1/weights': tf.get_variable(name='InceptionV1/Conv2d_2b_1x1/weights',initializer=Conv2d_2b_1x1),
            'InceptionV1/Conv2d_2c_3x3/weights': tf.get_variable(name='InceptionV1/Conv2d_2c_3x3/weights',initializer=Conv2d_2c_3x3),

            'InceptionV1/Conv2d_1a_7x7/bias': tf.get_variable(name='InceptionV1/Conv2d_1a_7x7/bias',initializer=Conv2d_1a_7x7_bias),
            'InceptionV1/Conv2d_2b_1x1/bias': tf.get_variable(name='InceptionV1/Conv2d_2b_1x1/bias',initializer=Conv2d_2b_1x1_bias),
            'InceptionV1/Conv2d_2c_3x3/bias': tf.get_variable(name='InceptionV1/Conv2d_2c_3x3/bias',initializer=Conv2d_2c_3x3_bias),
            #first inception
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_3b_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_3b_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_3b_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_3b_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_3b_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_3b_Branch_3_Conv2d_0b_1x1),
            # first inception bias
            'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_3b_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_3b_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_3b_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_3b_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_3b_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_3b_Branch_3_Conv2d_0b_1x1_bias),
            #second inception
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_3c_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_3c_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_3c_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_3c_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_3c_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_3c_Branch_3_Conv2d_0b_1x1),
            # second inception bias
            'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_3c_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_3c_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_3c_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_3c_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_3c_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_3c_Branch_3_Conv2d_0b_1x1_bias),
            #third inception
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_4b_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_4b_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_4b_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_4b_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_4b_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_4b_Branch_3_Conv2d_0b_1x1),
            # third inception bias
            'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_4b_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_4b_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_4b_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_4b_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_4b_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_4b_Branch_3_Conv2d_0b_1x1_bias),
            # fourth inception
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_4c_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_4c_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_4c_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_4c_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_4c_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_4c_Branch_3_Conv2d_0b_1x1),
            # fourth inception bias
            'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_4c_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_4c_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_4c_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_4c_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_4c_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_4c_Branch_3_Conv2d_0b_1x1_bias),
            # fifth inception
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_4d_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_4d_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_4d_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_4d_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_4d_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_4d_Branch_3_Conv2d_0b_1x1),
            # fifth inception bias
            'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_4d_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_4d_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_4d_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_4d_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_4d_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_4d_Branch_3_Conv2d_0b_1x1_bias),
            # sixth inception
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_4e_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_4e_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_4e_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_4e_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_4e_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_4e_Branch_3_Conv2d_0b_1x1),
            # sixth inception bias
            'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_4e_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_4e_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_4e_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_4e_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_4e_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_4e_Branch_3_Conv2d_0b_1x1_bias),
            # seventh inception
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_4f_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_4f_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_4f_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_4f_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_4f_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_4f_Branch_3_Conv2d_0b_1x1),
            # seventh inception bias
            'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_4f_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_4f_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_4f_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_4f_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_4f_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_4f_Branch_3_Conv2d_0b_1x1_bias),
            # eighth inception
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_5b_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_5b_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_5b_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_5b_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/weights',initializer=Mixed_5b_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_5b_Branch_3_Conv2d_0b_1x1),
            # eighth inception bias
            'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_5b_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_5b_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_5b_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_5b_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/bias',initializer=Mixed_5b_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_5b_Branch_3_Conv2d_0b_1x1_bias),
            # ninth inception
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights',initializer=Mixed_5c_Branch_0_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights',initializer=Mixed_5c_Branch_1_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights',initializer=Mixed_5c_Branch_1_Conv2d_0b_3x3),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights',initializer=Mixed_5c_Branch_2_Conv2d_0a_1x1),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/weights',initializer=Mixed_5c_Branch_2_Conv2d_0b_5x5),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights',initializer=Mixed_5c_Branch_3_Conv2d_0b_1x1),
            # ninth inception bias
            'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/bias',initializer=Mixed_5c_Branch_0_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/bias',initializer=Mixed_5c_Branch_1_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/bias',initializer=Mixed_5c_Branch_1_Conv2d_0b_3x3_bias),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/bias',initializer=Mixed_5c_Branch_2_Conv2d_0a_1x1_bias),
            'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/bias',initializer=Mixed_5c_Branch_2_Conv2d_0b_5x5_bias),
            'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/bias': tf.get_variable(name='InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/bias',initializer=Mixed_5c_Branch_3_Conv2d_0b_1x1_bias),
        }
        fc_variables = {
            'fc_layer_0/weights':tf.get_variable(shape=[1024,64],name='fc_layer_0/weights',initializer=tf.truncated_normal_initializer(stddev=0.1)),
            'fc_layer_0/bias':tf.get_variable(shape=[64],name='fc_layer_0/bias',initializer=tf.truncated_normal_initializer(stddev=0.1)),
            'fc_layer_1/weights': tf.get_variable(shape=[64, 11318], name='fc_layer_1/weights',initializer=tf.truncated_normal_initializer(stddev=0.1)),
            'fc_layer_1/bias':tf.get_variable(shape=[11318], name='fc_layer_1/bias',initializer=tf.truncated_normal_initializer(stddev=0.1))
        }
        return variables,fc_variables
    def CNN_Metric_Model(self,x):
        #layer 1 - conv
        w_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/weights']
        b_1 = self.var_dict['InceptionV1/Conv2d_1a_7x7/bias']
        padding1 = tf.constant([[0,0],[3,3],[3,3],[0,0]])
        input_d = tf.pad(x,paddings=padding1)
        h_conv1 = tf.nn.conv2d(input_d, w_1, strides=[1, 2, 2, 1], padding='VALID') + b_1
        h_conv1 = tf.nn.relu(h_conv1)
        #layer 1 - max pool
        padding_format = tf.constant([[0,0],[0,1],[0,1],[0,0]])
        h_conv1 = tf.pad(h_conv1,paddings=padding_format)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='VALID')
        #h_pool1 = tf.nn.local_response_normalization(h_pool1,depth_radius=5,alpha=0.0001,beta=0.75)
        #layer 2 - conv
        w_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/weights']
        b_2 = self.var_dict['InceptionV1/Conv2d_2b_1x1/bias']
        h_conv2 = tf.nn.conv2d(h_pool1, w_2, strides=[1, 1, 1, 1], padding='VALID') + b_2
        h_conv2 = tf.nn.relu(h_conv2)

        #layer 3 - conv
        padding3 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        h_conv2 = tf.pad(h_conv2, paddings=padding3)
        w_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/weights']
        b_3 = self.var_dict['InceptionV1/Conv2d_2c_3x3/bias']
        h_conv3 = tf.nn.conv2d(h_conv2, w_3, strides=[1, 1, 1, 1], padding='VALID') + b_3
        h_conv3 = tf.nn.relu(h_conv3)
        #h_conv3 = tf.nn.local_response_normalization(h_conv3, depth_radius=5, alpha=0.0001, beta=0.75)

        #layer 3 - max pool
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1], padding='VALID')
        #mixed layer 3b
        #first inception
        #branch 0
        w_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights']
        b_4 = self.var_dict['InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/bias']
        branch1_0 = tf.nn.conv2d(h_pool3, w_4, strides=[1, 1, 1, 1], padding='VALID') + b_4
        branch1_0 = tf.nn.relu(branch1_0)

        #branch 1
        w_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights']
        b_5 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/bias']
        branch1_1 = tf.nn.conv2d(h_pool3, w_5, strides=[1, 1, 1, 1], padding='VALID') + b_5
        branch1_1 = tf.nn.relu(branch1_1)

        padding6 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch1_1 = tf.pad(branch1_1, paddings=padding6)
        w_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights']
        b_6 = self.var_dict['InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/bias']
        branch1_1 = tf.nn.conv2d(branch1_1, w_6, strides=[1, 1, 1, 1], padding='VALID') + b_6
        branch1_1 = tf.nn.relu(branch1_1)

        #branch 2
        w_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights']
        b_7 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/bias']
        branch1_2 = tf.nn.conv2d(h_pool3, w_7, strides=[1, 1, 1, 1], padding='VALID') + b_7
        branch1_2 = tf.nn.relu(branch1_2)

        padding7 = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch1_2 = tf.pad(branch1_2, paddings=padding7)
        w_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/weights']
        b_8 = self.var_dict['InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_5x5/bias']
        branch1_2 = tf.nn.conv2d(branch1_2, w_8, strides=[1, 1, 1, 1], padding='VALID') + b_8
        branch1_2 = tf.nn.relu(branch1_2)

        #branch 3
        padding7 = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch1_3 = tf.pad(h_pool3, paddings=padding7)
        branch1_3 = tf.nn.max_pool(branch1_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights']
        b_9 = self.var_dict['InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/bias']
        branch1_3 = tf.nn.conv2d(branch1_3, w_9, strides=[1, 1, 1, 1], padding='VALID') + b_9
        branch1_3 = tf.nn.relu(branch1_3)

        incpt = tf.concat(
            axis=3, values=[branch1_0, branch1_1, branch1_2, branch1_3])
        #second inception
        #branch 0
        w_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights']
        b_10 = self.var_dict['InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/bias']
        branch2_0 = tf.nn.conv2d(incpt, w_10, strides=[1, 1, 1, 1], padding='VALID') + b_10
        branch2_0 = tf.nn.relu(branch2_0)

        #branch 1
        w_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights']
        b_11 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/bias']
        branch2_1 = tf.nn.conv2d(incpt, w_11, strides=[1, 1, 1, 1], padding='VALID') + b_11
        branch2_1 = tf.nn.relu(branch2_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch2_1 = tf.pad(branch2_1, paddings=padding_format)
        w_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights']
        b_12 = self.var_dict['InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/bias']
        branch2_1 = tf.nn.conv2d(branch2_1, w_12, strides=[1, 1, 1, 1], padding='VALID') + b_12
        branch2_1 = tf.nn.relu(branch2_1)

        #branch 2
        w_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights']
        b_13 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/bias']
        branch2_2 = tf.nn.conv2d(incpt, w_13, strides=[1, 1, 1, 1], padding='VALID') + b_13
        branch2_2 = tf.nn.relu(branch2_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch2_2 = tf.pad(branch2_2, paddings=padding_format)
        w_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/weights']
        b_14 = self.var_dict['InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_5x5/bias']
        branch2_2 = tf.nn.conv2d(branch2_2, w_14, strides=[1, 1, 1, 1], padding='VALID') + b_14
        branch2_2 = tf.nn.relu(branch2_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch2_3 = tf.pad(incpt, paddings=padding_format)
        branch2_3 = tf.nn.max_pool(branch2_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights']
        b_15 = self.var_dict['InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/bias']
        branch2_3 = tf.nn.conv2d(branch2_3, w_15, strides=[1, 1, 1, 1], padding='VALID') + b_15
        branch2_3 = tf.nn.relu(branch2_3)

        incpt = tf.concat(
            axis=3, values=[branch2_0, branch2_1, branch2_2, branch2_3])
        padding_format = tf.constant([[0,0],[0,1],[0,1],[0,0]])
        incpt = tf.pad(incpt,paddings=padding_format)
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='VALID')
        #third inception
        #branch 0
        w_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights']
        b_16 = self.var_dict['InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/bias']
        branch3_0 = tf.nn.conv2d(incpt, w_16, strides=[1, 1, 1, 1], padding='VALID') + b_16
        branch3_0 = tf.nn.relu(branch3_0)

        #branch 1
        w_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights']
        b_17 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/bias']
        branch3_1 = tf.nn.conv2d(incpt, w_17, strides=[1, 1, 1, 1], padding='VALID') + b_17
        branch3_1 = tf.nn.relu(branch3_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch3_1 = tf.pad(branch3_1, paddings=padding_format)
        w_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights']
        b_18 = self.var_dict['InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/bias']
        branch3_1 = tf.nn.conv2d(branch3_1, w_18, strides=[1, 1, 1, 1], padding='VALID') + b_18
        branch3_1 = tf.nn.relu(branch3_1)

        #branch 2
        w_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights']
        b_19 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/bias']
        branch3_2 = tf.nn.conv2d(incpt, w_19, strides=[1, 1, 1, 1], padding='VALID') + b_19
        branch3_2 = tf.nn.relu(branch3_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch3_2 = tf.pad(branch3_2, paddings=padding_format)
        w_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/weights']
        b_20 = self.var_dict['InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_5x5/bias']
        branch3_2 = tf.nn.conv2d(branch3_2, w_20, strides=[1, 1, 1, 1], padding='VALID') + b_20
        branch3_2 = tf.nn.relu(branch3_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch3_3 = tf.pad(incpt, paddings=padding_format)
        branch3_3 = tf.nn.max_pool(branch3_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights']
        b_21 = self.var_dict['InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/bias']
        branch3_3 = tf.nn.conv2d(branch3_3, w_21, strides=[1, 1, 1, 1], padding='VALID') + b_21
        branch3_3 = tf.nn.relu(branch3_3)

        incpt = tf.concat(
            axis=3, values=[branch3_0, branch3_1, branch3_2, branch3_3])
        #fourth inception
        #branch 0
        w_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights']
        b_22 = self.var_dict['InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/bias']
        branch4_0 = tf.nn.conv2d(incpt, w_22, strides=[1, 1, 1, 1], padding='VALID') + b_22
        branch4_0 = tf.nn.relu(branch4_0)

        #branch 1
        w_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights']
        b_23 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/bias']
        branch4_1 = tf.nn.conv2d(incpt, w_23, strides=[1, 1, 1, 1], padding='VALID') + b_23
        branch4_1 = tf.nn.relu(branch4_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch4_1 = tf.pad(branch4_1, paddings=padding_format)
        w_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights']
        b_24 = self.var_dict['InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/bias']
        branch4_1 = tf.nn.conv2d(branch4_1, w_24, strides=[1, 1, 1, 1], padding='VALID') + b_24
        branch4_1 = tf.nn.relu(branch4_1)

        #branch 2
        w_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights']
        b_25 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/bias']
        branch4_2 = tf.nn.conv2d(incpt, w_25, strides=[1, 1, 1, 1], padding='VALID') + b_25
        branch4_2 = tf.nn.relu(branch4_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch4_2 = tf.pad(branch4_2, paddings=padding_format)
        w_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/weights']
        b_26 = self.var_dict['InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_5x5/bias']
        branch4_2 = tf.nn.conv2d(branch4_2, w_26, strides=[1, 1, 1, 1], padding='VALID') + b_26
        branch4_2 = tf.nn.relu(branch4_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch4_3 = tf.pad(incpt, paddings=padding_format)
        branch4_3 = tf.nn.max_pool(branch4_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights']
        b_27 = self.var_dict['InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/bias']
        branch4_3 = tf.nn.conv2d(branch4_3, w_27, strides=[1, 1, 1, 1], padding='VALID') + b_27
        branch4_3 = tf.nn.relu(branch4_3)

        incpt = tf.concat(
            axis=3, values=[branch4_0, branch4_1, branch4_2, branch4_3])
        #fifth inception
        #branch 0
        w_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights']
        b_28 = self.var_dict['InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/bias']
        branch5_0 = tf.nn.conv2d(incpt, w_28, strides=[1, 1, 1, 1], padding='VALID') + b_28
        branch5_0 = tf.nn.relu(branch5_0)

        #branch 1
        w_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights']
        b_29 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/bias']
        branch5_1 = tf.nn.conv2d(incpt, w_29, strides=[1, 1, 1, 1], padding='VALID') + b_29
        branch5_1 = tf.nn.relu(branch5_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch5_1 = tf.pad(branch5_1, paddings=padding_format)
        w_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights']
        b_30 = self.var_dict['InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/bias']
        branch5_1 = tf.nn.conv2d(branch5_1, w_30, strides=[1, 1, 1, 1], padding='VALID') + b_30
        branch5_1 = tf.nn.relu(branch5_1)

        #branch 2
        w_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights']
        b_31 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/bias']
        branch5_2 = tf.nn.conv2d(incpt, w_31, strides=[1, 1, 1, 1], padding='VALID') + b_31
        branch5_2 = tf.nn.relu(branch5_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch5_2 = tf.pad(branch5_2, paddings=padding_format)
        w_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/weights']
        b_32 = self.var_dict['InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_5x5/bias']
        branch5_2 = tf.nn.conv2d(branch5_2, w_32, strides=[1, 1, 1, 1], padding='VALID') + b_32
        branch5_2 = tf.nn.relu(branch5_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch5_3 = tf.pad(incpt, paddings=padding_format)
        branch5_3 = tf.nn.max_pool(branch5_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights']
        b_33 = self.var_dict['InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/bias']
        branch5_3 = tf.nn.conv2d(branch5_3, w_33, strides=[1, 1, 1, 1], padding='VALID') + b_33
        branch5_3 = tf.nn.relu(branch5_3)

        incpt = tf.concat(
            axis=3, values=[branch5_0, branch5_1, branch5_2, branch5_3])
        #sixth inception
        #branch 0
        w_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights']
        b_34 = self.var_dict['InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/bias']
        branch6_0 = tf.nn.conv2d(incpt, w_34, strides=[1, 1, 1, 1], padding='VALID') + b_34
        branch6_0 = tf.nn.relu(branch6_0)

        #branch 1
        w_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights']
        b_35 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/bias']
        branch6_1 = tf.nn.conv2d(incpt, w_35, strides=[1, 1, 1, 1], padding='VALID') + b_35
        branch6_1 = tf.nn.relu(branch6_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch6_1 = tf.pad(branch6_1, paddings=padding_format)
        w_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights']
        b_36 = self.var_dict['InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/bias']
        branch6_1 = tf.nn.conv2d(branch6_1, w_36, strides=[1, 1, 1, 1], padding='VALID') + b_36
        branch6_1 = tf.nn.relu(branch6_1)

        #branch 2
        w_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights']
        b_37 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/bias']
        branch6_2 = tf.nn.conv2d(incpt, w_37, strides=[1, 1, 1, 1], padding='VALID') + b_37
        branch6_2 = tf.nn.relu(branch6_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch6_2 = tf.pad(branch6_2, paddings=padding_format)
        w_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/weights']
        b_38 = self.var_dict['InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_5x5/bias']
        branch6_2 = tf.nn.conv2d(branch6_2, w_38, strides=[1, 1, 1, 1], padding='VALID') + b_38
        branch6_2 = tf.nn.relu(branch6_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch6_3 = tf.pad(incpt, paddings=padding_format)
        branch6_3 = tf.nn.max_pool(branch6_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights']
        b_39 = self.var_dict['InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/bias']
        branch6_3 = tf.nn.conv2d(branch6_3, w_39, strides=[1, 1, 1, 1], padding='VALID') + b_39
        branch6_3 = tf.nn.relu(branch6_3)

        incpt = tf.concat(
            axis=3, values=[branch6_0, branch6_1, branch6_2, branch6_3])
        #seventh inception
        #branch 0
        w_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights']
        b_40 = self.var_dict['InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/bias']
        branch7_0 = tf.nn.conv2d(incpt, w_40, strides=[1, 1, 1, 1], padding='VALID') + b_40
        branch7_0 = tf.nn.relu(branch7_0)

        #branch 1
        w_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights']
        b_41 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/bias']
        branch7_1 = tf.nn.conv2d(incpt, w_41, strides=[1, 1, 1, 1], padding='VALID') + b_41
        branch7_1 = tf.nn.relu(branch7_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch7_1 = tf.pad(branch7_1, paddings=padding_format)
        w_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights']
        b_42 = self.var_dict['InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/bias']
        branch7_1 = tf.nn.conv2d(branch7_1, w_42, strides=[1, 1, 1, 1], padding='VALID') + b_42
        branch7_1 = tf.nn.relu(branch7_1)

        #branch 2
        w_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights']
        b_43 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/bias']
        branch7_2 = tf.nn.conv2d(incpt, w_43, strides=[1, 1, 1, 1], padding='VALID') + b_43
        branch7_2 = tf.nn.relu(branch7_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch7_2 = tf.pad(branch7_2, paddings=padding_format)
        w_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/weights']
        b_44 = self.var_dict['InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_5x5/bias']
        branch7_2 = tf.nn.conv2d(branch7_2, w_44, strides=[1, 1, 1, 1], padding='VALID') + b_44
        branch7_2 = tf.nn.relu(branch7_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch7_3 = tf.pad(incpt, paddings=padding_format)
        branch7_3 = tf.nn.max_pool(branch7_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights']
        b_45 = self.var_dict['InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/bias']
        branch7_3 = tf.nn.conv2d(branch7_3, w_45, strides=[1, 1, 1, 1], padding='VALID') + b_45
        branch7_3 = tf.nn.relu(branch7_3)

        incpt = tf.concat(
            axis=3, values=[branch7_0, branch7_1, branch7_2, branch7_3])
        padding_format = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        incpt = tf.pad(incpt, paddings=padding_format)
        incpt = tf.nn.max_pool(incpt, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='VALID')
        #eighth inception
        #branch 0
        w_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights']
        b_46 = self.var_dict['InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/bias']
        branch8_0 = tf.nn.conv2d(incpt, w_46, strides=[1, 1, 1, 1], padding='VALID') + b_46
        branch8_0 = tf.nn.relu(branch8_0)

        #branch 1
        w_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights']
        b_47 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/bias']
        branch8_1 = tf.nn.conv2d(incpt, w_47, strides=[1, 1, 1, 1], padding='VALID') + b_47
        branch8_1 = tf.nn.relu(branch8_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch8_1 = tf.pad(branch8_1, paddings=padding_format)
        w_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights']
        b_48 = self.var_dict['InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/bias']
        branch8_1 = tf.nn.conv2d(branch8_1, w_48, strides=[1, 1, 1, 1], padding='VALID') + b_48
        branch8_1 = tf.nn.relu(branch8_1)

        #branch 2
        w_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights']
        b_49 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/bias']
        branch8_2 = tf.nn.conv2d(incpt, w_49, strides=[1, 1, 1, 1], padding='VALID') + b_49
        branch8_2 = tf.nn.relu(branch8_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch8_2 = tf.pad(branch8_2, paddings=padding_format)
        w_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/weights']
        b_50 = self.var_dict['InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_5x5/bias']
        branch8_2 = tf.nn.conv2d(branch8_2, w_50, strides=[1, 1, 1, 1], padding='VALID') + b_50
        branch8_2 = tf.nn.relu(branch8_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch8_3 = tf.pad(incpt, paddings=padding_format)
        branch8_3 = tf.nn.max_pool(branch8_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights']
        b_51 = self.var_dict['InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/bias']
        branch8_3 = tf.nn.conv2d(branch8_3, w_51, strides=[1, 1, 1, 1], padding='VALID') + b_51
        branch8_3 = tf.nn.relu(branch8_3)

        incpt = tf.concat(
            axis=3, values=[branch8_0, branch8_1, branch8_2, branch8_3])
        #ninth inception
        #branch 0
        w_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights']
        b_52 = self.var_dict['InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/bias']
        branch9_0 = tf.nn.conv2d(incpt, w_52, strides=[1, 1, 1, 1], padding='VALID') + b_52
        branch9_0 = tf.nn.relu(branch9_0)

        #branch 1
        w_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights']
        b_53 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/bias']
        branch9_1 = tf.nn.conv2d(incpt, w_53, strides=[1, 1, 1, 1], padding='VALID') + b_53
        branch9_1 = tf.nn.relu(branch9_1)

        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch9_1 = tf.pad(branch9_1, paddings=padding_format)
        w_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights']
        b_54 = self.var_dict['InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/bias']
        branch9_1 = tf.nn.conv2d(branch9_1, w_54, strides=[1, 1, 1, 1], padding='VALID') + b_54
        branch9_1 = tf.nn.relu(branch9_1)

        #branch 2
        w_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights']
        b_55 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/bias']
        branch9_2 = tf.nn.conv2d(incpt, w_55, strides=[1, 1, 1, 1], padding='VALID') + b_55
        branch9_2 = tf.nn.relu(branch9_2)

        padding_format = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
        branch9_2 = tf.pad(branch9_2, paddings=padding_format)
        w_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/weights']
        b_56 = self.var_dict['InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_5x5/bias']
        branch9_2 = tf.nn.conv2d(branch9_2, w_56, strides=[1, 1, 1, 1], padding='VALID') + b_56
        branch9_2 = tf.nn.relu(branch9_2)

        #branch 3
        padding_format = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        branch9_3 = tf.pad(incpt, paddings=padding_format)
        branch9_3 = tf.nn.max_pool(branch9_3, ksize=[1, 3, 3, 1],
                                 strides=[1, 1, 1, 1], padding='VALID')
        w_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights']
        b_57 = self.var_dict['InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/bias']
        branch9_3 = tf.nn.conv2d(branch9_3, w_57, strides=[1, 1, 1, 1], padding='VALID') + b_57
        branch9_3 = tf.nn.relu(branch9_3)

        nets = tf.concat(
            axis=3, values=[branch9_0, branch9_1, branch9_2, branch9_3])

        nets = tf.nn.avg_pool(nets, ksize=[1, 7, 7, 1],
                       strides=[1, 1, 1, 1], padding='VALID')

        nets = tf.reshape(nets, [-1, 1024])

        #fc layer
        w_58 = self.fc_var_dict['fc_layer_0/weights']
        b_58 = self.fc_var_dict['fc_layer_0/bias']
        nets = tf.add(tf.matmul(nets,w_58),b_58)
        test = nets

        w_59 = self.fc_var_dict['fc_layer_1/weights']
        b_59 = self.fc_var_dict['fc_layer_1/bias']
        nets = tf.add(tf.matmul(nets, w_59), b_59)
        nets = tf.reshape(nets, [-1, 11318])
        return nets,test

    def triplet_loss(slef,margins, oa, op, on):
        margin_0 = margins[0]
        margin_1 = margins[1]
        margin_2 = margins[2]

        eucd_p = tf.pow(tf.subtract(oa, op), 2)
        eucd_p = tf.reduce_sum(eucd_p, 1)
        eucd_p = tf.sqrt(eucd_p + 1e-6)

        eucd_n1 = tf.pow(tf.subtract(oa, on), 2)
        eucd_n1 = tf.reduce_sum(eucd_n1, 1)
        eucd_n1 = tf.sqrt(eucd_n1 + 1e-6)

        eucd_n2 = tf.pow(tf.subtract(op, on), 2)
        eucd_n2 = tf.reduce_sum(eucd_n2, 1)
        eucd_n2 = tf.sqrt(eucd_n2 + 1e-6)

        random_negative_margin = tf.constant(margin_0)
        rand_neg = tf.pow(tf.maximum(tf.subtract(random_negative_margin,
                                                 tf.minimum(eucd_n1, eucd_n2)), 0), 2)

        positive_margin = tf.constant(margin_1)

        with tf.name_scope('all_loss'):
            # invertable loss for standard patches
            with tf.name_scope('rand_neg'):
                rand_neg = tf.pow(tf.maximum(tf.subtract(random_negative_margin,
                                                         tf.minimum(eucd_n1, eucd_n2)), 0), 2)
            # covariance loss for transformed patches
            with tf.name_scope('pos'):
                pos = tf.pow(tf.maximum(tf.subtract(positive_margin,
                                                    tf.subtract(tf.minimum(eucd_n1, eucd_n2), eucd_p)), 0), 2)
            # total loss
            with tf.name_scope('loss'):
                losses = rand_neg + pos
                loss = tf.reduce_mean(losses)

        # write summary
        # tf.summary.scalar('random_negative_loss', rand_neg)
        # tf.summary.scalar('positive_loss', pos)
        tf.summary.scalar('total_loss', loss)

        return loss, eucd_p, eucd_n1, eucd_n2

    def create_indices(self,labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in xrange(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def generate_triplet(self,_labels, _n_samples):
        # retrieve loaded patches and labels
        labels = _labels
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self.create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        _index_1 = []
        _index_2 = []
        _index_3 = []
        # generate the triplets
        pbar = xrange(_n_samples)

        for x in pbar:
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            _index_1.append(idx_a)
            _index_2.append(idx_p)
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            _index_3.append(idx_n)

        _index_1 = np.array(_index_1)
        _index_2 = np.array(_index_2)
        _index_3 = np.array(_index_3)

        temp_index = np.arange(_index_1.shape[0])

        np.random.shuffle(temp_index)
        _index_1 = _index_1[temp_index]
        _index_2 = _index_2[temp_index]
        _index_3 = _index_3[temp_index]

        return _index_1, _index_2, _index_3

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    cnn_triplet = CNN_Triplet_Metric(sess=sess)


