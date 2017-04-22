#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import os
import numpy as np
import tensorflow as tf
import input_data
import model
import datetime

#%%

N_CLASSES = 15
IMG_W = 100  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 100
BATCH_SIZE = 32
CAPACITY = 100
MAX_STEP = 3000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


#%%
def run_training():
    
    # you need to change the directories to yours.
    #train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
    train_dir = 'D:/python/kewin_code/yanyan/train/'
    #logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
    logs_train_dir = 'D:/python/kewin_code/yanyan/logs/'

    train, train_label = input_data.get_files(train_dir)
    
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)      
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 100 == 0:
                now = datetime.datetime.now()
                print('Time',now.strftime('%Y-%m-%d %H:%M:%S') ,'Step %d, train loss = %.6f, train accuracy = %.6f%%' %(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 500 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    #coord.join(threads)
    sess.close()
#run_training()

    

#%% Evaluate one image
# when training, comment the following codes.


from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   print(n)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   plt.show()

   image = image.resize([100, 100])
   image = np.array(image)
   return image


def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   #train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
   #train_dir = 'D:/python/kewin_code/cats_vs_dogs/data/train/'
   train_dir = 'D:/python/kewin_code/yanyan/train/'
   train, train_label = input_data.get_files(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 15

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 100, 100, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[100, 100, 3])

       # you need to change the directories to yours.
       #logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'
       logs_train_dir = 'D:/python/kewin_code/yanyan/logs/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           print(prediction)
           max_index = np.argmax(prediction)
           print(max_index)

           if max_index==0:
               print('This is a 1 with possibility %.6f' %prediction[:, 0])
           elif max_index==1:
               print('This is a 2 with possibility %.6f' %prediction[:, 1])
           elif max_index==2:
               print('This is a 3 with possibility %.6f' %prediction[:, 2])
           elif max_index==3:
               print('This is a 4 with possibility %.6f' %prediction[:, 3])
           elif max_index==4:
               print('This is a 5 with possibility %.6f' %prediction[:, 4])
           elif max_index==5:
               print('This is a 6 with possibility %.6f' %prediction[:, 5])
           elif max_index==6:
               print('This is a 7 with possibility %.6f' %prediction[:, 6])
           elif max_index==7:
               print('This is a 8 with possibility %.6f' %prediction[:, 7])
           elif max_index==8:
               print('This is a 9 with possibility %.6f' %prediction[:, 8])
           elif max_index==9:
               print('This is a 10 with possibility %.6f' %prediction[:, 9])
           elif max_index==10:
               print('This is a 11 with possibility %.6f' %prediction[:, 10])
           elif max_index==11:
               print('This is a 12 with possibility %.6f' %prediction[:, 11])
           elif max_index==12:
               print('This is a 13 with possibility %.6f' %prediction[:, 12])
           elif max_index==13:
               print('This is a 14 with possibility %.6f' %prediction[:, 13])
           else:
               print('This is a 15 with possibility %.6f' %prediction[:, 14])


evaluate_one_image()

#%%





