import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import model_nvidia
import sys

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_dir', './driving-datasets',
    """Directory that stores input recorded front view images and steering wheel angles.""")

tf.compat.v1.app.flags.DEFINE_string(
    'model_file', './model_files/model.ckpt',
    """Path to the model parameter file.""")

def _generate_feature_image(feature_map, shape):
    dim = feature_map.shape[2]
    row_step = feature_map.shape[0]
    col_step = feature_map.shape[1]

    feature_image = np.zeros([row_step*shape[0], col_step*shape[1]])
    min_val = np.min(feature_map)
    max_val = np.max(feature_map)
    minmax = np.fabs(min_val - max_val)
    cnt = 0
    for row in range(shape[0]):
        row_idx = row_step * row
        row_idx_nxt = row_step * (row + 1)
        for col in range(shape[1]):
            col_idx = col_step * col
            col_idx_nxt = col_step * (col + 1)
            feature_image[row_idx:row_idx_nxt, col_idx:col_idx_nxt] = (feature_map[:, :, cnt] - min_val) * 1.0/minmax
            cnt += 1
    return feature_image

def show_activation(argv=None):
    full_image = plt.imread(FLAGS.dataset_dir + "/sample_4" + ".jpg")
    image = tf.image.resize(full_image, [66, 200]) / 255.0
    fig = plt.figure('Visualization of Internal CNN State')
    plt.subplot(211)
    plt.title('Normalized input planes 3@66x200 to the CNN')
    
    with tf.compat.v1.Session() as sess:
        image_np = sess.run(image)
    
    plt.imshow(image_np)

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.model_file)
        print("Load session successfully")

        conv1act, conv2act, conv3act, conv4act, conv5act = sess.run(
            [model_nvidia.h_conv1, model_nvidia.h_conv2, model_nvidia.h_conv3, model_nvidia.h_conv4, model_nvidia.h_conv5],
            feed_dict={
                model_nvidia.x: [image_np]
            }
        )

    
        conv1img = _generate_feature_image(conv1act[0], [6, int(conv1act.shape[3]/6)])
        conv2img = _generate_feature_image(conv2act[0], [6, int(conv1act.shape[3]/6)])
        

        averageC5 = np.mean(conv5act, axis=3).squeeze(axis=0)
        averageC4 = np.mean(conv4act, axis=3).squeeze(axis=0)
        averageC3 = np.mean(conv3act, axis=3).squeeze(axis=0)
        averageC2 = np.mean(conv2act, axis=3).squeeze(axis=0)
        averageC1 = np.mean(conv1act, axis=3).squeeze(axis=0)


        # expand dims
        averageC1 = np.expand_dims(averageC1, axis=-1)
        averageC2 = np.expand_dims(averageC2, axis=-1)
        averageC3 = np.expand_dims(averageC3, axis=-1)
        averageC4 = np.expand_dims(averageC4, axis=-1)
        averageC5 = np.expand_dims(averageC5, axis=-1)
        

        print("############### shape:", averageC4.shape)
        # Resize tensors using tf.image.resize
        
        averageC5up = tf.image.resize(averageC5, [averageC4.shape[0], averageC4.shape[1]])   # averageC5up shape(3,20,1)  
        multC45 = tf.multiply(averageC5up, averageC4)
        multC45up = tf.image.resize(multC45, [averageC3.shape[0], averageC3.shape[1]])
        multC34 = tf.multiply(multC45up, averageC3)
        multC34up = tf.image.resize(multC34, [averageC2.shape[0], averageC2.shape[1]])
        multC23 = tf.multiply(multC34up, averageC2)
        multC23up = tf.image.resize(multC23, [averageC1.shape[0], averageC1.shape[1]])
        multC12 = tf.multiply(multC23up, averageC1)
        multC12up = tf.image.resize(multC12, [image.shape[0], image.shape[1]])

        with tf.compat.v1.Session() as sess:
            multC12up = multC12up.eval()

        salient_mask = (multC12up - np.min(multC12up))/(np.max(multC12up) - np.min(multC12up))
        plt.subplot(223)
        plt.title('Activation of the first layer feature maps')
        plt.imshow(conv1img, cmap='gray')

        plt.subplot(224)
        plt.title('Activation of the second layer feature maps')
        plt.imshow(conv2img, cmap='gray')

    plt.show()

if __name__ == '__main__':
    tf.compat.v1.app.run(main=show_activation, argv=[sys.argv[0]])
