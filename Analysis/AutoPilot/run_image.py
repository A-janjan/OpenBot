import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
import os

# Check if on Windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

frame = cv2.imread('driving-datasets/sample2.jpg')
image = cv2.resize(frame, (200, 66)) / 255.0

# Assuming model.y is the output tensor
degrees = sess.run(model.y, feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265
if not windows:
    # Using 'os.system' instead of 'call' as it's more compatible across platforms
    os.system("clear")
print("Predicted steering angle: " + str(degrees) + " degrees")

cv2.imshow('frame', frame)
# Make smooth angle transitions by turning the steering wheel based on the difference of the current angle
# and the predicted angle
smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
    degrees - smoothed_angle)
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow("steering wheel", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
