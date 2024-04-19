# Autopilot-TensorFlow
A TensorFlow implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes. ( thanks to [Sully Chen](https://github.com/SullyChen/) )

# IMPORTANT
Absolutely, under NO circumstance, should one ever pilot a car using computer vision software trained with this code (or any home made software for that matter). It is extremely dangerous to use your own self-driving software in a car, even if you think you know what you're doing, not to mention it is quite illegal in most places and any accidents will land you in huge lawsuits.

This code is purely for research and statistics, absolutley NOT for application or testing of any sort.

# How to Use


Use `python run_image.py` to run the model on a image.

Use `python run.py` to run the model on a live webcam feed.

Use `python run_video.py` to run the model on a video.

Use `python run_dataset.py` to run the model on the dataset.

Use `python train.py` to train the model.

Note : Download the [dataset](https://github.com/SullyChen/driving-datasets) and extract into the driving-datasets folder and then train the model or you can take pictures and then resize them to 455x256 pixels.


To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.

# See activations

You can see activation of pilotnet layers with executing `python show_activation.py`.

example:

![activation_show](https://github.com/A-janjan/OpenBot/assets/62621376/f603c8a1-eaa0-469a-bc4c-3e2ddfb79905)
