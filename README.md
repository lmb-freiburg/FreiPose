## FreiPose: A Deep Learning Framework for Precise Animal Motion Capture in 3D Spaces

**IMPORTANT: Code will be added upon paper acceptance**

We present a general purpose framework to estimate 3D locations of predefined landmarks (f.e. skeletal joints of an animal)
 given information from multiple cameras using deep learning methods.
Our frame work is configurable which makes it easy to adapt to the specific experimentation needs and general in terms of
number, type and location of cameras used as well as adaptable over a wide range of subjects of interest (animals, objects, humans, ...). 
Our goal is to bring deep learning based methods into the labs, without the need of expert knowledge about machine learning.

This is the code published in conjunction with our recent paper
    ```
    
	@TechReport{Freipose2020,
	  author    = {Christian Zimmermann, Artur Schneider, Mansour Alyahyay, Thomas Brox and Ilka Diester},
	  title     = {A Deep Learning Framework for Precise Animal Motion Capture in 3D Spaces},
	  year      = {2020},
	  url          = {"https://lmb.informatik.uni-freiburg.de/projects/freipose/"}
	}
     
     

# Overview

This is the main repository containing the code for pose estimation and a more detailed user guide. 
There is also a [Docker container](https://github.com/lmb-freiburg/FreiPose-docker) that provides a runtime 
that is easy to use and a tutorial example to familiarize yourself with the work flow.


# Installation

Using a [Docker container](https://github.com/lmb-freiburg/FreiPose-docker) is the recommended way to use FreiPose.


# User Guide

This is a short introduction on how FreiPose internally works and what scripts are available.

## Concepts

There are two main concepts used in FreiPose:

**Model**

* Is accessed through the config/Model.py class, which loads a `.cfg.json` file, f.e. `config/model_rat.cfg.json`
* A model has some associated data and a defining skeleton
* Inside the model file other `.cfg.json` files are referenced containing the actual data and skeleton information
* Reason for this split is, that the skeleton is usually shared across multiple groups of data instances (i.e. you have different animals in your experiments, but all of them have the same skeleton)
* The model file is passed to almost all scripts of FreiPose
* The skeleton file describes keypoints and related information (limbs, colors, ...), f.e. `config/skel_rat.cfg.json`
* The data file describes labeled chunks of data and trained networks that can be used for inference, f.e. `config/data_rat.cfg.json`


**Param**

* Is accessed through config/Param.py, which loads multiple `.cfg.json` files
* Contains parameters that configure the appearence and behavior of FreiPose
* config/bb_network.cfg.json for the Bounding Box Network (usually no changes needed)
* config/pose_network.cfg.json for the Pose Estimation Network (usually no changes needed)
* config/viewer.cfg.json for the select.py and label.py tools (there are some sizes that you might want to change for a more comfortable viewing on you screen)


## Command Reference

**IMPORTANT: Code will be added upon paper acceptance**

Each of following scripts will provide a more detailed description on its usage when it is called with a **--help** flag.


**eval.py**
Given a Pose Network, defined through the passed model file, it calculates and average euclidean error wrt a labeled dataset.
Which network weights are used is defined by the last entry in `pose_networks`, which is defined through the data entry of the model.

**label.py**
Given model and datset path the labeling allows to add keypoint annotations to the frames and save them into an `anno.json`.
The dataset is written to disk by `select.py`. Also see the [detailed description](Readme_Label.md) of the Labeling Tool.

**predict_bb.py**
Given model and path to a video it will predict bounding boxes around the animal. 
Its sufficient to pass the path to the video of any camera (f.e. run000_cam1.avi). 
Other cameras will be discovered automatically (f.e. run000_cam2.avi, run000_cam3.avi, ...). 

**predict_pose.py**
Given model and path to a video it will predict the animals pose.
Please note that its necessary to FIRST predict the bounding boxes and then pose.
 
Its sufficient to pass the path to the video of any camera (f.e. run000_cam1.avi). 
Other cameras will be discovered automatically (f.e. run000_cam2.avi, run000_cam3.avi, ...). 

**preproc_data.py**
Given a model file it preprocesses labeled data to allow for effective network training.

**select.py**
Given model and prediction file it shows the predictions per frame. 
Allows to select frames for future labeling with label.py according to different sampling criterion.
The selected frames are then written to disk as single frames for the labeling procedure. Also see the [detailed description](Readme_Select.md) of the Selection Tool.

**show_labels.py**
Given a model labeled data of a certain set can be showed.

**show_pred.py**
Given model and prediction file the predictions are visualized and can be saved as a video.

**train_bb.py**
Train bounding box network on the labeled data.

**train_pose.py**
Train Pose network on labeled data.


## System Requirements

Hardware requirements:

- nVIDIA GPU with at least 8GB memory
- CPU that supports AVX instructions
- at least 4GB RAM

Recommended software versions:

    Ubuntu 18.04.3 LTS
    CUDA 10.1
    NVIDIA-Driver 418.87.01
    Docker 19.03.5 
    NVIDIA-container-runtime 1.0.0
     NVIDIA-container-toolkit 1.0.5-1 
    Python 3.6.9, important libraries:
    commentjson  0.8.2
    Cython  0.29.14
    h5py   2.10.0
    joblib    0.14.1
    Keras-Applications   1.0.8 
    Keras-Preprocessing  1.1.0
    matplotlib 3.1.2
    numpy  1.16.4
    opencv-python 4.1.2.30 
    pandas  0.25.3 
    Pillow  7.0.0 
    protobuf 3.11.2
    scipy  1.4.1
    tensorboard  1.13.1
    tensorflow-estimator 1.13.0
    tensorflow-gpu  1.13.1
    tensorpack 0.9.4  

 
