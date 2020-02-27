# FreiPose Prediction Files

Here we give a short introduction on what information is stored in FreiPose prediction files and give examples on how to read them.

Let's assume we have the following prediction file "/host/tutorial_data/pred_run01__00.json"

Then we access the data using python by:

    import json
    import numpy as np
    with open('/host/tutorial_data/pred_run01__00.json', 'r') as fi:
        data = json.load(fi)

    print(type(data)) # is a list
    num_frames = len(data)  # data is a list of predictions, each entry corresponds to one time instance
    print('Prediction file contains %d time steps' % num_frames)
    
    print(type(data[0]))  # is a dict
    fields = data[0].keys()
    print('Available fields: ', fields) # returns ['boxes', 'xyz'] after predict_bb.py and ['boxes', 'xyz', 'kp_xyz', 'kp_score'] after predict_pose.py succeeded

    boxes = np.array(data[0]['boxes'])  # predicted bounding boxes in normalized 2D coordinates
    print(boxes.shape)  # is (7, 4)  i.e. four 2D coordinates of 7 cameras

    xyz = np.array(data[0]['xyz'])  # predicted object center in 3D
    print(xyz.shape)  # is (1, 3)  i.e. one 3D point, center of the 3D prediction volume

    kp_xyz = np.array(data[0]['kp_xyz'])  # predicted pose at time 0 
    print(kp_xyz.shape)  # is (1, 12, 3)  i.e. 12 keypoints in 3D
    
    kp_score = np.array(data[0]['kp_score'])  # predicted scores at time 0, lower means less certain
    print(kp_score.shape)  # is (1, 12)  i.e. scores for 12 keypoints

using MATLAB:

    path2predictionfile='/host/tutorial_data/pred_run01__00.json';
    prediction = jsondecode(fileread(path2predictionfile));
    %prediction is a struct
    
    fields=fieldnames(prediction);    
    num_frames = length(prediction)
    prediction(1) % access the data from 1. Frame
    %    struct with fields:
    %      boxes: [7×4 double]
    %      xyz: [-0.1204 0.0304 0.5414]
    %      kp_xyz: [1×12×3 double]
    %      kp_score: [0.8409 0.7454 0.1927 0.4265 0.3073 0.9465 0.8292 0.7733 0.8842 0.8353 0.6709 0.8220]
    
    size(prediction(1).kp_xyz) % is [1    12     3] i.e. 12 keypoints in 3D
    
    % extract all predicted keypoints into a matrix
    all_predicted_keypoints=zeros([num_frames,size(prediction(1).kp_xyz)]);
    for k_id = 1:length(num_frames)
        all_predicted_keypoints(k_id,:,:,:)=prediction(k_id).kp_xyz;
    end
    all_predicted_keypoints=squeeze(all_predicted_keypoints);
    
