%%
% This script shows how to access predicted pose data using MATLAB
%%

path2predictionfile='';
path2mFile='';

%load the prediction, prediction is a struct
prediction = jsondecode(fileread(path2predictionfile)); 

fields=fieldnames(prediction);  
num_frames = length(prediction);
prediction(1) 
% access the data from 1. Frame
%    struct with fields:
%      boxes: [7×4 double]
%      xyz: [-0.1204 0.0304 0.5414]
%      kp_xyz: [1×12×3 double]
%      kp_score: [0.8409 0.7454 0.1927 0.4265 0.3073 0.9465 0.8292 0.7733 0.8842 0.8353 0.6709 0.8220]

size(prediction(1).kp_xyz) % is [1    12     3] i.e. 12 keypoints in 3D

% extract all predicted keypoints into a matrix
all_predicted_keypoints=zeros([num_frames,size(prediction(1).kp_xyz)]);
for k_id = 1:length(prediction)
    all_predicted_keypoints(k_id,:,:,:)=prediction(k_id).kp_xyz;
end
all_predicted_keypoints=squeeze(all_predicted_keypoints);


