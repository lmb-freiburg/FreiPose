%% How to project XYZ-coordinates back into xy-image plane
% Given set of 3D XYZ points,e.g predicted keypoints, we would like to
% project and visualize them in one cameras' video
% Image coordinates are defined as x (= going right) and y (= going down),
% while the center is located in the cameras principal point (usually roughly in the image center)
% IMPORTANT: Make sure you have run Matlab_reader.m successfully before

%% PARAMETERS: Change these accordingly
path2videoFiles = './';  % Path to where the video data is
FrameNumber2start=1200; % Frame number you would like to see
distort=0; % if camera has strong distortion, it might be useful to account for distortion during projection, requires ComputerVision Toolbox
cam_id=5; % which camera are we looking at ?
k_id=11; % what keypoint are we looking at ?

%% Read the video file
path2video='path2videoFiles\run001_cam5.avi';
path2mFile='path2videoFiles\M.json';
vid=VideoReader(path2video);
numberofFrames2read=1;

vid.CurrentTime=FrameNumber2start/vid.FrameRate;
vid_frames=zeros(numberofFrames2read,vid.Height,vid.Width,3,'uint8'); % For RGB videos
for i=1:numberofFrames2read
    vid_frames(i,:,:,:)=readFrame(vid);
end

%% Project XYZ back into image plane of the given camera
% Load camera calibration
[M_cam,K_cam,cam_P]=getCamInfo(path2mFile,cam_id);

% Get predictions from one timestep of one keypoint
xyz=squeeze(all_predicted_keypoints(FrameNumber2start,k_id,:)); % see Matlab_reader how to get those

% Transform into coordinate frame of the chosen camera
xyz_h=[xyz',1];
trafo=xyz_h*M_cam';

% Project onto image plane
projection=trafo(1:3)*K_cam';
projection=projection(1:2)/(projection(3)+1e-10);
if distort==1    
    projection= undistortPoints(projection(1:2),cam_P);
end

%% Show the projection
figure(42);clf;
image(squeeze(vid_frames(1,:,:,:)));
hold on;
plot(projection(1),projection(2),'bx')
