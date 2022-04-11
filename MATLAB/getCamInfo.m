function [M_cam, K, cam_P] = getCamInfo(M_path, cam_id)
    % Given a path to a calibration file it returns the intrinsic and extrinsic calibration of the camera
    M = jsondecode(fileread(M_path));
    cam_name = ['cam'+string(cam_id)];
    M_cam = inv(getfield(M,'M',cam_name));
    K = getfield(M,'K',cam_name);
    dist = getfield(M,'dist',cam_name);
    % comment out if to computer vision package available
    cam_P = cameraParameters('IntrinsicMatrix',K','RadialDistortion',dist([1,2,5]),'TangentialDistortion',dist([3,4]));
end


