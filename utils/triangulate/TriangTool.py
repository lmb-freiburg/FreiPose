# Wrapper around the wrapper that adapts inputs
import numpy as np
import cv2

from .PyWrapper import *


class t_triangulation:
    LINEAR = 1,
    NONLINEAR = 2,
    RANSAC = 3



def undistort_points(points, K, dist):
    """ Given observed points this function returns where the point would lie when there would be no lens distortion."""
    points = np.reshape(points, [-1, 2]).astype(np.float32)

    # Runs an iterative algorithm to invert what distort_points(..) does
    points_dist = cv2.undistortPoints(np.expand_dims(points, 0),
                                           K, np.squeeze(dist),
                                           P=K)
    return np.squeeze(points_dist, 0)


class TriangTool(object):
    def __init__(self):
        self.pySfmWrapper = PySfmTool()

    @staticmethod
    def _calc_normalization_2d(points, scale=np.sqrt(2)):
        """ Normalizes the points to have zero mean and given scale.
            Returns the normalized points and the matrix that undoes inversion.
        """
        assert len(points.shape) == 2, "Points have to be 2D"
        assert points.shape[1] == 2, "Points have to be 2D"
        m = np.mean(points, 0)
        rms = np.mean(np.sqrt(np.sum(np.square(points - m), 1)))
        s = scale / rms
        points_scaled = s * (points - m)

        # matrix that normalizes
        T = np.array([[s, 0.0, -s*m[0]],
                      [0.0, s, -s*m[1]],
                      [0.0, 0.0, 1.0]])

        # matrix that denormalizes (the inverse)
        T_inv = np.array([[1.0/s, 0.0, m[0]],
                          [0.0, 1.0/s, m[1]],
                          [0.0, 0.0, 1.0]])
        return points_scaled, T, T_inv

    def _triangulateLinear(self, P_list, x_list):
        """ Triangulate 2D observations in x using the projection matrices P of at least two cameras.
            P_list: list of np.arrays of shape 3x4
            x_list: list of np.arrays of shape 2
        """
        assert len(P_list) == len(x_list), "There have to be as many observed points as corresponding cameras."
        for P, x in zip(P_list, x_list):
            assert len(P.shape) == 2, "Shape mismatch."
            assert (P.shape[0] == 3) and (P.shape[1] == 4), "Shape mismatch."
            assert len(x.shape) == 1, "Shape mismatch."
            assert x.shape[0] == 2, "Shape mismatch."

        # flatten arrays into vectors
        x_list = [x.flatten().tolist() for x in x_list]
        P_list = [x.flatten('F').tolist() for x in P_list]

        # run
        result = self.pySfmWrapper.triangulateLinear(P_list, x_list)

        return np.reshape(result, [1, 3])

    def _triangulateLinearAndNonLinear(self, P_list, x_list):
        """ Triangulate 2D observations in x using the projection matrices P of at least two cameras.
            P_list: list of np.arrays of shape 3x4
            x_list: list of np.arrays of shape 2
        """
        assert len(P_list) == len(x_list), "There have to be as many observed points as corresponding cameras."
        for P, x in zip(P_list, x_list):
            assert len(P.shape) == 2, "Shape mismatch."
            assert (P.shape[0] == 3) and (P.shape[1] == 4), "Shape mismatch."
            assert len(x.shape) == 1, "Shape mismatch."
            assert x.shape[0] == 2, "Shape mismatch."

        # flatten arrays into vectors
        x_list = [x.flatten().tolist() for x in x_list]
        P_list = [x.flatten('F').tolist() for x in P_list]

        # run
        result = self.pySfmWrapper.triangulateLinearAndNonLinear(P_list, x_list)

        return np.reshape(result, [1, 3])

    def _triangulateRansac(self, P_list, x_list, threshold, probability, outlierRatio):
        """ Triangulate 2D observations in x using the projection matrices P of at least two cameras.
            P_list: list of np.arrays of shape 3x4
            x_list: list of np.arrays of shape 2
        """
        assert len(P_list) == len(x_list), "There have to be as many observed points as corresponding cameras."
        for P, x in zip(P_list, x_list):
            assert len(P.shape) == 2, "Shape mismatch."
            assert (P.shape[0] == 3) and (P.shape[1] == 4), "Shape mismatch."
            assert len(x.shape) == 1, "Shape mismatch."
            assert x.shape[0] == 2, "Shape mismatch."

        # flatten arrays into vectors
        x_list = [x.flatten().tolist() for x in x_list]
        P_list = [x.flatten('F').tolist() for x in P_list]

        # run
        x, inliers = self.pySfmWrapper.triangulateRansac(P_list, x_list, threshold, probability, outlierRatio)

        return np.reshape(x, [1, 3]), inliers

    def triangulate(self, K_list, M_list, x_list, dist_list=None, mode=None, use_data_norm=False,
                    threshold=5.0, probability=0.99, outlierRatio=0.75):
        """ Triangulates 2D observations in x using the projection matrices P of at least two cameras into a single 3D point.
            K_list: list of np.arrays of shape 3x3, camera intrinsic
            dist_list: list of np.arrays of shape 1x5, camera distortion
            M_list: list of np.arrays of shape 4x4, camera extrinsic (mapping from world -> cam)
            x_list: list of np.arrays of shape Nx2, List of N observed points of one camera (undistorted)
            mode: type of triangulation, one of t_triangulation

            RANSAC ONLY PARAMETERS
            threshold: float, reprojection error below which the sample is considered an inlier.
            probability: float, probability that the optimal solution is found. Higher probability increases runtime.
            outlierRatio: float, how many of the samples are outliers.

            use_data_norm: boolean, Normalize 2D points for triangulation (in my experience this makes the result worse)
        """
        if dist_list is not None:
            assert len(K_list) == len(dist_list), "There have to be equally many camera parameters."
        assert len(K_list) == len(M_list), "There have to be equally many camera parameters."
        assert len(K_list) == len(x_list), "There have to be equally many cameras then made observation."
        assert (probability < 1.0) and (probability > 0.0), "probability range."
        assert (outlierRatio < 1.0) and (outlierRatio >= 0.0), "outlierRatio range."

        # normalize points to increase numerical stability
        T_list = list()
        x_list_norm = list()
        for x_cam in x_list:
            if use_data_norm:
                x_cam, T, _ = self._calc_normalization_2d(x_cam)
            else:
                T = np.eye(3)
            x_list_norm.append(x_cam)
            T_list.append(T)
        x_list = x_list_norm

        # reshape points so unique points is first dimension
        x_list = np.stack(x_list)  # this is num_cams x num_pts x 2
        x_list = x_list.transpose([1, 0, 2])

        # assemble projective matrices
        P_list = list()
        for T, K, M in zip(T_list, K_list, M_list):
            P_list.append( np.matmul(T, np.matmul(K, M[:3, :])) )

        # triangulate each unique point seperately
        x3d_list = list()
        inliers_list = list()
        for x_obs_list in x_list:
            if dist_list is not None:
                # undistort all observations
                x_obs_list = [undistort_points(x, K_list[i], dist_list[i]).squeeze() for i, x in enumerate(x_obs_list)]

            # triangulate observations in one point
            if (mode is None) or (mode == t_triangulation.LINEAR):
                x3d_list.append(self._triangulateLinear(P_list, x_obs_list))

            elif mode == t_triangulation.NONLINEAR:
                x3d_list.append(self._triangulateLinearAndNonLinear(P_list, x_obs_list))

            elif mode == t_triangulation.RANSAC:
                x, inliers = self._triangulateRansac(P_list, x_obs_list, threshold, probability, outlierRatio)
                x3d_list.append(x)
                inliers_list.append(inliers)
                # print('inliers', inliers)
                # print('Inlier ratio py', np.mean(inliers))

            else:
                assert 0, "Invalid triangulation mode."

        if mode == t_triangulation.RANSAC:
            return np.concatenate(x3d_list, 0), np.array(inliers_list)

        inliers = np.ones_like(np.array(x_obs_list)[:, 0])
        return np.concatenate(x3d_list, 0), inliers
