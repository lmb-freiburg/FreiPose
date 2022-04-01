from utils.triangulate.TriangTool import TriangTool, t_triangulation
import numpy as np


def _create_data():

    K1 = np.array([[300.0, 0.0, 150.0],
                  [0.0, 300.0, 150.0],
                  [0.0, 0.0, 1.0]])

    K2 = np.array([[300.0, 0.0, 150.0],
                   [0.0, 300.0, 150.0],
                   [0.0, 0.0, 1.0]])
    K_list = [K1, K2]

    M1 = np.eye(4)
    M2 = np.eye(4)
    M2[:3, 3] = np.array([1.0, 0.0, 0.0])  # translate a bit in x
    M_list = [M1, M2]

    p1 = np.array([[0.0, 0.0]])
    p2 = np.array([[10.0, 0.0]])
    p_list = np.concatenate([p1, p2], 0)

    # import utils.CamLib as cl
    # p3d = cl.triangulate_opencv(K1, np.zeros((5, 1)), M1,
    #                             K2, np.zeros((5, 1)), M2,
    #                             p1, p2)
    #
    # print(p3d)
    p3d = np.array([[-15, -15, 30]])
    return K_list, M_list, p_list, p3d


def test_basic():
    tool = TriangTool()

    # create some points with known 3D
    K_list, M_list, p_list, p3d = _create_data()

    point3d, inlier = tool.triangulate(K_list,
                                       M_list,
                                       np.expand_dims(np.array(p_list), 1),
                                       mode=t_triangulation.RANSAC,
                                       threshold=10.0)

    assert np.allclose(point3d, p3d), '3D Points differ.'
    assert np.sum(inlier) == 2, 'They should all be inliers.'


if __name__ == '__main__':
    """ Run a basic check of the triangulation toolbox. """
    # _create_data()
    test_basic()
