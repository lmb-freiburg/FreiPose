# distutils: language = c++
from libcpp.vector cimport vector


# Makes the cpp class available in here
cdef extern from "SfmWrapper.hpp":
    cdef cppclass SfmWrapper:
        SfmWrapper( ) except +
        vector[double] triangulateLinear(vector[vector[double]] P_list, vector[vector[double]] x_list) except +
        vector[double] triangulateLinearAndNonLinear(vector[vector[double]] P_list, vector[vector[double]] x_list) except +
        vector[double] triangulateRansac(vector[vector[double]] P_list, vector[vector[double]] x_list, vector[int]& inliers, double threshold, double probability, double outlierRatio) except +

cdef class PySfmTool:
    cdef SfmWrapper* c_SfmWrapper      # holds pointer to an C++ instance which we're wrapping

    def __cinit__(self):
        self.c_SfmWrapper = new SfmWrapper()

    def __dealloc__(self):
        del self.c_SfmWrapper

    def triangulateLinear(self, P_list, x_list):
        return self.c_SfmWrapper.triangulateLinear(P_list, x_list)

    def triangulateLinearAndNonLinear(self, P_list, x_list):
        return self.c_SfmWrapper.triangulateLinearAndNonLinear(P_list, x_list)

    def triangulateRansac(self, P_list, x_list, threshold, probability, outlierRatio):
        cdef vector[int] inliers
        x3d = self.c_SfmWrapper.triangulateRansac(P_list, x_list, inliers, threshold, probability, outlierRatio)
        return x3d, inliers
