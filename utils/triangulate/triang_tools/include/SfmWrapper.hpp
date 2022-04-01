#ifndef SFM_WRAPPER
#define SFM_WRAPPER

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "Eigen/Eigen"
#include "sfmtools.h"

typedef std::vector<double> vecd;

class SfmWrapper {
private:
public:
    SfmWrapper() {}
    ~SfmWrapper() {}
    
    // Given camera matrices P and observed 2D points x solves DLT to calculate the 3D point
    vecd triangulateLinear(std::vector<vecd> P_list, std::vector<vecd> x_list) {
        if (P_list.size() != x_list.size()) 
            throw std::invalid_argument("There have to be as many camera matrices as observed 2d points.");
        
        // Convert into Eigen matrices
        sfm::std_vector_Vector2d x_list_e;
        for (auto & iter: x_list) {
            if (iter.size() != 2) 
                throw std::invalid_argument("Vector of points x_list must always contain 2 values.");
            x_list_e.emplace_back(iter.data());
        }
        sfm::std_vector_Matrix3x4d P_list_e;
        for (auto & iter: P_list) {
            if (iter.size() != 12) 
                throw std::invalid_argument("Vector of camera matrices P_list must always contain 12 values.");
            P_list_e.emplace_back(iter.data());
        }
        
        // triangulate
        Eigen::Vector3d X = sfm::triangulateLinear((sfm::std_vector_Matrix3x4d) P_list_e,
                                                   (sfm::std_vector_Vector2d) x_list_e);
        
        // output
        vecd result;
        result.resize(3);
        Eigen::Vector3d::Map(&result[0], 3) = X;
        return result;
    }
    
    // Given camera matrices P and observed 2D points x solves DLT to calculate the 3D point and then runs a non linear optimization on top
    vecd triangulateLinearAndNonLinear(std::vector<vecd> P_list, std::vector<vecd> x_list) {
        if (P_list.size() != x_list.size()) 
            throw std::invalid_argument("There have to be as many camera matrices as observed 2d points.");
        
        // Convert into Eigen matrices
        sfm::std_vector_Vector2d x_list_e;
        for (auto & iter: x_list) {
            if (iter.size() != 2) 
                throw std::invalid_argument("Vector of points x_list must always contain 2 values.");
            x_list_e.emplace_back(iter.data());
        }
        sfm::std_vector_Matrix3x4d P_list_e;
        for (auto & iter: P_list) {
            if (iter.size() != 12) 
                throw std::invalid_argument("Vector of camera matrices P_list must always contain 12 values.");
            P_list_e.emplace_back(iter.data());
        }
        
        // triangulate
        Eigen::Vector3d X = sfm::triangulateLinear((sfm::std_vector_Matrix3x4d) P_list_e,
                                                   (sfm::std_vector_Vector2d) x_list_e);
        sfm::triangulateNonlinear(X, P_list_e, x_list_e);
        
        // output
        vecd result;
        result.resize(3);
        Eigen::Vector3d::Map(&result[0], 3) = X;
        return result;
    }
    
    // Given camera matrices P and observed 2D points uses RANSAC to most likely 3d point
    vecd triangulateRansac(std::vector<vecd> P_list, std::vector<vecd> x_list, std::vector<int>& inliers_list,
        float threshold, float probability = 0.9, float outlierRatio=0.25
    ) {
        if (P_list.size() != x_list.size()) 
            throw std::invalid_argument("There have to be as many camera matrices as observed 2d points.");
        
        // Convert into Eigen matrices
        sfm::std_vector_Vector2d x_list_e;
        for (auto & iter: x_list) {
            if (iter.size() != 2) 
                throw std::invalid_argument("Vector of points x_list must always contain 2 values.");
            x_list_e.emplace_back(iter.data());
        }
        sfm::std_vector_Matrix3x4d P_list_e;
        for (auto & iter: P_list) {
            if (iter.size() != 12) 
                throw std::invalid_argument("Vector of camera matrices P_list must always contain 12 values.");
            P_list_e.emplace_back(iter.data());
        }
        
        // Calculate how many iterations we need to achieve a certain probabilty to find the optimal solution
        int samples = static_cast<int>(std::round(
                                        std::log(1 - probability) / std::log( 1 - (1-outlierRatio)*(1-outlierRatio) )
                                  ));
        
        // triangulate
        Eigen::Vector3d X;
        Eigen::VectorXi inliers;
        sfm::triangulateRANSAC(X, inliers,
                               (sfm::std_vector_Matrix3x4d) P_list_e,
                               (sfm::std_vector_Vector2d) x_list_e,
                               samples, threshold);
        
        // output
        vecd result;
        result.resize(3);
        Eigen::Vector3d::Map(&result[0], 3) = X;
        
        inliers_list.resize(inliers.size());
        Eigen::VectorXi::Map(&inliers_list[0], inliers.size()) = inliers;
        return result;
    }
};

#endif