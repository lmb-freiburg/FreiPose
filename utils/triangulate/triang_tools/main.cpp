#include <iostream>
#include <vector>
#include "Eigen/Eigen"

#include "sfmtools.h"

#include "SfmWrapper.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello, world!" << std::endl;
    
    std::vector< std::vector< double > > x_list;  // All points
    std::vector< std::vector< double > > P_list;  // All camera matrices
    
    // POINT 1
    std::vector<double> vec;
    vec.emplace_back(1.0);
    vec.emplace_back(0.5);    
    x_list.emplace_back(vec);
    
    // POINT 2
    vec.clear();
    vec.emplace_back(1.0);
    vec.emplace_back(1.0);    
    x_list.emplace_back(vec);
    
    // P1
    vec.clear();
    vec.emplace_back(1.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(0.0); 
    
    vec.emplace_back(0.0); 
    vec.emplace_back(1.0);
    vec.emplace_back(0.0); 
    
    vec.emplace_back(0.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(1.0); 
    
    vec.emplace_back(1.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(0.0); 
    P_list.emplace_back(vec);
    
    // P2
    vec.clear();
    vec.emplace_back(1.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(0.0); 
    
    vec.emplace_back(0.0); 
    vec.emplace_back(1.0);
    vec.emplace_back(0.0); 
    
    vec.emplace_back(0.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(1.0); 
    
    vec.emplace_back(0.0);
    vec.emplace_back(0.0); 
    vec.emplace_back(-1.0); 
    P_list.emplace_back(vec);
    
    
    SfmWrapper sfm;
    auto result = sfm.triangulateLinear(P_list, x_list);
    
    std::cout << "result=\n";
    for (auto& iter: result)
        std::cout << iter << "\n";
    
    
    /*void sfm::triangulate(
      std_vector_Vector3T& X,
      const Eigen::Matrix<T,3,4>& P1,
      const Eigen::Matrix<T,3,4>& P2,
      const std_vector_Vector2T& x1,
      const std_vector_Vector2T& x2
      )*/
    return 0;
}
