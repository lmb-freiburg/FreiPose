// Copyright (C) Benjamin Ummenhofer
// Modified by Christian Zimmermann
// SfMTools is a small structure from motion library
#ifndef SFMTOOLS_H_
#define SFMTOOLS_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>


namespace sfm
{

  template <class T>
  struct Type
  {
    typedef Eigen::Matrix<T,3,4> Matrix3x4;
    typedef std::vector<Eigen::Matrix<T,3,4>,Eigen::aligned_allocator<Eigen::Matrix<T,3,4> > > std_vector_Matrix3x4;
    typedef std::vector<Eigen::Matrix<T,3,1>,Eigen::aligned_allocator<Eigen::Matrix<T,3,1> > > std_vector_Vector3;
    typedef std::vector<Eigen::Matrix<T,2,1>,Eigen::aligned_allocator<Eigen::Matrix<T,2,1> > > std_vector_Vector2;
  };

  typedef Eigen::Matrix<double,3,4> Matrix3x4d;
  typedef Type<double>::std_vector_Matrix3x4 std_vector_Matrix3x4d;
  typedef Type<double>::std_vector_Vector3 std_vector_Vector3d;
  typedef Type<double>::std_vector_Vector2 std_vector_Vector2d;
  
  typedef Eigen::Matrix<float,3,4> Matrix3x4f;
  typedef Type<float>::std_vector_Matrix3x4 std_vector_Matrix3x4f;
  typedef Type<float>::std_vector_Vector3 std_vector_Vector3f;
  typedef Type<float>::std_vector_Vector2 std_vector_Vector2f;

  // some templated type defines to make function signatures readable
  #define std_vector_Matrix3x4T std::vector<Eigen::Matrix<T,3,4>,Eigen::aligned_allocator<Eigen::Matrix<T,3,4> > >
  #define std_vector_Vector2T std::vector<Eigen::Matrix<T,2,1>,Eigen::aligned_allocator<Eigen::Matrix<T,2,1> > >
  #define std_vector_Vector3T std::vector<Eigen::Matrix<T,3,1>,Eigen::aligned_allocator<Eigen::Matrix<T,3,1> > >


  

  /*!
   *  Computes whether the point is in front of the camera or not.
   *
   *  \param X  A 3d point in world coordinates
   *  \param R  The rotation for the world to camera view transform
   *  \param t  The translation for the world to camera view transform
   *  \param X  The world point
   *  \return True if the point X is in front of the camera, else False
   */
  template <class T>
  bool isPointInFrontOfCamera( const Eigen::Matrix<T,3,1>& X, 
                               const Eigen::Matrix<T,3,3>& R, 
                               const Eigen::Matrix<T,3,1>& t )
  {
    T z = (R.template block<1,3>(2,0)*X)(0,0) + t(2);
    return z > T(0);
  }


  /*!
   *  Computes whether the point is in front of the camera or not.
   *
   *  \param X  A 3d point in world coordinates
   *  \param P  The camera matrix
   *  \return True if the point X is in front of the camera, else False
   */
  template <class T>
  inline
  bool isPointInFrontOfCamera( const Eigen::Matrix<T,3,1>& X, 
                               const Eigen::Matrix<T,3,4>& P )
  {
    T z = (P.template block<1,3>(2,0) * X)(0,0) + P(2,3);
    if( z <= 0 )
      return false;
    else
      return true;
  }


  /*!
   *  Projects a 3d point to the image plane using the matrix K[R,t]. 
   *
   *  \param K   Camera intrinsics
   *  \param R   The rotation for the world to camera view transform
   *  \param t   The translation for the world to camera view transform
   *  \param X   The world point
   *
   *  \return The projection of the 3d world point
   */
  template <class T>
  inline
  Eigen::Matrix<T,2,1> projectPoint( const Eigen::Matrix<T,3,3>& K, 
                                     const Eigen::Matrix<T,3,3>& R, 
                                     const Eigen::Matrix<T,3,1>& t,
                                     const Eigen::Matrix<T,3,1>& X )
  {
    Eigen::Matrix<T,3,1> tmp = K*(R*X + t);
    Eigen::Matrix<T,2,1> x = tmp.template topRows<2>()/tmp.z();
    return x;
  }


  /*!
   *  Projects a 3d point to the image plane using the matrix P. 
   *
   *  \param P   Camera matrix
   *  \param X   The world point
   *
   *  \return The projection of the 3d world point
   */
  template <class T>
  inline
  Eigen::Matrix<T,2,1> projectPoint( const Eigen::Matrix<T,3,4>& P, 
                                     const Eigen::Matrix<T,3,1>& X )
  {
    Eigen::Matrix<T,4,1> tmp2;
    tmp2 << X[0], X[1], X[2], 1.0;
    
    Eigen::Matrix<T,3,1> tmp = P*tmp2;
    Eigen::Matrix<T,2,1> x = tmp.template topRows<2>()/tmp.z();
    return x;
  }


  /*!
   *  Projects a 3d point to the image plane using the matrix K[R,t] and the 
   *  distortion coefficients kc.
   *  For more details look here: 
   *  http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/parameters.html 
   *
   *  \param K   Camera intrinsics
   *  \param R   The rotation for the world to camera view transform
   *  \param t   The translation for the world to camera view transform
   *  \param kc  Distortion coefficients [k1,k2,p1,p2,k3]
   *  \param X   The world point
   *
   *  \return The projection of the 3d world point
   */
  template <class T>
  inline
  Eigen::Matrix<T,2,1> projectPoint( const Eigen::Matrix<T,3,3>& K, 
                                     const Eigen::Matrix<T,3,3>& R, 
                                     const Eigen::Matrix<T,3,1>& t,
                                     const Eigen::Matrix<T,5,1>& kc, 
                                     const Eigen::Matrix<T,3,1>& X )
  {
    Eigen::Matrix<T,3,1> tmp = R*X + t;
    Eigen::Matrix<T,2,1> x_n = tmp.template topRows<2>()/tmp.z();
    double r2 = x_n.x()*x_n.x() + x_n.y()* x_n.y();
    double r4 = r2*r2;
    double r6 = r2*r4;
    
    Eigen::Matrix<T,2,1> dx;
    dx.x() = 2*kc(2)*x_n.x()*x_n.y() + kc(3)*(r2+2*x_n.x()*x_n.x());
    dx.y() = kc(2)*(r2+2*x_n.y()*x_n.y()) + 2*kc(3)*x_n.x()*x_n.y();

    Eigen::Matrix<T,2,1> x_d;
    x_d = (1+ kc(0)*r2 + kc(1)*r4 + kc(4)*r6)*x_n + dx;
    x_d = K*x_d.homogeneous();
    return x_d;
  }


  /*! 
   *  Computes the 3D points given the measurements x1, x2 and the camera parameters.
   *  This method uses the DLT algorithm to compute a first guess for the 3D point.
   *  Then the nonlinear method is used to refine the point position.
   *
   *  \param X    The resulting vector of triangulated points
   *  \param K1   Camera intrinsics of the first camera
   *  \param R1   The rotation for the world to camera view transform for the 
   *              first camera
   *  \param t1   The translation for the world to camera view transform for 
   *              the first camera
   *
   *  \param K2   Camera intrinsics of the second camera
   *  \param R2   The rotation for the world to camera view transform for the 
   *              second camera
   *  \param t2   The translation for the world to camera view transform for 
   *              the second camera
   *
   *  \param x1   The measured 2D points in the first image
   *  \param x2   The measured 2D points in the second image
   *
   */
  template <class T>
  void triangulate(
      std_vector_Vector3T& X,
      const Eigen::Matrix<T,3,3>& K1,
      const Eigen::Matrix<T,3,3>& R1,
      const Eigen::Matrix<T,3,1>& t1,
      const Eigen::Matrix<T,3,3>& K2,
      const Eigen::Matrix<T,3,3>& R2,
      const Eigen::Matrix<T,3,1>& t2,
      const std_vector_Vector2T& x1,
      const std_vector_Vector2T& x2
      );



  /*! 
   *  Computes the 3D point given the measurements x1, x2 and the camera parameters.
   *  This method uses the DLT algorithm to compute a first guess for the 3D point.
   *  Then the nonlinear method is used to refine the point position.
   *
   *  \param K1   Camera intrinsics of the first camera
   *  \param R1   The rotation for the world to camera view transform for the 
   *              first camera
   *  \param t1   The translation for the world to camera view transform for 
   *              the first camera
   *
   *  \param K2   Camera intrinsics of the second camera
   *  \param R2   The rotation for the world to camera view transform for the 
   *              second camera
   *  \param t2   The translation for the world to camera view transform for 
   *              the second camera
   *
   *  \param x1   The measured 2D point in the first image
   *  \param x2   The measured 2D point in the second image
   *
   *  \return The triangulated 3D point
   */
  template <class T>
  Eigen::Matrix<T,3,1> triangulate( 
      const Eigen::Matrix<T,3,3>& K1,
      const Eigen::Matrix<T,3,3>& R1,
      const Eigen::Matrix<T,3,1>& t1,
      const Eigen::Matrix<T,3,3>& K2,
      const Eigen::Matrix<T,3,3>& R2,
      const Eigen::Matrix<T,3,1>& t2,
      const Eigen::Matrix<T,2,1>& x1,
      const Eigen::Matrix<T,2,1>& x2
      )
  {
    typename Type<T>::std_vector_Vector2 tmp1(1, x1);
    typename Type<T>::std_vector_Vector2 tmp2(1, x2);
    typename Type<T>::std_vector_Vector3 X;
    
    triangulate(X, K1, R1, t1, K2, R2, t2, tmp1, tmp2);
    return X[0];
  }






  /*! 
   *  Computes the 3D points given the measurements x1, x2 and the camera parameters.
   *  This method uses the DLT algorithm to compute a first guess for the 3D point.
   *  Then the nonlinear method is used to refine the point position.
   *
   *  \param P1   Camera matrix corresponding to x1
   *  \param P2   Camera matrix corresponding to x2
   *
   *  \param x1   The measured 2D point in the first image
   *  \param x2   The measured 2D point in the second image
   *
   *  \return The triangulated 3D point
   */
  template <class T>
  Eigen::Matrix<T,3,1> triangulate(
      const Eigen::Matrix<T,3,4>& P1,
      const Eigen::Matrix<T,3,4>& P2,
      const Eigen::Matrix<T,2,1>& x1,
      const Eigen::Matrix<T,2,1>& x2
      )
  {
    typename Type<T>::std_vector_Vector2 tmp1(1, x1);
    typename Type<T>::std_vector_Vector2 tmp2(1, x2);
    typename Type<T>::std_vector_Vector3 X;
    
    triangulate(X, P1, P2, tmp1, tmp2);
    return X[0];
  }



  /*! 
   *  Computes the 3D points given the measurements x1, x2 and the camera parameters.
   *  This method uses the DLT algorithm to compute a first guess for the 3D point.
   *  Then the nonlinear method is used to refine the point position.
   *
   *  \param X    The resulting vector of triangulated points
   *  \param P1   Camera matrix corresponding to x1
   *  \param P2   Camera matrix corresponding to x2
   *
   *  \param x1   The measured 2D points in the first image
   *  \param x2   The measured 2D points in the second image
   *
   */
  template <class T>
  void triangulate(
      std_vector_Vector3T& X,
      const Eigen::Matrix<T,3,4>& P1,
      const Eigen::Matrix<T,3,4>& P2,
      const std_vector_Vector2T& x1,
      const std_vector_Vector2T& x2
      );



  /*!
   *  Computes the 3D point using the DLT. Assumes that the point is not at
   *  infinity and that the homogeneous coordinates of the 2d points
   *  are (x,y,1). 
   *  This function does not perform any prior normalization.
   *  
   *  \param P      Vector of at least two 3x4 camera matrices
   *  \param x      Vector of at least two 2D points
   *  \return Inhomogeneous triangulated 3D point
   */
  template <class T>
  Eigen::Matrix<T,3,1> triangulateLinear(
      const std_vector_Matrix3x4T& P,
      const std_vector_Vector2T& x );


  /*!
   *  Computes the 3D point using the reprojection error. 
   *  The 3D point is optimized using the Levenberg-Marquardt method.
   *  This function does not perform any prior normalization.
   *  
   *  \param X        The 3D point to be computed. X is also used as 
   *                  initial guess.
   *  \param P        Vector of at least two 3x4 camera matrices
   *  \param x        Vector of at least two 2D points
   *  \param weights  Optional weights, to steer for the importance 
   *                  of the camera and their measurements.
   */
  template <class T>
  void triangulateNonlinear(
      Eigen::Matrix<T,3,1>& X,
      const std_vector_Matrix3x4T& P,
      const std_vector_Vector2T& x,
      const std::vector<T> weights = std::vector<T>() );


  /*!
   *  Computes the 3D point using using a RANSAC approach. 
   *  For samples iterations a minimal subset of P and x is chosen and a 3D point
   *  is triangulated with DLT. Each solution is scored wrt. the reprojection error
   *  and the number of inliers is calculated using the threshold provided.
   *  For the final solution the whole inlier set is used and a non linear triangulation
   *  method is used to minimize the error.
   *  
   *  \param X          The 3D point to be computed.
   *  \param inliers    Samples that agree with the solution found.
   *  \param P          Vector of at least two 3x4 camera matrices
   *  \param x          Vector of at least two 2D points
   *  \param samples    Number of iterations.
   *  \param threshold  Reprojection error below which a sample is considered an inlier.
   */
  void triangulateRANSAC(
    Eigen::Vector3d& X,
    Eigen::VectorXi& inliers,
    const std_vector_Matrix3x4d& P,
    const std_vector_Vector2d& x,
    int samples,
    double threshold );


  
  /*!
   *  Computes the corrected measurement points corresponding to x1 and x2.
   *  The corrected points can then be triangulated using the DLT algorithm.
   *  
   *  \param F      The fundamental matrix such that x1*F*x2 evaluates to 0
   *                for perfect points x1,x2
   *  \param x1     2D point in the first image
   *  \param x2     2D point in the second image
   *  \param corrected_x1  The corrected version of point x1
   *  \param corrected_x2  The corrected version of point x2
   */
  template <class T>
  void triangulateOptimalCorrection(
      Eigen::Matrix<T,2,1>& x1_corrected,
      Eigen::Matrix<T,2,1>& x2_corrected,
      const Eigen::Matrix<T,3,3>& F,
      const Eigen::Matrix<T,2,1>& x1,
      const Eigen::Matrix<T,2,1>& x2 );


  /*!
   *  Computes the camera pose for known 2D-3D correspondences.
   *  LO-RANSAC is used to compute the set of inliers.
   *  The final pose is computed iteratively using the Orthogonal Iteration
   *  algorithm.
   *  
   *
   *  \param R       The rotation matrix R in x=K(R*X+t)
   *  \param t       The translation in x=K(R*X+t)
   *  \param inlier  Output for the set of found inliers
   *  \param K       The camera intrinsics
   *  \param xvec    2D measurements
   *  \param Xvec    3D points corresponding to the measurements
   *  \param rounds    Number of models that will be computed
   *  \param threshold This is the allowed reprojection error for inliers 
   */
  template <class T>
  void estimatePoseRANSAC(
      Eigen::Matrix<T,3,3>& R,
      Eigen::Matrix<T,3,1>& t,
      Eigen::VectorXi& inliers,
      const Eigen::Matrix<T,3,3>& K,
      const std_vector_Vector2T& xvec,
      const std_vector_Vector3T& Xvec,
      int rounds = 15,
      T threshold = 3 );



  /*!
   *  Computes the camera pose for known 2D-3D correspondences.
   *  The solution is computed iteratively using the Orthogonal Iteration 
   *  algorithm. 
   *
   *  \param R         Initial guess for the rotation matrix R in x=K(R*X+t)
   *  \param t         Initial guess for the translation in x=K(R*X+t)
   *  \param K         The camera intrinsics
   *  \param x         2D measurements
   *  \param X         3D points corresponding to the measurements
   */
  template <class T>
  void estimatePose( 
      Eigen::Matrix<T,3,3>& R,
      Eigen::Matrix<T,3,1>& t,
      const Eigen::Matrix<T,3,3>& K,
      const std_vector_Vector2T& xvec,
      const std_vector_Vector3T& Xvec );

  template <class T>
  void estimatePose( 
      Eigen::Matrix<T,3,3>& R,
      Eigen::Matrix<T,3,1>& t,
      const Eigen::Matrix<T,3,3>& K,
      const std_vector_Vector2T& xvec,
      const std_vector_Vector3T& Xvec,
      const std::vector<T>& weights );



  
  /*!
   *  Computes the 4x4 transformation matrix to transform the src points to the
   *  dst points.  using LO-RANSAC
   *
   *  \param M       The 4x4 matric describing the scale, rotation and translation
   *  \param inlier  Output for the set of found inliers
   *  \param src     Source point set
   *  \param dst     Destination point set
   *  \param weights     Positive weighting factors
   *  \param scaling If true the transformation contains scaling
   *  \param rounds  Number of models that will be computed
   *  \param threshold This is the allowed euclidean distance for inliers 
   *
   */
  template <class T>
  void computeTransformationRANSAC(
      Eigen::Matrix<T,4,4>& M,
      Eigen::VectorXi& inliers,
      const std_vector_Vector3T& src,
      const std_vector_Vector3T& dst,
      const std::vector<T>& weights,
      bool scaling = true,
      int rounds = 15,
      T threshold = 3 );




  /*!
   *  Translates the centroid of the points to the origin then scales the 
   *  points such that the average distance to the centroid is sqrt(2)
   *
   *  \param x  The points to be normalized
   *  \return The transformation that was used to transform the points
   */
  template <class T>
  Eigen::Matrix<T,3,3> normalizePoints( std_vector_Vector2T& x );



  /*!
   *  Computes the fundamental matrix using the normalized 8 point algorithm.
   *  
   *  \param x1                Points in the first image
   *  \param x2                Points in the second image
   *  \return The fundamental matrix F that fullfills x2'*F*x1 = 0 best in a 
   *          least squares sense for the algebraic error. 
   *          The rank of the returned matrix is 2.
   *          
   */
  Eigen::Matrix3d computeFundamental8Point(
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2 );




  /*!
   *  Computes the fundamental matrix using the Sampson distance.
   *
   *  \param F   The fundamental matrix F with x2'*F*x1 = 0.
   *             F must be an initial estimate of the fundamental matrix.
   *             Will be overwritten with the refined fundamental matrix.
   *             The rank of the returned matrix is 2.
   *  \param x1  Points in the first image
   *  \param x2  Points in the second image
   *
   */
  void computeFundamentalSampson(
      Eigen::Matrix3d& F,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2 );
  


  /*!
   *  Computes the fundamental matrix using LO-RANSAC
   *
   *  \param F       The fundamental matrix F with x2'*F*x1 = 0.
   *                 The rank of the returned matrix is 2.
   *  \param inlier  Output for the set of found inliers
   *  \param x1      Points in the first image
   *  \param x2      Points in the second image
   *  \param rounds  Number of models that will be computed
   *  \param threshold This is the allowed sampson distance 
   *
   */
  void computeFundamentalRANSAC(
      Eigen::Matrix3d& F,
      Eigen::VectorXi& inliers,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2,
      int rounds = 50,
      double threshold = 3 );


  

  /*!
   *  Computes a homography for the point sets with the DLT algorithm
   *
   *  \param x1      Points in the first image
   *  \param x2      Points in the second image
   *  \return The homography H that fullfills x2 = H*x1 best.
   *          
   */
  Eigen::Matrix3d computeHomography(
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2 );


  /*!
   *  Computes the homography matrix using the Sampson distance.
   *
   *  \param H   The homography matrix H with x2=H*x1.
   *             H must be an initial estimate of the homography matrix.
   *             Will be overwritten with the refined homography matrix.
   *  \param x1  Points in the first image
   *  \param x2  Points in the second image
   *
   */
  void computeHomographySampson(
      Eigen::Matrix3d& H,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2 );



  /*!
   *  Computes the homography matrix using LO-RANSAC
   *
   *  \param H       The homography matrix H with x2= H*x1.
   *  \param inlier  Output for the set of found inliers
   *  \param x1      Points in the first image
   *  \param x2      Points in the second image
   *  \param rounds  Number of models that will be computed
   *  \param threshold This is the allowed symmetric transfer error for inliers 
   *
   */
  void computeHomographyRANSAC(
      Eigen::Matrix3d& H,
      Eigen::VectorXi& inliers,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2,
      int rounds = 15,
      double threshold = 3 );



  /*!
   *  Computes the Geometric Robust Information Criterion based on the 
   *  symmetric distances to the epipolar lines
   *
   *  \param F      The fundamental matrix F
   *  \param x1     Points in the first image
   *  \param x2     Points in the second image
   *  \param sigma  Standard deviation of the error
   */
  template <class T>
  T computeGRICFundamental( const Eigen::Matrix<T,3,3>& F,
                            const std_vector_Vector2T& x1,
                            const std_vector_Vector2T& x2,
                            T sigma );
                            
  /*!
   *  Computes the Geometric Robust Information Criterion based on the 
   *  symmetric transfer error of the points
   *
   *  \param H      The homography matrix H
   *  \param x1     Points in the first image
   *  \param x2     Points in the second image
   *  \param sigma  Standard deviation of the error
   */
  template <class T>
  T computeGRICHomography( const Eigen::Matrix<T,3,3>& H,
                           const std_vector_Vector2T& x1,
                           const std_vector_Vector2T& x2,
                           T sigma );

  /*!
   *  Computes the relative pose of the second camera (R,t),
   *  where the camera matrix would be P2=K2*[R,t].
   *
   *  \param R  Output: The orientation to be computed
   *  \param t  Output: The translation to be computed
   *  \param E  The essential matrix
   *  \param x1 Image points in the first image. Should fulfill x2'*E*x1 = 0
   *  \param x2 Image points in the second image. Should fulfill x2'*E*x1 = 0
   */
  void extractExtrinsicsFromEssential (
                               Eigen::Matrix3d& R, 
                               Eigen::Vector3d& t,
                               const Eigen::Matrix3d& E, 
                               const std_vector_Vector2d& x1,
                               const std_vector_Vector2d& x2 );

  /*!
   *  Computes the relative pose of the second camera (R,t),
   *  where the camera matrix would be P2=K2*[R,t].
   *
   *  \param R  Output: The orientation to be computed
   *  \param t  Output: The translation to be computed
   *  \param F  The fundamental matrix
   *  \param K1 Calibration matrix of the first camera
   *  \param K2 Calibration matrix of the second camera
   *  \param x1 Image points in the first image. Should fulfill x2'*F*x1 = 0
   *  \param x2 Image points in the second image. Should fulfill x2'*F*x1 = 0
   */
  void extractExtrinsicsFromFundamental(
                               Eigen::Matrix3d& R, 
                               Eigen::Vector3d& t,
                               const Eigen::Matrix3d& F, 
                               const Eigen::Matrix3d& K1,
                               const Eigen::Matrix3d& K2,
                               const std_vector_Vector2d& x1,
                               const std_vector_Vector2d& x2 );


  /*!
   *  Computes the fundamental matrix from the given camera matrices
   *
   *  \param P1  camera matrix
   *  \param P2  another camera matrix
   *  \return The fundamental matrix F
   */
  template <class T>
  Eigen::Matrix<T,3,3> computeFundamentalFromCameras( 
                               const Eigen::Matrix<T,3,4>& P1,
                               const Eigen::Matrix<T,3,4>& P2 );



} // namespace sfm


// 
// Header only functions
//
#include "umeyama_weighted.hh"


#ifndef SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#undef std_vector_Matrix3x4T
#undef std_vector_Vector2T
#undef std_vector_Vector3T
#endif


#endif /* SFMTOOLS_H_ */
