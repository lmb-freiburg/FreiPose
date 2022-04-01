#define SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#include "sfmtools.h"
#include <ransac.h>
#include <stdexcept>
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace {

  template <class T>
  void solveQuartic( const Eigen::Matrix<T,5,1>& factors, Eigen::Matrix<T,4,1>& realRoots)
  {
    T A = factors(4);
    T B = factors(3);
    T C = factors(2);
    T D = factors(1);
    T E = factors(0);

    T A_pw2 = A*A;
    T B_pw2 = B*B;
    T A_pw3 = A_pw2*A;
    T B_pw3 = B_pw2*B;
    T A_pw4 = A_pw3*A;
    T B_pw4 = B_pw3*B;

    T alpha = -3*B_pw2/(8*A_pw2)+C/A;
    T beta = B_pw3/(8*A_pw3)-B*C/(2*A_pw2)+D/A;
    T gamma = -3*B_pw4/(256*A_pw4)+B_pw2*C/(16*A_pw3)-B*D/(4*A_pw2)+E/A;

    T alpha_pw2 = alpha*alpha;
    T alpha_pw3 = alpha_pw2*alpha;

    std::complex<T> P (-alpha_pw2/12-gamma,0);
    std::complex<T> Q (-alpha_pw3/108+alpha*gamma/3-std::pow(beta,2)/8,0);
    std::complex<T> R = -Q/T(2)+std::sqrt(std::pow(Q,T(2))/T(4)+std::pow(P,T(3))/T(27));

    std::complex<T> U = std::pow(R,T(1.0/3.0));
    std::complex<T> y;

    if (U.real() == 0)
      y = T(-5.0)*alpha/T(6)-std::pow(Q,T(1.0/3.0));
    else
      y = T(-5.0)*alpha/T(6)-P/(T(3)*U)+U;

    std::complex<T> w = std::sqrt(alpha+T(2)*y);

    std::complex<T> temp;

    temp = -B/(T(4)*A) + T(0.5)*(w+std::sqrt(-(T(3)*alpha+T(2)*y+T(2)*beta/w)));
    realRoots(0) = temp.real();
    temp = -B/(T(4)*A) + T(0.5)*(w-std::sqrt(-(T(3)*alpha+T(2)*y+T(2)*beta/w)));
    realRoots(1) = temp.real();
    temp = -B/(T(4)*A) + T(0.5)*(-w+std::sqrt(-(T(3)*alpha+T(2)*y-T(2)*beta/w)));
    realRoots(2) = temp.real();
    temp = -B/(T(4)*A) + T(0.5)*(-w-std::sqrt(-(T(3)*alpha+T(2)*y-T(2)*beta/w)));
    realRoots(3) = temp.real();
  }
  
  /*
   *  Computes the camera pose using three 2D-3D correspondences.
   *  The algorithm returns 4 solutions.
   *
   *  See following paper for details:
   *  
   *  Kneip, Laurent, Davide Scaramuzza, and Roland Siegwart. 
   *  "A novel parametrization of the perspective-three-point problem for a direct 
   *  computation of absolute camera position and orientation." 
   *  Computer Vision and Pattern Recognition (CVPR), 2011 IEEE Conference on. IEEE, 2011.
   *  
   *  \param R1       Output: The rotation matrix R in x=K(R*X+t)
   *  \param t1       Output: The translation in x=K(R*X+t)
   *  \param f1       The normalized direction defined by the 1st image point
   *  \param f2       The normalized direction defined by the 2nd image point
   *  \param f3       The normalized direction defined by the 3rd image point
   *  \param P1       The 3D point corresponding to f1
   *  \param P2       The 3D point corresponding to f2
   *  \param P3       The 3D point corresponding to f3
   */
  template <class T>
  void estimatePoseMinimal(
      Eigen::Matrix<T,3,3>& R1,
      Eigen::Matrix<T,3,1>& t1,
      Eigen::Matrix<T,3,3>& R2,
      Eigen::Matrix<T,3,1>& t2,
      Eigen::Matrix<T,3,3>& R3,
      Eigen::Matrix<T,3,1>& t3,
      Eigen::Matrix<T,3,3>& R4,
      Eigen::Matrix<T,3,1>& t4,
      const Eigen::Matrix<T,3,1>& f1,
      const Eigen::Matrix<T,3,1>& f2,
      const Eigen::Matrix<T,3,1>& f3,
      const Eigen::Matrix<T,3,1>& P1,
      const Eigen::Matrix<T,3,1>& P2,
      const Eigen::Matrix<T,3,1>& P3 )
  {
    // copy inputs to allow permutation
    Eigen::Matrix<T,3,1> f1_(f1);
    Eigen::Matrix<T,3,1> f2_(f2);
    Eigen::Matrix<T,3,1> f3_(f3);
    Eigen::Matrix<T,3,1> P1_(P1);
    Eigen::Matrix<T,3,1> P2_(P2);
    Eigen::Matrix<T,3,1> P3_(P3);

    // intermediate camera frame tau with transformation matrix M
    Eigen::Matrix<T,3,3> M;
    Eigen::Matrix<T,3,1> f3t;
    {
      M.row(0) = f1_.transpose();
      Eigen::Matrix<T,3,1> f1xf2(f1_.cross(f2_));
      M.row(2) = f1xf2.normalized();
      M.row(1) = M.row(2).cross(M.row(0));

      f3t = M*f3_;
      if( f3t.z() > 0 )
      {
        f1_ = f2;
        f2_ = f1;
        P1_ = P2;
        P2_ = P1;

        M.row(0) = f1_.transpose();
        Eigen::Matrix<T,3,1> f1xf2(f1_.cross(f2_));
        M.row(2) = f1xf2.normalized();
        M.row(1) = M.row(2).cross(M.row(0));
        
        f3t = M*f3_;
      }
    }

    // intermediate world frame eta with transformation matrix N
    Eigen::Matrix<T,3,3> N;
    Eigen::Matrix<T,3,1> P3n;
    {
      N.row(0) = (P2_-P1_).normalized();
      Eigen::Matrix<T,3,1> P1P3(P3_-P1_);
      N.row(2) = (N.row(0).transpose().cross(P1P3)).normalized();
      N.row(1) = N.row(2).cross(N.row(0));

      P3n = N*P1P3;
    }

    T d12 = (P2_-P1_).norm();
    T cos_beta = f1_.dot(f2_);
    T b = std::sqrt(1/(1-cos_beta*cos_beta)-1);
    if( cos_beta < 0 )
      b = -b;

    T phi1 = f3t.x()/f3t.z();
    T phi2 = f3t.y()/f3t.z();

    T phi1_pw2 = phi1*phi1;
    T phi2_pw2 = phi2*phi2;

    T p1 = P3n.x();
    T p2 = P3n.y();
    T p1_pw2 = p1*p1;
    T p1_pw3 = p1_pw2*p1;
    T p1_pw4 = p1_pw2*p1_pw2;
    T p2_pw2 = p2*p2;
    T p2_pw3 = p2_pw2*p2;
    T p2_pw4 = p2_pw2*p2_pw2;

    T d12_pw2 = d12*d12;
    T b_pw2 = b*b;

    // polynom a(4)*x^4 + a(3)*x^3 + a(2)*x^2 + a(1)*x + a(0)
    Eigen::Matrix<T,5,1> a;
    a(0) = -2*phi1*phi2*p1*p2_pw2*d12*b+phi2_pw2*p2_pw2*d12_pw2+2*p1_pw3*d12-p1_pw2*d12_pw2+phi2_pw2*p1_pw2*p2_pw2-p1_pw4-2*phi2_pw2*p1*p2_pw2*d12+phi1_pw2*p1_pw2*p2_pw2+phi2_pw2*p2_pw2*d12_pw2*b_pw2;

    a(1) = 2*p1_pw2*p2*d12*b+2*phi1*phi2*p2_pw3*d12-2*phi2_pw2*p2_pw3*d12*b-2*p1*p2*d12_pw2*b;

    a(2) = -phi2_pw2*p1_pw2*p2_pw2-phi2_pw2*p2_pw2*d12_pw2*b_pw2-phi2_pw2*p2_pw2*d12_pw2+phi2_pw2*p2_pw4+phi1_pw2*p2_pw4+2*p1*p2_pw2*d12+2*phi1*phi2*p1*p2_pw2*d12*b-phi1_pw2*p1_pw2*p2_pw2+2*phi2_pw2*p1*p2_pw2*d12-p2_pw2*d12_pw2*b_pw2-2*p1_pw2*p2_pw2;

    a(3) = 2*p2_pw3*d12*b+2*phi2_pw2*p2_pw3*d12*b-2*phi1*phi2*p2_pw3*d12;

    a(4) = -phi2_pw2*p2_pw4-phi1_pw2*p2_pw4-p2_pw4;

    Eigen::Matrix<T,4,1> real_roots;
    solveQuartic(a,real_roots);

    for( int i = 0; i < 4; i++ )
    {
      T cot_alpha = (-phi1*p1/phi2-real_roots(i)*p2+d12*b)/(-phi1*real_roots(i)*p2/phi2+p1-d12);

      T cos_theta = real_roots(i);
      T sin_theta = std::sqrt(1-real_roots(i)*real_roots(i));
      T sin_alpha = std::sqrt(1/(cot_alpha*cot_alpha+1));
      T cos_alpha = std::sqrt(1-sin_alpha*sin_alpha);

      if (cot_alpha < 0)
        cos_alpha = -cos_alpha;

      Eigen::Matrix<T,3,1> C( d12*cos_alpha*(sin_alpha*b+cos_alpha),
                              cos_theta*d12*sin_alpha*(sin_alpha*b+cos_alpha),
                              sin_theta*d12*sin_alpha*(sin_alpha*b+cos_alpha));

      C = P1_ + N.transpose()*C;

      Eigen::Matrix<T,3,3> R;
      R.row(0) = Eigen::Matrix<T,3,1>(	-cos_alpha,		-sin_alpha*cos_theta,	-sin_alpha*sin_theta );
      R.row(1) = Eigen::Matrix<T,3,1>(	sin_alpha,		-cos_alpha*cos_theta,	-cos_alpha*sin_theta );
      R.row(2) = Eigen::Matrix<T,3,1>(	0,				-sin_theta,				cos_theta );

      R = N.transpose()*R.transpose()*M;

      R.transposeInPlace();

      switch( i )
      {
      case 0:
        t1 = -R*C;
        R1 = R;
        break;
      case 1:
        t2 = -R*C;
        R2 = R;
        break;
      case 2:
        t3 = -R*C;
        R3 = R;
        break;
      default:
        t4 = -R*C;
        R4 = R;
        break;
      }
    }
  }



  template <class T>
  struct estimatePoseRANSAC_ModelFn
  {
    typedef Eigen::Matrix<T,3,4,Eigen::DontAlign> ModelType;
    enum Parameters {MINIMUM_SAMPLES=3};

    estimatePoseRANSAC_ModelFn( const Eigen::Matrix<T,3,3>& K,
                                const std_vector_Vector2T& xvec,
                                const std_vector_Vector3T& Xvec )
      :K(K),xvec(xvec),Xvec(Xvec)
    { 
      // compute normalized directions
      const T inv_focalx = 1/K(0,0);
      const T inv_focaly = 1/K(1,1);

      fvec.resize(xvec.size());
      for( int i = 0; i < (int)xvec.size(); ++i )
      {
        const Eigen::Matrix<T,2,1>& x = xvec[i];
        Eigen::Matrix<T,3,1>& f = fvec[i];
        f.x() = inv_focalx*(x.x() - K(0,2));
        f.y() = inv_focaly*(x.y() - K(1,2));
        f.z() = 1;
        f.normalize();
      }
    }


    bool operator()(std::vector<ModelType>& M_vec, const int* idx)
    {
      M_vec.clear();
      ModelType M;
      Eigen::Matrix<T,3,3> R1,R2,R3,R4;
      Eigen::Matrix<T,3,1> t1,t2,t3,t4;

      estimatePoseMinimal(R1,t1,R2,t2,R3,t3,R4,t4,
        fvec[idx[0]],fvec[idx[1]],fvec[idx[2]],Xvec[idx[0]],Xvec[idx[1]],Xvec[idx[2]]);

      M << R1,t1;
      M_vec.push_back(M);
      M << R2,t2;
      M_vec.push_back(M);
      M << R3,t3;
      M_vec.push_back(M);
      M << R4,t4;
      M_vec.push_back(M);
      
      return true;
    }

    bool operator()(ModelType& M, const std::vector<bool>& inliers)
    {
      // build inlier set
      int inliers_count = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
        if( inliers[i] )
          ++inliers_count;

      std_vector_Vector2T xvec_inliers(inliers_count);
      std_vector_Vector3T Xvec_inliers(inliers_count);
      int index = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
      {
        if( inliers[i] )
        {
          xvec_inliers[index] = xvec[i];
          Xvec_inliers[index] = Xvec[i];
          ++index;
        }
      }

      // use the current model as initial guess for the iterative algorithm
      Eigen::Matrix<T,3,3> R = M.template block<3,3>(0,0);
      Eigen::Matrix<T,3,1> t = M.col(3);
      sfm::estimatePose(R,t,K,xvec_inliers,Xvec_inliers);
      M << R,t;

      return true;
    }

    const Eigen::Matrix<T,3,3> K;
    const std_vector_Vector2T& xvec;
    const std_vector_Vector3T& Xvec;
    std_vector_Vector3T fvec;
  };

  template <class T>
  struct estimatePoseRANSAC_DistanceFn
  {
    typedef Eigen::Matrix<T,3,4,Eigen::DontAlign> ModelType;

    estimatePoseRANSAC_DistanceFn( const Eigen::Matrix<T,3,3>& K,
                                   const std_vector_Vector2T& xvec,
                                   const std_vector_Vector3T& Xvec )
      :K(K),xvec(xvec),Xvec(Xvec)
    { }


    T operator()(const Eigen::Matrix<T,3,4,Eigen::DontAlign>& M, int idx)
    {
      Eigen::Matrix<T,3,1> tmp = K*(M.template block<3,3>(0,0)*Xvec[idx]+M.col(3));
      Eigen::Matrix<T,2,1> x = tmp.template topRows<2>()/tmp.z();
      return (x-xvec[idx]).squaredNorm();
    }


    const Eigen::Matrix<T,3,3> K;
    const std_vector_Vector2T& xvec;
    const std_vector_Vector3T& Xvec;
  };


} // namespace



template <class T>
void sfm::estimatePoseRANSAC(
    Eigen::Matrix<T,3,3>& R,
    Eigen::Matrix<T,3,1>& t,
    Eigen::VectorXi& inliers,
    const Eigen::Matrix<T,3,3>& K,
    const std_vector_Vector2T& xvec,
    const std_vector_Vector3T& Xvec,
    int rounds,
    T threshold )
{
  if( Xvec.size() != xvec.size() )
    throw std::runtime_error("estimatePoseRANSAC: 2D-3D point correspondences count mismatch\n");
  // 3 points are sufficient for a minimal solution but 6 general points give a unique solution
  if( Xvec.size() < 6 )
    throw std::runtime_error("estimatePoseRANSAC: Not enough points\n");

  estimatePoseRANSAC_ModelFn<T> modelFn(K,xvec,Xvec);
  estimatePoseRANSAC_DistanceFn<T> distanceFn(K,xvec,Xvec);

  typedef sfm::internal::RANSAC<T,estimatePoseRANSAC_ModelFn<T>,estimatePoseRANSAC_DistanceFn<T> > RANSACType;
    
  RANSACType ransac(modelFn, distanceFn, xvec.size(), threshold*threshold);
  typename RANSACType::ModelType M;
  ransac.run(M,inliers,rounds);

  R = M.template block<3,3>(0,0);
  t = M.col(3);
}
template void sfm::estimatePoseRANSAC( 
      Eigen::Matrix3d& R,
      Eigen::Vector3d& t,
      Eigen::VectorXi& inliers,
      const Eigen::Matrix3d& K,
      const std_vector_Vector2d& xvec,
      const std_vector_Vector3d& Xvec,
      int rounds,
      double threshold );
template void sfm::estimatePoseRANSAC( 
      Eigen::Matrix3f& R,
      Eigen::Vector3f& t,
      Eigen::VectorXi& inliers,
      const Eigen::Matrix3f& K,
      const std_vector_Vector2f& xvec,
      const std_vector_Vector3f& Xvec,
      int rounds,
      float threshold );





template <class T>
void sfm::estimatePose( 
      Eigen::Matrix<T,3,3>& R,
      Eigen::Matrix<T,3,1>& t,
      const Eigen::Matrix<T,3,3>& K,
      const std_vector_Vector2T& xvec,
      const std_vector_Vector3T& Xvec )
{
  if( Xvec.size() != xvec.size() )
    throw std::runtime_error("estimatePose: 2D-3D point correspondences count mismatch\n");
  if( Xvec.size() < 4 )
    throw std::runtime_error("estimatePose: Not enough points\n");
  Eigen::Matrix<T,3,3> K_inv( K.inverse() );
  int n_points = xvec.size();

  // input arrays for the absolute pose problem
  Eigen::Matrix<T,3,Eigen::Dynamic> src(3,n_points);
  Eigen::Matrix<T,3,Eigen::Dynamic> dst(3,n_points);
  for( int i = 0; i < n_points; ++i )
  {
    src.col(i) = Xvec[i];
  }

  // V_vector stores the line of sight projection matrices
  std::vector<Eigen::Matrix<T,3,3> > V_vector;
  V_vector.resize(n_points);

  // 
  // construct the line of sight projection matrices
  //
  Eigen::Matrix<T,3,3> V_sum;
  V_sum.setZero();
  for( int i = 0; i < n_points; ++i )
  {
    Eigen::Matrix<T,3,1> v;
    v << xvec[i], 1; 
    v = K_inv*v;
    v /= v.z();

    Eigen::Matrix<T,3,3>& V = V_vector[i];
    V = v*v.transpose() / v.dot(v);
    V_sum += V;
  }

  // 
  // precompute some matrices
  //
  Eigen::Matrix<T,3,3> M;
  M = (Eigen::Matrix<T,3,3>::Identity() - (V_sum/n_points)).inverse();
  M /= n_points;


  for( int i = 0; i < 50; ++i )
  {
    Eigen::Matrix<T,3,3> R_old = R; 
    Eigen::Matrix<T,3,1> t_old = t;

    // compute t
    Eigen::Matrix<T,3,1> tmp;
    tmp.setZero();
    for( int i = 0; i < n_points; ++i )
    {
      tmp += (V_vector[i] - Eigen::Matrix<T,3,3>::Identity())*R*Xvec[i];
    }
    t = M * tmp;

    // compute q
    for( int i = 0; i < n_points; ++i )
    {
      Eigen::Matrix<T,3,3>& V = V_vector[i];
      dst.col(i) = V*(R*Xvec[i]+t);
    }

    Eigen::Matrix<T,4,4> transformation = Eigen::umeyama(src,dst,false);
    R = transformation.template block<3,3>(0,0);
    t = transformation.template block<3,1>(0,3);
    
    T diff_R = (R_old - R).squaredNorm();
    T diff_t = (t_old - t).squaredNorm();

    //std::cerr << "diff_R = " << diff_R << "   diff_t = " << diff_t << std::endl;
    if( diff_R < 1e-6f && diff_t < 1e-6f )
      break;
  }

}
template void sfm::estimatePose( 
      Eigen::Matrix3d& R,
      Eigen::Vector3d& t,
      const Eigen::Matrix3d& K,
      const std_vector_Vector2d& xvec,
      const std_vector_Vector3d& Xvec );
template void sfm::estimatePose( 
      Eigen::Matrix3f& R,
      Eigen::Vector3f& t,
      const Eigen::Matrix3f& K,
      const std_vector_Vector2f& xvec,
      const std_vector_Vector3f& Xvec );



  
template <class T>
void sfm::estimatePose( 
      Eigen::Matrix<T,3,3>& R,
      Eigen::Matrix<T,3,1>& t,
      const Eigen::Matrix<T,3,3>& K,
      const std_vector_Vector2T& xvec,
      const std_vector_Vector3T& Xvec,
      const std::vector<T>& weights )
{
  if( Xvec.size() != xvec.size() || Xvec.size() != weights.size() )
    throw std::runtime_error("estimatePose: 2D-3D point correspondences count mismatch\n");
  if( Xvec.size() < 4 )
    throw std::runtime_error("estimatePose: Not enough points\n");
  Eigen::Matrix<T,3,3> K_inv( K.inverse() );
  int n_points = xvec.size();

  // input arrays for the absolute pose problem
  Eigen::Matrix<T,3,Eigen::Dynamic> src(3,n_points);
  Eigen::Matrix<T,3,Eigen::Dynamic> dst(3,n_points);
  for( int i = 0; i < n_points; ++i )
  {
    src.col(i) = Xvec[i];
  }


  // V_vector stores the line of sight projection matrices
  std::vector<Eigen::Matrix<T,3,3> > V_vector;
  V_vector.resize(n_points);

  // 
  // construct the line of sight projection matrices
  //
  T weights_sum = 0;
  Eigen::Matrix<T,3,3> V_sum;
  V_sum.setZero();
  for( int i = 0; i < n_points; ++i )
  {
    Eigen::Matrix<T,3,1> v;
    v << xvec[i], 1; 
    v = K_inv*v;
    v /= v.z();

    Eigen::Matrix<T,3,3>& V = V_vector[i];
    V = v*v.transpose() / v.dot(v);
    V_sum += V * weights[i];
    weights_sum += weights[i];
  }
  T weights_sum_inv = 1/weights_sum;

  // 
  // precompute some matrices
  //
  Eigen::Matrix<T,3,3> M;
  M = (Eigen::Matrix<T,3,3>::Identity() - (V_sum*weights_sum_inv)).inverse();
  M *= weights_sum_inv;


  for( int i = 0; i < 50; ++i )
  {
    Eigen::Matrix<T,3,3> R_old = R; 
    Eigen::Matrix<T,3,1> t_old = t;

    // compute t
    Eigen::Matrix<T,3,1> tmp;
    tmp.setZero();
    for( int i = 0; i < n_points; ++i )
    {
      tmp += (V_vector[i] - Eigen::Matrix<T,3,3>::Identity())*R*Xvec[i] * weights[i];
    }
    t = M * tmp;

    // compute q
    for( int i = 0; i < n_points; ++i )
    {
      Eigen::Matrix<T,3,3>& V = V_vector[i];
      dst.col(i) = V*(R*Xvec[i]+t);
    }

    Eigen::Matrix<T,4,4> transformation = umeyama_weighted(src,dst,weights,false);
    R = transformation.template block<3,3>(0,0);
    t = transformation.template block<3,1>(0,3);
    
    T diff_R = (R_old - R).squaredNorm();
    T diff_t = (t_old - t).squaredNorm();

    //std::cerr << "diff_R = " << diff_R << "   diff_t = " << diff_t << std::endl;
    if( diff_R < 1e-6f && diff_t < 1e-6f )
      break;
  }

}
template void sfm::estimatePose( 
      Eigen::Matrix3d& R,
      Eigen::Vector3d& t,
      const Eigen::Matrix3d& K,
      const std_vector_Vector2d& xvec,
      const std_vector_Vector3d& Xvec,
      const std::vector<double>& weights );
template void sfm::estimatePose( 
      Eigen::Matrix3f& R,
      Eigen::Vector3f& t,
      const Eigen::Matrix3f& K,
      const std_vector_Vector2f& xvec,
      const std_vector_Vector3f& Xvec,
      const std::vector<float>& weights );




namespace
{

  template <class T>
  struct estimateTransformationRANSAC_ModelFn
  {
    typedef Eigen::Matrix<T,4,4,Eigen::DontAlign> ModelType;
    enum Parameters {MINIMUM_SAMPLES=3};

    estimateTransformationRANSAC_ModelFn( const std_vector_Vector3T& src,
                                          const std_vector_Vector3T& dst,
                                          const std::vector<T>& weights,
                                          bool scaling )
      :src(src),dst(dst),weights(weights),scaling(scaling)
    { }


    bool operator()(std::vector<ModelType>& M_vec, const int* idx)
    {
      M_vec.resize(1);
      ModelType& M = M_vec[0];
      Eigen::Matrix<T,3,3> src_min, dst_min;
      std::vector<T> weights_min(3);

      for( int i = 0; i < 3; ++i )
      {
        src_min.col(i) = src[idx[i]];
        dst_min.col(i) = dst[idx[i]];
        weights_min[i] = weights[idx[i]];
      }

      M = sfm::umeyama_weighted(src_min,dst_min,weights_min,scaling);
      
      return true;
    }

    bool operator()(ModelType& M, const std::vector<bool>& inliers)
    {
      // build inlier set
      int inliers_count = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
        if( inliers[i] )
          ++inliers_count;

      Eigen::Matrix<T,3,Eigen::Dynamic> src_inl(3,inliers_count);
      Eigen::Matrix<T,3,Eigen::Dynamic> dst_inl(3,inliers_count);
      std::vector<T> weights_inl(inliers_count);
      int index = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
      {
        if( inliers[i] )
        {
          src_inl.col(index) = src[i];
          dst_inl.col(index) = dst[i];
          weights_inl[index] = weights[i];
          ++index;
        }
      }

      M = sfm::umeyama_weighted(src_inl,dst_inl,weights_inl,scaling);

      return true;
    }

    const std_vector_Vector3T& src;
    const std_vector_Vector3T& dst;
    const std::vector<T>& weights;
    const bool scaling;
  };

  template <class T>
  struct estimateTransformationRANSAC_DistanceFn
  {
    typedef Eigen::Matrix<T,4,4,Eigen::DontAlign> ModelType;
    enum Parameters {MINIMUM_SAMPLES=3};

    estimateTransformationRANSAC_DistanceFn( const std_vector_Vector3T& src,
                                             const std_vector_Vector3T& dst,
                                             const std::vector<T>& weights )
      :src(src),dst(dst),weights(weights)
    { }


    T operator()(const ModelType& M, int idx)
    {
      Eigen::Matrix<T,4,1> Msrc = M*src[idx].homogeneous();
      return weights[idx]*(dst[idx]-Msrc.template topRows<3>()).squaredNorm();
    }

    const std_vector_Vector3T& src;
    const std_vector_Vector3T& dst;
    const std::vector<T>& weights;
  };

} // namespace


template <class T>
void sfm::computeTransformationRANSAC(
    Eigen::Matrix<T,4,4>& M,
    Eigen::VectorXi& inliers,
    const std_vector_Vector3T& src,
    const std_vector_Vector3T& dst,
    const std::vector<T>& weights,
    bool scaling,
    int rounds,
    T threshold )
{
  if( src.size() != dst.size() )
    throw std::runtime_error("estimateTransformationRANSAC: src and dst point count mismatch\n");
  if( src.size() < 3 )
    throw std::runtime_error("estimateTransformationRANSAC: Not enough points\n");
  if( weights.size() != src.size() )
    throw std::runtime_error("estimateTransformationRANSAC: weight vector size is wrong\n");


  estimateTransformationRANSAC_ModelFn<T> modelFn(src,dst,weights,scaling);
  estimateTransformationRANSAC_DistanceFn<T> distanceFn(src,dst,weights);

  typedef sfm::internal::RANSAC<T,estimateTransformationRANSAC_ModelFn<T>,estimateTransformationRANSAC_DistanceFn<T> > RANSACType;
    
  RANSACType ransac(modelFn, distanceFn, src.size(), threshold*threshold);
  typename RANSACType::ModelType M_tmp;
  M_tmp.setZero(); // get rid of unitialized value warning
  ransac.run(M_tmp,inliers,rounds);

  M = M_tmp;
}
template 
void sfm::computeTransformationRANSAC(
    Eigen::Matrix<float,4,4>& M,
    Eigen::VectorXi& inliers,
    const std_vector_Vector3f& src,
    const std_vector_Vector3f& dst,
    const std::vector<float>& weights,
    bool scaling,
    int rounds,
    float threshold );
template 
void sfm::computeTransformationRANSAC(
    Eigen::Matrix<double,4,4>& M,
    Eigen::VectorXi& inliers,
    const std_vector_Vector3d& src,
    const std_vector_Vector3d& dst,
    const std::vector<double>& weights,
    bool scaling,
    int rounds,
    double threshold );

