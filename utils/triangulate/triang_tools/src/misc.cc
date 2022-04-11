#define SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#include "sfmtools.h"
#include <stdexcept>
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>



template <class T>
Eigen::Matrix<T,3,3> sfm::normalizePoints( std_vector_Vector2T& xvec )
{
  int N = xvec.size();
  if( N == 0 )
    throw std::runtime_error("normalizePoints: Cannot normalize 0 points\n");
  else if( N == 1 )
  {
    Eigen::Matrix<T,3,3> _T;
    _T << 1, 0, -xvec[0].x(),
         0, 1, -xvec[0].y(),
         0, 0,            1;   
    xvec[0].setZero();
    return _T;
  }

  // compute current centroid
  Eigen::Matrix<T,2,1> cog(0,0);
  for( int i = 0; i < N; ++i )
    cog += xvec[i]; 
  cog /= N;

  T distance = 0;
  // translate points and compute distance
  for( int i = 0; i < N; ++i )
  {
    xvec[i] -= cog;
    distance += xvec[i].norm();
  }
  distance /= N;

  T scale = sqrt(2)/distance;
  for( int i = 0; i < N; ++i )
    xvec[i] *= scale;

  Eigen::Matrix<T,3,3> _T;
  _T << scale,     0, -scale*cog.x(),
           0, scale, -scale*cog.y(),
           0,     0,              1;   
  return _T;
}
template Eigen::Matrix<double,3,3> sfm::normalizePoints(std_vector_Vector2d&);
template Eigen::Matrix<float,3,3> sfm::normalizePoints(std_vector_Vector2f&);

void sfm::extractExtrinsicsFromEssential( 
    Eigen::Matrix3d& R, 
    Eigen::Vector3d& t,
    const Eigen::Matrix3d& E, 
    const std_vector_Vector2d& points1,
    const std_vector_Vector2d& points2 )
{
  Eigen::Matrix3d W;
  W << 0, -1, 0,
       1,  0, 0,
       0,  0, 1;
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

  //std::cout << "\nE=" << E << "\n\n";
  //std::cout << "\nU=" << svd.matrixU() << "\n\n";
  //std::cout << "\nS=" << svd.singularValues() << "\n\n";
  //std::cout << "\nV=" << svd.matrixV() << "\n\n";

  Eigen::Matrix<double,3,4> P1;
  Eigen::Matrix<double,3,4> P2;
  P1.setIdentity();
  std_vector_Matrix3x4d Pvec;
  Pvec.resize(2, P1);
  std_vector_Vector2d xvec;
  xvec.resize(2);

  // 
  // Test all 4 cases for P2
  //
  int case_score[] = {0, 0, 0, 0};
  Eigen::Matrix3d Ra( svd.matrixU() * W * svd.matrixV().transpose() );
  Eigen::Matrix3d Rb( svd.matrixU() * W.transpose() * svd.matrixV().transpose() );
  if( Ra.determinant() < 0 )
  {
    Ra *= -1;
  }
  if( Rb.determinant() < 0 )
  {
    Rb *= -1;
  }
  
  // case 0
  t = svd.matrixU().col(2);
  P2 << Ra,t;
  Pvec[1] = P2;

  // use 100 points for the tests
  int i_step = points1.size()/100;
  if( i_step < 1 )
    i_step = 1;
 
  for( size_t i = 0; i < points1.size(); i+=i_step )
  {
    int c = 0;
    xvec[0] = points1[i]; 
    xvec[1] = points2[i]; 
    Eigen::Vector3d X = triangulateLinear( Pvec, xvec );

    double z = X.z();
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;

    z = (P2.block(2,0,1,3) * X)(0,0) + P2(2,3);
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;
  }


  // case 1
  t = -svd.matrixU().col(2);
  P2 << Ra,t;
  Pvec[1] = P2;

  for( size_t i = 0; i < points1.size(); i+=i_step )
  {
    int c = 1;
    xvec[0] = points1[i]; 
    xvec[1] = points2[i]; 
    Eigen::Vector3d X = triangulateLinear( Pvec, xvec );

    double z = X.z();
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;

    z = (P2.block(2,0,1,3) * X)(0,0) + P2(2,3);
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;
  }


  // case 2
  t = svd.matrixU().col(2);
  P2 << Rb,t;
  Pvec[1] = P2;

  for( size_t i = 0; i < points1.size(); i+=i_step )
  {
    int c = 2;
    xvec[0] = points1[i]; 
    xvec[1] = points2[i]; 
    Eigen::Vector3d X = triangulateLinear( Pvec, xvec );

    double z = X.z();
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;

    z = (P2.block(2,0,1,3) * X)(0,0) + P2(2,3);
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;
  }


  // case 3
  t = -svd.matrixU().col(2);
  P2 << Rb,t;
  Pvec[1] = P2;

  for( size_t i = 0; i < points1.size(); i+=i_step )
  {
    int c = 3;
    xvec[0] = points1[i]; 
    xvec[1] = points2[i]; 
    Eigen::Vector3d X = triangulateLinear( Pvec, xvec );

    double z = X.z();
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;

    z = (P2.block(2,0,1,3) * X)(0,0) + P2(2,3);
    if( z < 0 )
      case_score[c]--;
    else
      case_score[c]++;
  }

  //std::cerr << "<<" << case_score[0] << ", " << case_score[1] << ", " 
                    //<< case_score[2] << ", " << case_score[3] << ">>\n";

  // pick best score
  int best_case = 0;
  int best_score = case_score[0];
  for( int i = 1; i < 4; ++i )
    if( case_score[i] > best_score )
    {
      best_case = i;
      best_score = case_score[i];
    }

  switch( best_case )
  {
  case 0:
    t = svd.matrixU().col(2);
    R = Ra;
    break;
  case 1:
    t = -svd.matrixU().col(2);
    R = Ra;
    break;
  case 2:
    t = svd.matrixU().col(2);
    R = Rb;
    break;
  case 3:
    t = -svd.matrixU().col(2);
    R = Rb;
    break;
  }

  //std::cerr << "Ra=" << Ra << std::endl<< std::endl;
  //std::cerr << "Rb=" << Rb << std::endl<< std::endl;
}



void sfm::extractExtrinsicsFromFundamental( 
    Eigen::Matrix3d& R, 
    Eigen::Vector3d& t,
    const Eigen::Matrix3d& F, 
    const Eigen::Matrix3d& K1,
    const Eigen::Matrix3d& K2,
    const std_vector_Vector2d& x1,
    const std_vector_Vector2d& x2 )
{
  Eigen::Matrix3d E;
  E = K2.transpose() * F * K1; 

  std_vector_Vector2d x1_normed(x1);
  std_vector_Vector2d x2_normed(x2);

  Eigen::Matrix3d K1_inv( K1.inverse() );
  Eigen::Matrix3d K2_inv( K2.inverse() );

  for( size_t i = 0; i < (int)x1_normed.size(); ++i )
  {
    Eigen::Vector3d tmp;
    tmp << x1_normed[i], 1;
    tmp = K1_inv * tmp;
    tmp.topRows(2) /= tmp.z();
    x1_normed[i] = tmp.topRows(2);

    tmp << x2_normed[i], 1;
    tmp = K2_inv * tmp;
    tmp.topRows(2) /= tmp.z();
    x2_normed[i] = tmp.topRows(2);
  }

  extractExtrinsicsFromEssential( R, t, E, x1_normed, x2_normed );
}



template <class T>
Eigen::Matrix<T,3,3> sfm::computeFundamentalFromCameras(
                                      const Eigen::Matrix<T,3,4>& P1, 
                                      const Eigen::Matrix<T,3,4>& P2)
{
  Eigen::Matrix<T,3,3> F;
  Eigen::Matrix<T,2,4> X1, X2, X3;
  X1 = P1.bottomRows(2);
  X2.topRows(1) = P1.bottomRows(1);
  X2.bottomRows(1) = P1.topRows(1);
  X3 = P1.topRows(2);

  Eigen::Matrix<T,2,4> Y1, Y2, Y3;
  Y1 = P2.bottomRows(2);
  Y2.topRows(1) = P2.bottomRows(1);
  Y2.bottomRows(1) = P2.topRows(1);
  Y3 = P2.topRows(2);

  Eigen::Matrix<T,4,4> tmp;
  tmp << X1, Y1;
  F(0,0) = tmp.determinant();
  tmp << X2, Y1;
  F(0,1) = tmp.determinant();
  tmp << X3, Y1;
  F(0,2) = tmp.determinant();
  
  tmp << X1, Y2;
  F(1,0) = tmp.determinant();
  tmp << X2, Y2;
  F(1,1) = tmp.determinant();
  tmp << X3, Y2;
  F(1,2) = tmp.determinant();
  
  tmp << X1, Y3;
  F(2,0) = tmp.determinant();
  tmp << X2, Y3;
  F(2,1) = tmp.determinant();
  tmp << X3, Y3;
  F(2,2) = tmp.determinant();
  
  return F;
}
template Eigen::Matrix<double,3,3> sfm::computeFundamentalFromCameras(const Matrix3x4d&, const Matrix3x4d&);
template Eigen::Matrix<float,3,3> sfm::computeFundamentalFromCameras(const Matrix3x4f&, const Matrix3x4f&);


