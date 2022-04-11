#define SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#include "sfmtools.h"
#include <stdexcept>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>


namespace {

  template <class T>
  inline Eigen::Matrix<T,2,1> algebraicDistance( const Eigen::Matrix<T,3,3>& H, 
                                                 const Eigen::Matrix<T,2,1>& x1,
                                                 const Eigen::Matrix<T,2,1>& x2 )
  {
    Eigen::Matrix<T,2,1> d;
    d.x() = -1*(x1.dot(H.template block<1,2>(1,0)) + H(1,2))
            +x2.y()*(x1.dot(H.template block<1,2>(2,0)) + H(2,2));
    d.y() = 1*(x1.dot(H.template block<1,2>(0,0)) + H(0,2))
            -x2.x()*(x1.dot(H.template block<1,2>(2,0)) + H(2,2));
    return d;
  }


  // taken from the matlab code vgg_H_sampson_distance_sqr.m from the 
  // Multiple View Geometry book
  template <class T>
  inline T sampsonDistanceSqr( const Eigen::Matrix<T,3,3>& H, 
                               const Eigen::Matrix<T,2,1>& x1,
                               const Eigen::Matrix<T,2,1>& x2 )
  {
    Eigen::Matrix<T,2,1> alg = algebraicDistance(H,x1,x2);
    

    Eigen::Matrix<T,3,1> G1, G2;
    G1.x() = H(0,0) - x2.x() * H(2,0);
    G1.y() = H(0,1) - x2.x() * H(2,1);
    G1.z() =-x1.x() * H(2,0) - x1.y() * H(2,1) - H(2,2);

    G2.x() = H(1,0) - x2.y() * H(2,0);
    G2.y() = H(1,1) - x2.y() * H(2,1);
    G2.z() =-x1.x() * H(2,0) - x1.y() * H(2,1) - H(2,2);

    T magG1 = G1.norm();
    T magG2 = G2.norm();
    T magG1G2 = G1.x()*G2.x() + G1.y()*G2.y();

    T alpha = std::acos(magG1G2 /(magG1*magG2));


    Eigen::Matrix<T,2,1> d;
    d.x() = alg.x()/magG1;
    d.y() = alg.y()/magG2;

    T result = (d.dot(d) - 2.0*d.x()*d.y()*std::cos(alpha))/std::sin(alpha); 
    return result;
  }


  template <class T>
  inline T symmetricTransferError( const Eigen::Matrix<T,3,3>& H,
                                   const Eigen::Matrix<T,3,3>& H_inv,
                                   const Eigen::Matrix<T,2,1>& x1,
                                   const Eigen::Matrix<T,2,1>& x2 )
  {
    Eigen::Matrix<T,3,1> tmp;
    tmp << x1.x(), x1.y(), T(1);
    tmp = H*tmp;
    tmp.template topRows<2>() /= tmp.z();

    T err = (tmp.template topRows<2>() - x2).squaredNorm();

    tmp << x2.x(), x2.y(), T(1);
    tmp = H_inv*tmp;
    tmp.template topRows<2>() /= tmp.z();

    err += (tmp.template topRows<2>() - x1).squaredNorm();
    return err;
  }



  // Base class for all Functors
  template <class _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
  struct Functor
  {
    Functor( int _inputs, int _values )
      :nInputs(_inputs), nValues(_values)
    { }

    typedef _Scalar Scalar;
    enum {
      InputsAtCompileTime = NX,
      ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;


    const int nInputs;
    const int nValues;

    int inputs() const
    {
      return nInputs;
    }

    int values() const
    {
      return nValues;
    }

  };


  // Functor for computing the sampson error in a homography matrix algorithm.
  struct SampsonCostFunctor : public Functor<double>
  {
    SampsonCostFunctor( 
        const sfm::std_vector_Vector2d* _x1vec,
        const sfm::std_vector_Vector2d* _x2vec,
        int _H_max_index )
      :Functor<double>(8,_x1vec->size()), nPoints(_x1vec->size()), 
      x1vec(_x1vec), x2vec(_x2vec),
      H_max_index(_H_max_index)
    { }

    const int nPoints;
    const sfm::std_vector_Vector2d* x1vec;
    const sfm::std_vector_Vector2d* x2vec;
    int H_max_index;


    inline
    static double cost( const Eigen::Matrix3d& H, 
                        const Eigen::Vector2d& x1, 
                        const Eigen::Vector2d& x2 )
    {
      return  sampsonDistanceSqr(H, x1, x2);     
    }
    

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
      Eigen::Matrix3d H;
      {
        int j = 0;
        for( int i = 0; i < 9; ++i )
        {
          if( i != H_max_index )
          {
            H(i) = x(j);
            ++j;
          }
          else
            H(i) = 1;
        }
      }

      Eigen::Vector3d x_tmp;
      for( int i = 0; i < nPoints; ++i )
      {
        fvec(i) = cost(H, (*x1vec)[i], (*x2vec)[i]);
      }

      return 0;
    }


  
  };





  // stores N random numbers in r that are in the range [0,max).
  // The numbers in r do not contain duplicates
  template <int N>
  inline
  void randomNoDuplicates(int* r, int max)
  {
    // remove duplicates
    for( int i = 0; i < N; ++i )
    {
      bool duplicate;
      do
      {
        duplicate = false;
        r[i] = (int)((rand()/(float)RAND_MAX-std::numeric_limits<float>::epsilon())*max);
        //r[i] = rand() % max;

        // check for duplicates
        for( int j = 0; j < i; ++j )
          if( r[i] == r[j] )
          {
            duplicate = true;
            break;
          }

      } while( duplicate );
    }
  }


  boost::random::mt19937 rng;

  // stores N random numbers in r.
  // The numbers in r do not contain duplicates
  template <int N>
  inline
  void randomNoDuplicates(int* r,boost::random::uniform_int_distribution<>& dist)
  {
    // remove duplicates
    for( int i = 0; i < N; ++i )
    {
      bool duplicate;
      do
      {
        duplicate = false;
        r[i] = dist(rng);

        // check for duplicates
        for( int j = 0; j < i; ++j )
          if( r[i] == r[j] )
          {
            duplicate = true;
            break;
          }

      } while( duplicate );
    }
  }


} // namespace


Eigen::Matrix3d sfm::computeHomography(
    const std_vector_Vector2d& x1vec,
    const std_vector_Vector2d& x2vec )
{
  if( x1vec.size() < 4 )
    throw std::runtime_error("computeHomography: 4 point correspondences required\n");
  if( x1vec.size() != x2vec.size() )
    throw std::runtime_error("computeHomography: number of points mismatch\n");
  // 
  // Normalization: Translate centroid to the origin and scale such that the
  // mean distance of all points to the origin is sqrt(2)
  //
  std_vector_Vector2d x1vec_norm(x1vec);
  std_vector_Vector2d x2vec_norm(x2vec);
  Eigen::Matrix3d T1( normalizePoints(x1vec_norm) );
  Eigen::Matrix3d T2( normalizePoints(x2vec_norm) );
  
  Eigen::MatrixXd A(2*x1vec.size(),9);

  for( int i = 0; i < (int)x1vec.size(); ++i )
  {
    Eigen::Vector2d& x1 = x1vec_norm[i];
    Eigen::Vector2d& x2 = x2vec_norm[i];
    A(i*2+0,0) = 0;
    A(i*2+0,1) = 0;
    A(i*2+0,2) = 0;
    A(i*2+0,3) = -x1.x();
    A(i*2+0,4) = -x1.y();
    A(i*2+0,5) = -1;
    A(i*2+0,6) = x2.y()*x1.x();
    A(i*2+0,7) = x2.y()*x1.y();
    A(i*2+0,8) = x2.y();

    A(i*2+1,0) = x1.x();
    A(i*2+1,1) = x1.y();
    A(i*2+1,2) = 1;
    A(i*2+1,3) = 0;
    A(i*2+1,4) = 0;
    A(i*2+1,5) = 0;
    A(i*2+1,6) = -x2.x()*x1.x();
    A(i*2+1,7) = -x2.x()*x1.y();
    A(i*2+1,8) = -x2.x();

  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
  Eigen::Matrix3d H;
  H.row(0) = svd.matrixV().block(0,8,3,1).transpose();
  H.row(1) = svd.matrixV().block(3,8,3,1).transpose();
  H.row(2) = svd.matrixV().block(6,8,3,1).transpose();

  // 
  // denormalize
  //
  Eigen::Matrix3d T2_inv(Eigen::Matrix3d::Zero());
  T2_inv(0,0) = 1/T2(0,0);
  T2_inv(1,1) = 1/T2(1,1);
  T2_inv(2,2) = 1;
  T2_inv(0,2) = -T2(0,2)/T2(0,0);
  T2_inv(1,2) = -T2(1,2)/T2(1,1);
      
  H = T2_inv*H*T1;

  return H;
}




void sfm::computeHomographySampson(
    Eigen::Matrix3d& H,
    const std_vector_Vector2d& x1,
    const std_vector_Vector2d& x2 )
{
  int nPoints = x1.size();

  // scale such that the largest element of H is 1
  int H_max_index = 0;
  for( int i = 1; i < 9; ++i )
  {
    if( std::abs(H(i)) > std::abs(H(H_max_index)) )
      H_max_index = i;
  }
  H /= H(H_max_index);

  // 
  // Setup variable vector
  //
  Eigen::VectorXd vars( 8 );
  {
    int j = 0;
    for( int i = 0; i < 9; ++i )
    {
      if( i != H_max_index )
      {
        vars(j) = H(i);
        ++j;
      }
    }
  }


  SampsonCostFunctor functor(&x1, &x2, H_max_index);
  Eigen::VectorXd res( nPoints );

  //functor(vars, res);
  //res.cwiseProduct(res);
  //std::cout << "resnorm = " << res.norm() << std::endl;

  Eigen::NumericalDiff<SampsonCostFunctor> numDiff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SampsonCostFunctor> > lm(numDiff);
  int status = lm.minimizeInit(vars);
  int steps = 0;
  do
  {
    //std::cout << " step = " << steps << std::endl;
    status = lm.minimizeOneStep(vars);
    ++steps;
  } while (status == Eigen::LevenbergMarquardtSpace::Running && steps < 50); 

  //functor(vars, res);
  //res.cwiseProduct(res);

  {
    int j = 0;
    for( int i = 0; i < 9; ++i )
    {
      if( i != H_max_index )
      {
        H(i) = vars(j);
        ++j;
      }
      else
        H(i) = 1;
    }
  }


}




void sfm::computeHomographyRANSAC(
    Eigen::Matrix3d& H,
    Eigen::VectorXi& inliers,
    const std_vector_Vector2d& x1,
    const std_vector_Vector2d& x2,
    int samples,
    double threshold )
{
  int nPoints = x1.size();

  Eigen::Matrix3d H_tmp, H_tmp_inv;
  int nInliers_best = -1; //  ensures H_best is replaced in the first iteration
  std::vector<bool>* inliers_best = new std::vector<bool>(nPoints,false);
  std::vector<bool>* inliers_tmp = new std::vector<bool>(nPoints,false);

  std_vector_Vector2d x1_min(4,Eigen::Vector2d());
  std_vector_Vector2d x2_min(4,Eigen::Vector2d());
  
  std_vector_Vector2d x1_inl;
  std_vector_Vector2d x2_inl;
  x1_inl.reserve(nPoints);
  x2_inl.reserve(nPoints);

  boost::random::uniform_int_distribution<> dist(0,nPoints-1);

  for( int i = 0; i < samples; ++i )
  {
    // draw 4 random points
    int idx[4];
    //randomNoDuplicates<4>(idx, nPoints);
    randomNoDuplicates<4>(idx, dist);

    int nInliers = 0;

    // compute H from minimal set
    for( int i = 0; i < 4; ++i )
    {
      x1_min[i] = x1[idx[i]];
      x2_min[i] = x2[idx[i]];
    }
    H_tmp = computeHomography(x1_min, x2_min);
    H_tmp_inv = H_tmp.inverse();

    // determine inliers
    // and add to the x1_inl, x2_inl vectors
    for( int i = 0; i < nPoints; ++i )
    {
      //double cost = SampsonCostFunctor::cost( H_tmp, x1[i], x2[i] );
      double cost = symmetricTransferError( H_tmp, H_tmp_inv, x1[i], x2[i] );
      if( cost < threshold )
      {
        (*inliers_tmp)[i] = true;
        x1_inl.push_back(x1[i]);
        x2_inl.push_back(x2[i]);
        ++nInliers;
      }
      else
        (*inliers_tmp)[i] = false;
    }
    // compute again H but this time using the inliers
    if( x1_inl.size() >= 4 )
    {
      H_tmp = computeHomography(x1_inl, x2_inl);
      if( x1_inl.size() > 20 ) // TODO the optimization fails sometimes if there are not enough points
        computeHomographySampson(H_tmp, x1_inl, x2_inl);
    }

    x1_inl.clear();
    x2_inl.clear();

    // determine the inliers again to get the consensus set for the current model
    nInliers = 0;
    H_tmp_inv = H_tmp.inverse();
    for( int i = 0; i < nPoints; ++i )
    {
      //double cost = SampsonCostFunctor::cost( H_tmp, x1[i], x2[i] );
      double cost = symmetricTransferError( H_tmp, H_tmp_inv, x1[i], x2[i] );
      if( cost < threshold )
      {
        (*inliers_tmp)[i] = true;
        ++nInliers;
      }
      else
        (*inliers_tmp)[i] = false;
    }

    if( nInliers > nInliers_best )
    {
      nInliers_best = nInliers;
      std::swap( inliers_tmp, inliers_best );
    }

  }

  // recompute H for the last time with the final set of inliers
  inliers.resize(nPoints,1);
  for( int i = 0; i < nPoints; ++i )
  {
    if( (*inliers_best)[i] )
    {
      x1_inl.push_back(x1[i]);
      x2_inl.push_back(x2[i]);
      inliers(i) = 1;
    }
    else
      inliers(i) = 0;
  }
  H = computeHomography(x1_inl, x2_inl);
  if( x1_inl.size() > 20 ) // TODO the optimization fails sometimes if there are not enough points
    computeHomographySampson(H,x1_inl, x2_inl);

  delete inliers_tmp;
  delete inliers_best;
}


#include <iostream>

template <class T>
T sfm::computeGRICHomography( const Eigen::Matrix<T,3,3>& H,
                              const std_vector_Vector2T& x1,
                              const std_vector_Vector2T& x2,
                              T sigma )
{
  T k(8), d(2), r(4), n(x1.size());
  T lambda1( std::log(r) );
  T lambda2( std::log(r*n) );
  T lambda3(2);

  Eigen::Matrix<T,3,3> H_inv( H.inverse() );
  T sigma_sqr_inv( T(1)/(sigma*sigma) );
  T tmp( lambda3*(r-d) );

  T result(0);
  for( int i = 0; i < (int)x1.size(); ++i )
  {
    //T err( symmetricTransferError(H,H_inv,x1[i],x2[i]) );
    T err( sampsonDistanceSqr(H,x1[i],x2[i]) );
    result += std::min(err*sigma_sqr_inv, tmp);
    //std::cout << "(err*sigma_sqr_inv, tmp)" << err*sigma_sqr_inv <<"   " << tmp << std::endl;
  }

  result += n*d*lambda1 + k*lambda2;

  return result; 
}
template double sfm::computeGRICHomography(const Eigen::Matrix3d&, 
                                           const std_vector_Vector2d&,
                                           const std_vector_Vector2d&,
                                           double );
template float  sfm::computeGRICHomography(const Eigen::Matrix3f&, 
                                           const std_vector_Vector2f&,
                                           const std_vector_Vector2f&,
                                           float );

