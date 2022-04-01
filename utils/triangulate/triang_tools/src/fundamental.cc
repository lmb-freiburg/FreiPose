#define SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#include "sfmtools.h"
#include <ransac.h>
#include <stdexcept>
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace {


  // Computes the symmetric distance to the epipolar lines
  template <class T>
  inline T symmetricTransferError( const Eigen::Matrix<T,3,3>& F,
                                   const Eigen::Matrix<T,2,1>& x1,
                                   const Eigen::Matrix<T,2,1>& x2 )
  {
    Eigen::Matrix<T,3,1> _x1, _x2;
    _x1 << x1, 1;
    _x2 << x2, 1;
    Eigen::Matrix<T,3,1> l2 = F*_x1;
    Eigen::Matrix<T,3,1> l1 = F.transpose()*_x2;

    T tmp = l2.dot(_x2);
    T err = (tmp*tmp)/l2.template topRows<2>().squaredNorm();

    tmp = l1.dot(_x1);
    err += (tmp*tmp)/l1.template topRows<2>().squaredNorm();

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


  // Functor for computing the sampson error in a fundamental matrix algorithm.
  template <class T>
  struct SampsonCostFunctor : public Functor<T>
  {
    SampsonCostFunctor( 
        const typename sfm::Type<T>::std_vector_Vector2* _x1vec,
        const typename sfm::Type<T>::std_vector_Vector2* _x2vec,
        int _F_max_index )
      :Functor<T>(8,_x1vec->size()), nPoints(_x1vec->size()), 
      x1vec(_x1vec), x2vec(_x2vec),
      F_max_index(_F_max_index)
    { }

    const int nPoints;
    const sfm::std_vector_Vector2d* x1vec;
    const sfm::std_vector_Vector2d* x2vec;
    int F_max_index;


    inline
    static T cost( const Eigen::Matrix<T,3,3>& F, 
                   const Eigen::Matrix<T,2,1>& x1, 
                   const Eigen::Matrix<T,2,1>& x2 )
    {
      Eigen::Matrix<T,3,1> Fx1 = F*x1.homogeneous();
      Eigen::Matrix<T,3,1> FTx2 = F.transpose()*x2.homogeneous();

      T numerator = x2.homogeneous().dot(Fx1);
      numerator *= numerator;

      T denominator = Fx1.template topRows<2>().squaredNorm() + FTx2.template topRows<2>().squaredNorm();

      return numerator / denominator;
    }
    

    int operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, Eigen::Matrix<T,Eigen::Dynamic,1>& fvec) const
    {
      Eigen::Matrix<T,3,3> F;
      {
        int j = 0;
        for( int i = 0; i < 9; ++i )
        {
          if( i != F_max_index )
          {
            F(i) = x(j);
            ++j;
          }
          else
            F(i) = 1;
        }
      }

      // TODO check this
      // enforce det(F) = 0
      //Eigen::JacobiSVD<Eigen::Matrix3d> svd(F,Eigen::ComputeFullU|Eigen::ComputeFullV);
      //Eigen::Vector3d S = svd.singularValues();
      //S(2) = 0;
      //F = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();

      Eigen::Matrix<T,3,1> x_tmp;
      for( int i = 0; i < nPoints; ++i )
      {
        fvec(i) = cost(F, (*x1vec)[i], (*x2vec)[i]);
      }

      return 0;
    }


    //int df(const Eigen::VectorXd& x, Eigen::MatrixXd& jac )
    //{
      //Eigen::Matrix3d F;
      //F.col(0) = x.block(0,0,3,1);
      //F.col(1) = x.block(3,0,3,1);
      //F.block(0,2,2,1) = x.block(6,0,2,1);
      //F(2,2) = 1;
      
      //for( int i = 0; i < nPoints; ++i )
      //{
        //Eigen::Vector3d x1;
        //x1 << (*x1vec)[i], 1;
        //Eigen::Vector3d x2;
        //x2 << (*x2vec)[i], 1;

        ////double dotx1fc1 = x1.dot(F.col(0));
        ////double dotx1fc2 = x1.dot(F.col(1));
        ////double dotx1fc3 = x1.dot(F.col(2));
        //double dotx2fc1 = x2.dot(F.col(0));
        //double dotx2fc2 = x2.dot(F.col(1));
        //double dotx2fc3 = x2.dot(F.col(2));
        //double dotx1fr1 = x1.dot(F.row(0));
        //double dotx1fr2 = x1.dot(F.row(1));
        ////double dotx1fr3 = x1.dot(F.row(2));
        ////double dotx2fr1 = x2.dot(F.row(0));
        ////double dotx2fr2 = x2.dot(F.row(1));
        ////double dotx2fr3 = x2.dot(F.row(2));

        //double num = dotx2fc3 + dotx2fc1*x1.x() + dotx2fc2*x1.y();
        //double num_sqr = num*num;
        //double denom = dotx1fr1*dotx1fr1  
                     //+ dotx1fr2*dotx1fr2  
                     //+ dotx2fc1*dotx2fc1  
                     //+ dotx2fc2*dotx2fc2;
        //double denom_sqr = denom*denom;
        
        //double eta = num/denom;
        //double eta_sqr = num_sqr/denom_sqr;

        //jac(i,0) = 2*x1.x()*x2.x()*eta - 2*(dotx1fr1*x1.x()+dotx2fc1*x2.x())*eta_sqr;
        //jac(i,1) = 2*x1.x()*x2.y()*eta - 2*(dotx1fr2*x1.x()+dotx2fc1*x2.y())*eta_sqr;
        //jac(i,2) = 2*x1.x()*eta        - 2*dotx2fc1*eta_sqr;
        //jac(i,3) = 2*x1.y()*x2.x()*eta - 2*(dotx1fr1*x1.y()+dotx2fc2*x2.x())*eta_sqr;
        //jac(i,4) = 2*x1.x()*x2.y()*eta - 2*(dotx1fr2*x1.y()+dotx2fc2*x2.y())*eta_sqr;
        //jac(i,5) = 2*x1.y()*eta        - 2*dotx2fc2*eta_sqr;
        //jac(i,6) = 2*x2.x()*eta        - 2*dotx1fr1*eta_sqr;
        //jac(i,7) = 2*x2.y()*eta        - 2*dotx1fr2*eta_sqr;
      //}
      //return 0;
    //}

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


Eigen::Matrix3d sfm::computeFundamental8Point(
      const std_vector_Vector2d& x1vec,
      const std_vector_Vector2d& x2vec )
{
  if( x1vec.size() < 8 )
    throw std::runtime_error("computeFundamental8Point: 8 point correspondences required\n");
  if( x1vec.size() != x2vec.size() )
    throw std::runtime_error("computeFundamental8Point: number of points mismatch\n");
  // 
  // Normalization: Translate centroid to the origin and scale such that the
  // mean distance of all points to the origin is sqrt(2)
  //
  std_vector_Vector2d x1vec_norm(x1vec);
  std_vector_Vector2d x2vec_norm(x2vec);
  Eigen::Matrix3d T1( normalizePoints(x1vec_norm) );
  Eigen::Matrix3d T2( normalizePoints(x2vec_norm) );
  
  Eigen::MatrixXd A(x1vec.size(),9);

  for( int i = 0; i < A.rows(); ++i )
  {
    Eigen::Vector2d& x1 = x1vec_norm[i];
    Eigen::Vector2d& x2 = x2vec_norm[i];
    A(i,0) = x2(0)*x1(0);
    A(i,1) = x2(0)*x1(1);
    A(i,2) = x2(0);
    A(i,3) = x2(1)*x1(0);
    A(i,4) = x2(1)*x1(1);
    A(i,5) = x2(1);
    A(i,6) = x1(0);
    A(i,7) = x1(1);
    A(i,8) = 1;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
  Eigen::Matrix3d F;
  F.row(0) = svd.matrixV().block<3,1>(0,8).transpose();
  F.row(1) = svd.matrixV().block<3,1>(3,8).transpose();
  F.row(2) = svd.matrixV().block<3,1>(6,8).transpose();

  // Enforce the 'det(F) = 0' contraint
  Eigen::JacobiSVD<Eigen::Matrix3d> svd2(F,Eigen::ComputeFullU|Eigen::ComputeFullV);
  Eigen::Vector3d S( svd2.singularValues() );
  S(2) = 0; 
  F = svd2.matrixU() * S.asDiagonal() * svd2.matrixV().transpose();
  F = T2.transpose()*F*T1;

  return F;
}




void sfm::computeFundamentalSampson(
    Eigen::Matrix3d& F,
    const std_vector_Vector2d& x1,
    const std_vector_Vector2d& x2 )
{
  int nPoints = x1.size();

  // scale such that the largest element of F is 1
  int F_max_index = 0;
  for( int i = 1; i < 9; ++i )
  {
    if( std::abs(F(i)) > std::abs(F(F_max_index)) )
      F_max_index = i;
  }
  F /= F(F_max_index);

  // 
  // Setup variable vector
  //
  Eigen::VectorXd vars( 8 );
  {
    int j = 0;
    for( int i = 0; i < 9; ++i )
    {
      if( i != F_max_index )
      {
        vars(j) = F(i);
        ++j;
      }
    }
  }

  // 
  // Ignore points that are close to the epipole
  //
  Eigen::Vector2d e1, e2;
  {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F,Eigen::ComputeFullU|Eigen::ComputeFullV);
    e1 = svd.matrixV().col(2).topRows<2>()/svd.matrixV().col(2).z();
    e2 = svd.matrixU().col(2).topRows<2>()/svd.matrixU().col(2).z();
    //std::cout << " e1 " << e1.transpose() << std::endl;
    //std::cout << " e2 " << e2.transpose() << std::endl;
    //std::cout << F*Eigen::Vector3d(e1.x(),e1.y(),1) << " \n\n" << F.transpose()*Eigen::Vector3d(e2.x(),e2.y(),1) << std::endl;
  }
  std_vector_Vector2d x1_tmp;
  x1_tmp.reserve(x1.size());
  std_vector_Vector2d x2_tmp;
  x2_tmp.reserve(x2.size());
  for( int i = 0; i < nPoints; ++ i )
  {
    if( (e1-x1[i]).squaredNorm() > 1.0e-4 && 
        (e2-x2[i]).squaredNorm() > 1.0e-4 )
    {
      x1_tmp.push_back(x1[i]);
      x2_tmp.push_back(x2[i]);
    }
    //else
    //{
      //std::cout << " point " << x1[i].transpose() << std::endl;
    //}
  }
  //std::cout << "points close to epipole " << x1.size() - x1_tmp.size() << std::endl;
  //std::cout << "points total " << x1.size() << std::endl;


  SampsonCostFunctor<double> functor(&x1_tmp, &x2_tmp, F_max_index);
  Eigen::VectorXd res( nPoints );

  //functor(vars, res);
  //res.cwiseProduct(res);
  //std::cout << "resnorm = " << res.norm() << std::endl;

  // numerical diff works better or df() implementation is wrong in SampsonCostFunctor
  Eigen::NumericalDiff<SampsonCostFunctor<double> > numDiff(functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<SampsonCostFunctor<double> > > lm(numDiff);
  //Eigen::LevenbergMarquardt<SampsonCostFunctor> lm(functor);
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
      if( i != F_max_index )
      {
        F(i) = vars(j);
        ++j;
      }
      else
        F(i) = 1;
    }
  }

  // make sure that det(F)=0 holds
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(F,Eigen::ComputeFullU|Eigen::ComputeFullV);
  Eigen::Vector3d S = svd.singularValues();
  S(2) = 0;
  F = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();

}


namespace
{
  struct computeFundamentalRANSAC_ModelFn
  {
    typedef Eigen::Matrix<double,3,3,Eigen::DontAlign> ModelType;
    enum Parameters {MINIMUM_SAMPLES=8};

    computeFundamentalRANSAC_ModelFn( const sfm::std_vector_Vector2d& x1,
                                      const sfm::std_vector_Vector2d& x2 )
      :x1(x1), x2(x2)
    {
      x1_min.resize(MINIMUM_SAMPLES);
      x2_min.resize(MINIMUM_SAMPLES);
    }

    bool operator()(std::vector<ModelType>& M_vec, const int* idx)
    {
      M_vec.clear();
      ModelType M;

      for( int i = 0; i < MINIMUM_SAMPLES; ++i )
      {
        x1_min[i] = x1[idx[i]];
        x2_min[i] = x2[idx[i]];
      }

      M = sfm::computeFundamental8Point(x1_min, x2_min);

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

      sfm::std_vector_Vector2d x1_inliers(inliers_count);
      sfm::std_vector_Vector2d x2_inliers(inliers_count);
      int index = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
      {
        if( inliers[i] )
        {
          x1_inliers[index] = x1[i];
          x2_inliers[index] = x2[i];
          ++index;
        }
      }

      Eigen::Matrix3d F = sfm::computeFundamental8Point(x1_inliers, x2_inliers);
      if( x1_inliers.size() > 20 ) // TODO the optimization fails sometimes if there are not enough points
        sfm::computeFundamentalSampson(F, x1_inliers, x2_inliers);
      M = F;
      return true;
    }

    const sfm::std_vector_Vector2d& x1;
    const sfm::std_vector_Vector2d& x2;
    sfm::std_vector_Vector2d x1_min;
    sfm::std_vector_Vector2d x2_min;
  };

  struct computeFundamentalRANSAC_DistanceFn
  {
    typedef Eigen::Matrix<double,3,3,Eigen::DontAlign> ModelType;

    computeFundamentalRANSAC_DistanceFn( const sfm::std_vector_Vector2d& x1,
                                         const sfm::std_vector_Vector2d& x2 )
      :x1(x1), x2(x2)
    { }


    double operator()(const ModelType& M, int idx)
    {
      double cost = SampsonCostFunctor<double>::cost( M, x1[idx], x2[idx] );
      return cost;
    }

    const sfm::std_vector_Vector2d& x1;
    const sfm::std_vector_Vector2d& x2;
  };

} // namespace

void sfm::computeFundamentalRANSAC(
    Eigen::Matrix3d& F,
    Eigen::VectorXi& inliers,
    const std_vector_Vector2d& x1,
    const std_vector_Vector2d& x2,
    int samples,
    double threshold )
{
  if( x1.size() != x2.size() )
    throw std::runtime_error("computeFundamentalRANSAC: x1 and x2 point count mismatch\n");
  if( x1.size() < 8 )
    throw std::runtime_error("computeFundamentalRANSAC: Not enough points\n");

  computeFundamentalRANSAC_ModelFn modelFn(x1,x2);
  computeFundamentalRANSAC_DistanceFn distanceFn(x1,x2);

  typedef sfm::internal::RANSAC<double,computeFundamentalRANSAC_ModelFn,computeFundamentalRANSAC_DistanceFn> RANSACType;
    
  RANSACType ransac(modelFn, distanceFn, x1.size(), threshold);
  typename RANSACType::ModelType M_tmp;
  M_tmp.setZero(); // get rid of unitialized value warning
  ransac.run(M_tmp,inliers,samples);

  F = M_tmp;
  /*
  int nPoints = x1.size();

  Eigen::Matrix3d F_tmp;
  int nInliers_best = -1; //  ensures F_best is replaced in the first iteration
  std::vector<bool>* inliers_best = new std::vector<bool>(nPoints,false);
  std::vector<bool>* inliers_tmp = new std::vector<bool>(nPoints,false);

  std_vector_Vector2d x1_min(8,Eigen::Vector2d());
  std_vector_Vector2d x2_min(8,Eigen::Vector2d());
  
  std_vector_Vector2d x1_inl;
  std_vector_Vector2d x2_inl;
  x1_inl.reserve(nPoints);
  x2_inl.reserve(nPoints);

  boost::random::uniform_int_distribution<> dist(0,nPoints-1);

  for( int i = 0; i < samples; ++i )
  {
    // draw 8 random points
    int idx[8];
    //randomNoDuplicates<8>(idx, nPoints);
    randomNoDuplicates<8>(idx, dist);

    int nInliers = 0;

    // compute F from minimal set
    for( int i = 0; i < 8; ++i )
    {
      x1_min[i] = x1[idx[i]];
      x2_min[i] = x2[idx[i]];
    }
    F_tmp = computeFundamental8Point(x1_min, x2_min);

    // determine inliers
    // and add to the x1_inl, x2_inl vectors
    for( int i = 0; i < nPoints; ++i )
    {
      double cost = SampsonCostFunctor<double>::cost( F_tmp, x1[i], x2[i] );
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
    // compute again F but this time using the inliers
    if( x1_inl.size() >= 8 )
    {
      F_tmp = computeFundamental8Point(x1_inl, x2_inl);
      if( x1_inl.size() > 20 ) // TODO the optimization fails sometimes if there are not enough points
        computeFundamentalSampson(F_tmp, x1_inl, x2_inl);
    }

    x1_inl.clear();
    x2_inl.clear();

    // determine the inliers again to get the consensus set for the current model
    nInliers = 0;
    for( int i = 0; i < nPoints; ++i )
    {
      double cost = SampsonCostFunctor<double>::cost( F_tmp, x1[i], x2[i] );
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

  // recompute F for the last time with the final set of inliers
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
  F = computeFundamental8Point(x1_inl, x2_inl);
  if( x1_inl.size() > 20 ) // TODO the optimization fails sometimes if there are not enough points
    computeFundamentalSampson(F,x1_inl, x2_inl);

  delete inliers_tmp;
  delete inliers_best;
  */
}



template <class T>
T sfm::computeGRICFundamental( const Eigen::Matrix<T,3,3>& F,
                               const std_vector_Vector2T& x1,
                               const std_vector_Vector2T& x2,
                               T sigma )
{
  T k(7), d(3), r(4), n(x1.size());
  T lambda1( std::log(r) );
  T lambda2( std::log(r*n) );
  T lambda3(2);

  T sigma_sqr_inv( T(1)/(sigma*sigma) );
  T tmp( lambda3*(r-d) );

  T result(0);
  for( int i = 0; i < (int)x1.size(); ++i )
  {
    T err( SampsonCostFunctor<T>::cost(F,x1[i],x2[i]) );
    result += std::min(err*sigma_sqr_inv, tmp);
    //std::cout << "        (err*sigma_sqr_inv, tmp)" << err*sigma_sqr_inv <<"   " << tmp << std::endl;
  }

  result += n*d*lambda1 + k*lambda2;

  return result; 
}
template double sfm::computeGRICFundamental(const Eigen::Matrix3d&, 
                                            const std_vector_Vector2d&,
                                            const std_vector_Vector2d&,
                                            double );
template float  sfm::computeGRICFundamental(const Eigen::Matrix3f&, 
                                            const std_vector_Vector2f&,
                                            const std_vector_Vector2f&,
                                            float );
