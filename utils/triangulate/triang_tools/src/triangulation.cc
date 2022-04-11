#define SFMTOOLS_NO_UNDEF_TEMPLATE_DEFINES
#include "sfmtools.h"
#include "ransac.h"
#include <stdexcept>
#include <iostream>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/NonLinearOptimization>

namespace {

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


  // Functor for computing the reprojection error in a triangulation algorithm.
  // Input is a 3D point X. Output is the x and y error for each camera
  template <class T>
  struct TriangulateReprojectionCostFunctor : public Functor<T>
  {
    TriangulateReprojectionCostFunctor( 
        const typename sfm::Type<T>::std_vector_Matrix3x4* _Pvec,
        const typename sfm::Type<T>::std_vector_Vector2* _xvec,
        const std::vector<T>* _weights
        )
      :Functor<T>(3,_Pvec->size()*2), n(_Pvec->size()), Pvec(_Pvec), xvec(_xvec),
       weights(_weights)
    { }

    const int n;
    const typename sfm::Type<T>::std_vector_Matrix3x4* Pvec;
    const typename sfm::Type<T>::std_vector_Vector2* xvec;
    const std::vector<T>* weights;

    int operator()(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
                   Eigen::Matrix<T,Eigen::Dynamic,1>& fvec) const
    {
      Eigen::Matrix<T,4,1> X;
      X << x, 1;
      for( int i = 0; i < n; ++i )
      {
        Eigen::Matrix<T,3,1> tmp = (*Pvec)[i]*X;
        if( std::abs(tmp.z() < T(1.0e-6)) )
        {
          // The point may not coincide with the camera -> high costs
          fvec(i*2+0) = std::numeric_limits<T>::max();
          fvec(i*2+1) = std::numeric_limits<T>::max();
        }
        else
        {
          Eigen::Matrix<T,2,1> x_proj = tmp.topRows(2) / tmp.z();
          Eigen::Matrix<T,2,1> diff = x_proj - (*xvec)[i];

          fvec(i*2+0) = (*weights)[i]*diff(0); 
          fvec(i*2+1) = (*weights)[i]*diff(1); 
        }
      }
      return 0;
    }

    int df(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, 
           Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& jac )
    {
      Eigen::Matrix<T,4,1> X;
      X << x, 1;

      for( int i = 0; i < n; ++i )
      {
        const Eigen::Matrix<T,3,4>& P = (*Pvec)[i];

        Eigen::Matrix<T,3,1> x_proj = P*X;
        if( std::abs(x_proj.z()) < 1.0e-6 )
        {
          // This is a desperate attempt to move the point away from the
          // camera in the next iteration
          jac.block(i*2,0,2,3).setRandom();
          jac.block(i*2,0,2,3) *= 1.0e-3;
        }
        else
        {
          T z_sqr = x_proj.z()*x_proj.z();
          
          jac(i*2+0,0) = (*weights)[i]*(P(0,0)*x_proj.z() - P(2,0)*x_proj.x())/(z_sqr);
          jac(i*2+0,1) = (*weights)[i]*(P(0,1)*x_proj.z() - P(2,1)*x_proj.x())/(z_sqr);
          jac(i*2+0,2) = (*weights)[i]*(P(0,2)*x_proj.z() - P(2,2)*x_proj.x())/(z_sqr);
          jac(i*2+1,0) = (*weights)[i]*(P(1,0)*x_proj.z() - P(2,0)*x_proj.y())/(z_sqr);
          jac(i*2+1,1) = (*weights)[i]*(P(1,1)*x_proj.z() - P(2,1)*x_proj.y())/(z_sqr);
          jac(i*2+1,2) = (*weights)[i]*(P(1,2)*x_proj.z() - P(2,2)*x_proj.y())/(z_sqr);
        }
      }
      return 0;
    }

    // this function is only for debugging
    static T energyFunc( const Eigen::Matrix<T,4,1>& X,
                         const typename sfm::Type<T>::std_vector_Matrix3x4& Pvec,
                         const typename sfm::Type<T>::std_vector_Vector2& xvec,
                         const std::vector<T> weights )
    {
      T energy = 0;
      int n = Pvec.size();
      for( int i = 0; i < n; ++i )
      {
        Eigen::Matrix<T,3,1> tmp = Pvec[i]*X;
        Eigen::Matrix<T,2,1> x_proj = tmp.topRows(2) / tmp.z();
        energy += (xvec[i] - x_proj).squaredNorm();
      }
      return energy;
    }

  };

} // namespace



template <class T>
void sfm::triangulate(
      std_vector_Vector3T& X,
      const Eigen::Matrix<T,3,3>& K1,
      const Eigen::Matrix<T,3,3>& R1,
      const Eigen::Matrix<T,3,1>& t1,
      const Eigen::Matrix<T,3,3>& K2,
      const Eigen::Matrix<T,3,3>& R2,
      const Eigen::Matrix<T,3,1>& t2,
      const std_vector_Vector2T& x1,
      const std_vector_Vector2T& x2
      )
{
  if( x1.size() != x2.size() )
  {
    throw std::runtime_error("triangulate: measurements count mismatch");
  }

  // normalize points to improve numerical stability
  typename Type<T>::std_vector_Vector2 x1_normalized(x1);
  typename Type<T>::std_vector_Vector2 x2_normalized(x2);

  Eigen::Matrix<T,3,3> T1 = normalizePoints( x1_normalized );
  Eigen::Matrix<T,3,3> T2 = normalizePoints( x2_normalized );

  typename Type<T>::std_vector_Matrix3x4 Pvec(2);
  Pvec[0] << R1, t1;
  Pvec[0] = T1*K1*Pvec[0];
  Pvec[1] << R2, t2;
  Pvec[1] = T2*K2*Pvec[1];
  
  typename Type<T>::std_vector_Vector2 xvec(2);

  X.resize(x1.size());
  for( size_t i = 0; i < x1_normalized.size(); ++i )
  {
    Eigen::Matrix<T,2,1>& _x1 = x1_normalized[i];
    Eigen::Matrix<T,2,1>& _x2 = x2_normalized[i];

    xvec[0] = _x1;
    xvec[1] = _x2;

    X[i] = triangulateLinear(Pvec, xvec);
    triangulateNonlinear(X[i], Pvec, xvec);
  }
}
template void sfm::triangulate(
      std_vector_Vector3d& X,
      const Eigen::Matrix<double,3,3>& K1,
      const Eigen::Matrix<double,3,3>& R1,
      const Eigen::Matrix<double,3,1>& t1,
      const Eigen::Matrix<double,3,3>& K2,
      const Eigen::Matrix<double,3,3>& R2,
      const Eigen::Matrix<double,3,1>& t2,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2
      );
template void sfm::triangulate(
      std_vector_Vector3f& X,
      const Eigen::Matrix<float,3,3>& K1,
      const Eigen::Matrix<float,3,3>& R1,
      const Eigen::Matrix<float,3,1>& t1,
      const Eigen::Matrix<float,3,3>& K2,
      const Eigen::Matrix<float,3,3>& R2,
      const Eigen::Matrix<float,3,1>& t2,
      const std_vector_Vector2f& x1,
      const std_vector_Vector2f& x2
      );




template <class T>
void sfm::triangulate(
      std_vector_Vector3T& X,
      const Eigen::Matrix<T,3,4>& P1,
      const Eigen::Matrix<T,3,4>& P2,
      const std_vector_Vector2T& x1,
      const std_vector_Vector2T& x2
      )
{
  if( x1.size() != x2.size() )
  {
    throw std::runtime_error("triangulate: measurements count mismatch");
  }

  X.resize(x1.size());

  typename Type<T>::std_vector_Matrix3x4 Pvec(2);
  Pvec[0] = P1;
  Pvec[1] = P2;
  typename Type<T>::std_vector_Vector2 xvec(2);

  for( size_t i = 0; i < x1.size(); ++i )
  {
    xvec[0] = x1[i];
    xvec[1] = x2[i];

    X[i] = triangulateLinear(Pvec, xvec);
    triangulateNonlinear( X[i], Pvec, xvec );    
  }

}
template void sfm::triangulate(
      std_vector_Vector3d& X,
      const Matrix3x4d& P1,
      const Matrix3x4d& P2,
      const std_vector_Vector2d& x1,
      const std_vector_Vector2d& x2
      );
template void sfm::triangulate(
      std_vector_Vector3f& X,
      const Matrix3x4f& P1,
      const Matrix3x4f& P2,
      const std_vector_Vector2f& x1,
      const std_vector_Vector2f& x2
      );






template <class T>
Eigen::Matrix<T,3,1> sfm::triangulateLinear(
    const std_vector_Matrix3x4T& Pvec,
    const std_vector_Vector2T& xvec )
{
  if( Pvec.size() != xvec.size() )
    throw std::runtime_error("triangulateLinear: Cameras and points count mismatch\n");
  if( Pvec.size() < 2 )
    throw std::runtime_error("triangulateLinear: Not enough cameras/points\n");

  int n = Pvec.size();


#if 1
  Eigen::Matrix<T,Eigen::Dynamic,3> A(2*n,3);
  Eigen::Matrix<T,Eigen::Dynamic,1> b(2*n);

  for( int i = 0; i < n; ++i )
  {
    T x = xvec[i].x();
    T y = xvec[i].y();
    T z = 1;

    const Eigen::Matrix<T,3,4>& P = Pvec[i];
    A.row(2*i+0) = y*P.block(2,0,1,3) - z*P.block(1,0,1,3); 
    A.row(2*i+1) = z*P.block(0,0,1,3) - x*P.block(2,0,1,3);

    b(2*i+0) = z*P(1,3) - y*P(2,3);
    b(2*i+1) = x*P(2,3) - z*P(0,3);
  }
#else
  // 3 equations version
  Eigen::Matrix<T,Eigen::Dynamic,3> A(3*n,3);
  Eigen::Matrix<T,Eigen::Dynamic,1> b(3*n);

  for( int i = 0; i < n; ++i )
  {
    T x = xvec[i].x();
    T y = xvec[i].y();
    T z = 1;

    const Eigen::Matrix<double,3,4>& P = Pvec[i];
    A.row(3*i+0) = y*P.block(2,0,1,3) - z*P.block(1,0,1,3); 
    A.row(3*i+1) = z*P.block(0,0,1,3) - x*P.block(2,0,1,3);
    A.row(3*i+2) = x*P.block(1,0,1,3) - y*P.block(0,0,1,3);

    b(3*i+0) = z*P(1,3) - y*P(2,3);
    b(3*i+1) = x*P(2,3) - z*P(0,3);
    b(3*i+2) = y*P(0,3) - x*P(1,3);
  }
#endif

  Eigen::JacobiSVD<Eigen::Matrix<T,Eigen::Dynamic,3> > svd(A,Eigen::ComputeFullU|Eigen::ComputeFullV);

  Eigen::Matrix<T,3,1> X; 
  X = svd.solve( b );

  return X;
}
template Eigen::Vector3d sfm::triangulateLinear(const std_vector_Matrix3x4d&,
                                                const std_vector_Vector2d&);
template Eigen::Vector3f sfm::triangulateLinear(const std_vector_Matrix3x4f&,
                                                const std_vector_Vector2f&);



template <class T>
void sfm::triangulateNonlinear( 
    Eigen::Matrix<T,3,1>& X,
    const std_vector_Matrix3x4T& Pvec,
    const std_vector_Vector2T& xvec,
    const std::vector<T> weights )
{
  if( Pvec.size() != xvec.size() )
    throw std::runtime_error("triangulateNonlinear: Cameras and points count mismatch\n");
  if( Pvec.size() < 2 )
    throw std::runtime_error("triangulateNonlinear: Not enough cameras/points\n");

  int n = Pvec.size();
  
  // setup weights
  std::vector<T> w( weights );
  w.resize( n, T(1) ); // set missing weights to 1

  TriangulateReprojectionCostFunctor<T> functor(&Pvec, &xvec, &w);
  //Eigen::NumericalDiff<TriangulateReprojectionCostFunctor<T> > numDiff(functor);
  //Eigen::LevenbergMarquardt<Eigen::NumericalDiff<TriangulateReprojectionCostFunctor<T> >,T> lm(numDiff);
  Eigen::LevenbergMarquardt<TriangulateReprojectionCostFunctor<T>,T> lm(functor);
  Eigen::Matrix<T,Eigen::Dynamic,1> vars( X );
  int status = lm.minimizeInit(vars);
  int steps = 0;
  do
  {
    //Eigen::Matrix<T,4,1> X_h;
    //X_h << vars, 1;
    //std::cerr << vars.transpose() << " energy = " << functor.energyFunc(X_h,Pvec,xvec,w) << std::endl;
    status = lm.minimizeOneStep(vars);
    ++steps;
  } while (status == Eigen::LevenbergMarquardtSpace::Running && steps < 50); 

  X = vars;
}
template void sfm::triangulateNonlinear(
    Eigen::Matrix<double,3,1>&,
    const std_vector_Matrix3x4d&,
    const std_vector_Vector2d&,
    const std::vector<double>);
template void sfm::triangulateNonlinear(
    Eigen::Matrix<float,3,1>&,
    const std_vector_Matrix3x4f&,
    const std_vector_Vector2f&,
    const std::vector<float>);




template <class T>
void sfm::triangulateOptimalCorrection(
      Eigen::Matrix<T,2,1>& x1_corrected,
      Eigen::Matrix<T,2,1>& x2_corrected,
      const Eigen::Matrix<T,3,3>& F,
      const Eigen::Matrix<T,2,1>& x1,
      const Eigen::Matrix<T,2,1>& x2 )
{
  // this is the optimal correction method from the paper:
  // "Triangulation from Two Views Revisited:Hartley-Sturm vs. Optimal Correction"
  // by Kenichi Kanatani, Yasuyuki Sugaya and Hirotaka Niitsuma
  Eigen::Matrix<T,3,3> Pk( Eigen::Matrix<T,3,3>::Identity() );
  Pk(2,2) = 0;

  // These are the corrected positions for x1 and x2
  Eigen::Matrix<T,3,1> _x1(x1(0),x1(1),1), _x2(x2(0),x2(1),1);

  T E = std::numeric_limits<T>::max();
  T err = std::numeric_limits<T>::max();
  int iterations = 0;
    
  Eigen::Matrix<T,2,1> dx1(0,0), dx2(0,0);  

  while( iterations < 1000 && err > 1e-6 )
  {
    T numerator;
    numerator = _x1.dot(F*_x2) + dx1.dot((F*_x2).topRows(2)) + dx2.dot((F.transpose()*_x1).topRows(2));
    T denominator;
    denominator = (F*_x2).dot( Pk*F*_x2 ) + (F.transpose()*_x1).dot(Pk*F.transpose()*_x1);

    T quotient = numerator / denominator;
    dx1 = quotient * (F * _x2).topRows(2);
    dx2 = quotient * (F.transpose() * _x1).topRows(2);

    T E_tmp = dx1.squaredNorm() + dx2.squaredNorm();
    
    err = std::abs(E - E_tmp);
    E = E_tmp;
    _x1.topRows(2) = x1 - dx1;
    _x2.topRows(2) = x2 - dx2;

    ++iterations;
  }
  //std::cerr << iterations << std::endl;
  x1_corrected = _x1.topRows(2);
  x2_corrected = _x2.topRows(2);
}
template void sfm::triangulateOptimalCorrection(
      Eigen::Vector2d& x1_corrected,
      Eigen::Vector2d&,
      const Eigen::Matrix3d&,
      const Eigen::Vector2d&,
      const Eigen::Vector2d& );
template void sfm::triangulateOptimalCorrection(
      Eigen::Vector2f& x1_corrected,
      Eigen::Vector2f&,
      const Eigen::Matrix3f&,
      const Eigen::Vector2f&,
      const Eigen::Vector2f& );



namespace
{
  // How to calculate a Model (--> How to triangulate a 3D point; Parameters we sample are cams and their observation)
  struct triangulateRANSAC_ModelFn
  {
    typedef Eigen::Matrix<double,3,1,Eigen::DontAlign> ModelType;
    enum Parameters {MINIMUM_SAMPLES=2};

    triangulateRANSAC_ModelFn( const sfm::std_vector_Matrix3x4d& P,
                               const sfm::std_vector_Vector2d& x )
      :P(P), x(x)
    {
      P_min.resize(MINIMUM_SAMPLES);
      x_min.resize(MINIMUM_SAMPLES);
    }
    
    // Calculate a model from given selection (triangulate a point from chosen subset of cams/observations)
    bool operator()(std::vector<ModelType>& x3d_vec, const int* idx)
    {
      x3d_vec.clear();
      ModelType x3d;

      for( int i = 0; i < MINIMUM_SAMPLES; ++i )
      {
        P_min[i] = P[idx[i]];
        x_min[i] = x[idx[i]];
      }

      x3d = sfm::triangulateLinear(P_min, x_min);

      x3d_vec.push_back(x3d);
      
      return true;
    }
    
    // Calculate final estimation from given inlier set
    bool operator()(ModelType& X, const std::vector<bool>& inliers)
    {
      // build inlier set
      int inliers_count = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
        if( inliers[i] )
          ++inliers_count;

      sfm::std_vector_Matrix3x4d P_inliers(inliers_count);
      sfm::std_vector_Vector2d x_inliers(inliers_count);
      int index = 0;
      for( int i = 0; i < (int)inliers.size(); ++i )
      {
        if( inliers[i] )
        {
          P_inliers[index] = P[i];
          x_inliers[index] = x[i];
          ++index;
        }
      }

      sfm::Vector3d tmp = sfm::triangulateLinear(P_inliers, x_inliers);
      sfm::triangulateNonlinear(tmp, P_inliers, x_inliers);
      X = tmp;
      return true;
    }
    
    // full data for the problem
    const sfm::std_vector_Matrix3x4d& P;
    const sfm::std_vector_Vector2d& x;
    
    // minimal subset 
    sfm::std_vector_Matrix3x4d P_min;
    sfm::std_vector_Vector2d x_min;
  };
    
  // How to score a given model (evaluate model on whole data available)
  struct triangulateRANSAC_DistanceFn
  {
    typedef Eigen::Matrix<double,3,1,Eigen::DontAlign> ModelType;

    triangulateRANSAC_DistanceFn( const sfm::std_vector_Matrix3x4d& P,
                                  const sfm::std_vector_Vector2d& x )
      :P(P), x(x)
    { }

    // Evaluate how well the model (a 3d point) is doing for a certain camera and observation
    double operator()(const ModelType& X, int idx)
    {
      // calculate reprojection error
      sfm::Vector3d tmp = X;
      sfm::Vector2d x_proj = sfm::projectPoint(P[idx], tmp);
      x_proj = x_proj - x[idx];
      double cost = std::sqrt(x_proj[0]*x_proj[0] + x_proj[1]*x_proj[1]); // L2 error
      return cost;
    }

    const sfm::std_vector_Matrix3x4d& P;
    const sfm::std_vector_Vector2d& x;
  };

} // namespace

void sfm::triangulateRANSAC(
    Eigen::Vector3d& X,
    Eigen::VectorXi& inliers,
    const std_vector_Matrix3x4d& P,
    const std_vector_Vector2d& x,
    int samples,
    double threshold )
{
  if( P.size() != x.size() )
    throw std::runtime_error("computeTriangulationRANSAC: P and x point count mismatch\n");
  if( P.size() < 2 )
    throw std::runtime_error("computeTriangulationRANSAC: Not enough samples\n");

  triangulateRANSAC_ModelFn modelFn(P, x);
  triangulateRANSAC_DistanceFn distanceFn(P, x);

  typedef sfm::internal::RANSAC<double, triangulateRANSAC_ModelFn, triangulateRANSAC_DistanceFn> RANSACType;
    
  RANSACType ransac(modelFn, distanceFn, P.size(), threshold);
  typename RANSACType::ModelType X_tmp;
  X_tmp.setZero(); // get rid of unitialized value warning
  ransac.run(X_tmp,inliers,samples);

  X = X_tmp;
}



