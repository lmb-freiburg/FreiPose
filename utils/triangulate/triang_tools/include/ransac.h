#ifndef RANSAC_H_
#define RANSAC_H_

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <Eigen/Core>

namespace sfm {
namespace internal {


template <class Real, 
          class MODEL_FN, 
          class DISTANCE_FN
         >
struct RANSAC
{
  typedef typename MODEL_FN::ModelType ModelType;

  /*!
   *  \param modelFn     The function that is used to compute the model
   *  \param distanceFn  The function to compute the distance to the model.
   *                     The distance is compared to the threshold to define
   *                     the inliers.
   *  \param size        The size of the data e.g. number of input points or
   *                     number of correspondences
   *  \param threshold   The threshold
   */
  RANSAC(MODEL_FN& modelFn, DISTANCE_FN& distanceFn, int size, Real threshold)
    :modelFn(modelFn),distanceFn(distanceFn),size(size),threshold(threshold)
  { }


  void randomNoDuplicates(int* r,boost::random::uniform_int_distribution<>& dist)
  {
    // remove duplicates
    for( int i = 0; i < MODEL_FN::MINIMUM_SAMPLES; ++i )
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


  void run(ModelType& model, Eigen::VectorXi& inliers, int iterations)
  {
    std::vector<ModelType> M_vec;
    int nInliers_best = -1; 
    Real distanceSum;
    Real dist_best = std::numeric_limits<Real>::infinity();
    std::vector<bool>* inliers_best = new std::vector<bool>(size,false);
    std::vector<bool>* inliers_tmp = new std::vector<bool>(size,false);
    
    boost::random::uniform_int_distribution<> dist(0,size-1);

    for( int i = 0; i < iterations; ++i )
    {
      // draw random samples
      int idx[MODEL_FN::MINIMUM_SAMPLES];
      randomNoDuplicates(idx, dist);

      bool success = modelFn(M_vec,idx);
      if( !success )
        continue;

      for( int j = 0; j < (int)M_vec.size(); ++j )
      {
        ModelType& M = M_vec[j];

        // determine inliers
        int nInliers = 0;
        distanceSum = 0;
        for( int k = 0; k < size; ++k )
        {
          Real dist = distanceFn(M,k);
          if( dist < threshold )
          {
            (*inliers_tmp)[k] = true;
            ++nInliers;
          }
          else
            (*inliers_tmp)[k] = false;
        }
        
        if( nInliers > MODEL_FN::MINIMUM_SAMPLES )
        {
          // compute again M but this time using the inliers
          bool success = modelFn(M,*inliers_tmp);
          if( !success )
            continue;

          // determine the inliers again to get the consensus set for the current model
          nInliers = 0;
          for( int k = 0; k < size; ++k )
          {
            Real dist = distanceFn(M,k);
            if( dist < threshold )
            {
              (*inliers_tmp)[k] = true;
              ++nInliers;
              distanceSum += dist;
            }
            else
              (*inliers_tmp)[k] = false;
          }
        }
        distanceSum /= nInliers; // average cost of the solutions inliers
        if(( nInliers > nInliers_best ) || 
            ( (nInliers == nInliers_best) && (distanceSum < dist_best)) ) // Use average cost as tie breaker
        {
          model = M;
          nInliers_best = nInliers;
          dist_best = distanceSum;
          std::swap( inliers_tmp, inliers_best );
        }
      }

    }
    inliers.resize(size,1);
    for( int i = 0; i < size; ++i )
      inliers(i) = (*inliers_best)[i] ? 1 : 0;

    delete inliers_tmp;
    delete inliers_best;
  }

  MODEL_FN& modelFn;
  DISTANCE_FN& distanceFn;
  int size;
  Real threshold;

  boost::random::mt19937 rng;
};

} // namespace internal
} // sfm

#endif /* RANSAC_H_ */
