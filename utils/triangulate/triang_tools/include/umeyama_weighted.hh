#ifndef UMEYAMA_WEIGHTED_H_
#define UMEYAMA_WEIGHTED_H_

#include <Eigen/Geometry>

namespace sfm
{
  using namespace Eigen;

  template <typename Derived, typename OtherDerived, typename T>
  typename internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type
  umeyama_weighted(const MatrixBase<Derived>& src, const MatrixBase<OtherDerived>& dst, const std::vector<T>& weights, bool with_scaling = true)
  {
    typedef typename internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type TransformationMatrixType;
    typedef typename internal::traits<TransformationMatrixType>::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename Derived::Index Index;

    EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL)
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename internal::traits<OtherDerived>::Scalar>::value),
      YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

    enum { Dimension = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, OtherDerived::RowsAtCompileTime) };

    typedef Matrix<Scalar, Dimension, 1> VectorType;
    typedef Matrix<Scalar, Dimension, Dimension> MatrixType;
    typedef typename internal::plain_matrix_type_row_major<Derived>::type RowMajorMatrixType;

    const Index m = src.rows(); // dimension
    const Index n = src.cols(); // number of measurements

    // required for demeaning ...
    //const RealScalar one_over_n = 1 / static_cast<RealScalar>(n);


    // computation of mean
    VectorType tmp1 = VectorType::Zero();
    VectorType tmp2 = VectorType::Zero();
    RealScalar weights_sum = 0;
    for( int i = 0; i < n; ++i )
    {
      tmp1 += src.col(i) * weights[i];
      tmp2 += dst.col(i) * weights[i];
      weights_sum += weights[i];
    }
    const RealScalar weights_sum_inv = 1/weights_sum;
    //const VectorType src_mean = src.rowwise().sum() * one_over_n;
    //const VectorType dst_mean = dst.rowwise().sum() * one_over_n;
    const VectorType src_mean = tmp1 * weights_sum_inv;
    const VectorType dst_mean = tmp2 * weights_sum_inv;

    // demeaning of src and dst points
    const RowMajorMatrixType src_demean = src.colwise() - src_mean;
    const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

    // Eq. (36)-(37)
    Scalar tmp3 = 0;
    for( int i = 0; i < n; ++i )
    {
      tmp3 += src_demean.col(i).squaredNorm()* weights[i];
    }
    //const Scalar src_var = src_demean.rowwise().squaredNorm().sum() * one_over_n;
    const Scalar src_var = tmp3 * weights_sum_inv;

    // Eq. (38)
    MatrixType tmp4 = MatrixType::Zero();
    for( int i = 0; i < n; ++i )
    {
      tmp4 += dst_demean.col(i)* src_demean.col(i).transpose() * weights[i];
    }
    //const MatrixType sigma = one_over_n * dst_demean * src_demean.transpose();
    const MatrixType sigma = tmp4 * weights_sum_inv;

    JacobiSVD<MatrixType> svd(sigma, ComputeFullU | ComputeFullV);

    // Initialize the resulting transformation with an identity matrix...
    TransformationMatrixType Rt = TransformationMatrixType::Identity(m+1,m+1);

    // Eq. (39)
    VectorType S = VectorType::Ones(m);
    if (sigma.determinant()<0) S(m-1) = -1;

    // Eq. (40) and (43)
    const VectorType& d = svd.singularValues();
    Index rank = 0; for (Index i=0; i<m; ++i) if (!internal::isMuchSmallerThan(d.coeff(i),d.coeff(0))) ++rank;
    if (rank == m-1) {
      if ( svd.matrixU().determinant() * svd.matrixV().determinant() > 0 ) {
        Rt.block(0,0,m,m).noalias() = svd.matrixU()*svd.matrixV().transpose();
      } else {
        const Scalar s = S(m-1); S(m-1) = -1;
        Rt.block(0,0,m,m).noalias() = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
        S(m-1) = s;
      }
    } else {
      Rt.block(0,0,m,m).noalias() = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
    }

    // Eq. (42)
    const Scalar c = 1/src_var * svd.singularValues().dot(S);

    // Eq. (41)
    // Note that we first assign dst_mean to the destination so that there no need
    // for a temporary.
    Rt.col(m).head(m) = dst_mean;
    Rt.col(m).head(m).noalias() -= c*Rt.topLeftCorner(m,m)*src_mean;

    if (with_scaling) Rt.block(0,0,m,m) *= c;

    return Rt;
  }


}

#endif /* UMEYAMA_WEIGHTED_H_ */
