//
// Created by alireza on 21/09/2020.
//

#ifndef DIPOA_EXTERNAL_LIBS_H
#define DIPOA_EXTERNAL_LIBS_H

#include "iostream"
#include "cmath"
#include "eigen3/Eigen/Dense"
#include <functional>
#include "mpi/mpi.h"
#include <cassert>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using Scalar = typename Eigen::MatrixXd::Scalar ;
using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>;
using Ind = typename Eigen::MatrixXd::Index ;

using ObjType = std :: function<Scalar(Vec)>;
using GradType = std :: function<Vec(Vec)>;
using HessType = std :: function<Mat(Vec)>;


#endif //DIPOA_EXTERNAL_LIBS_H
