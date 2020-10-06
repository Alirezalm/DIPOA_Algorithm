//
// Created by alireza on 21/09/2020.
//

#ifndef DIPOA_UNC_SOLVER_H
#define DIPOA_UNC_SOLVER_H

#include "external_libs.h"
#include "external_libs.h"

Vec conjugate_gradient(const Mat &A, const Vec &b, Vec x);

Vec truncated_newton(ObjType &obj_func, GradType &grad, HessType &hess, Vec x);


#endif //DIPOA_UNC_SOLVER_H
