//
// Created by alireza on 23/09/2020.
//

#ifndef DIPOA_RHADMM_H
#define DIPOA_RHADMM_H
#include "external_libs.h"
#include "unc_solver.h"

ObjType x_step_obj (ObjType &f, Vec &y, Vec &z, Scalar &rho);
GradType x_step_grad (GradType &grad, Vec &y, Vec &z, Scalar &rho);
HessType x_step_hess (HessType &hess, Scalar &rho);

Vec rhadmm(ObjType &obj, GradType &grad, HessType &hess, Vec &x, int &rank, Scalar &M, Vec delta,
           int &max_iter_num, bool sfp);



#endif //DIPOA_RHADMM_H
