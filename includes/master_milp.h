//
// Created by alireza on 10/2/20.
//

#ifndef DIPOA_MASTER_MILP_H
#define DIPOA_MASTER_MILP_H
#include "external_libs.h"
#include "../includes/cut_generation.h"

#include "results.h"

Vec  master_milp(CutStorage &StoragePool,int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut,
                 Scalar &lambda, double &elapsed_time, EventGen &event);
Vec  master_milp_gurobi(CutStorage &StoragePool,int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut,
                 Scalar &lambda, double &elapsed_time, EventGen &event);
//IloExpr dot_prod(Vec &grad, IloNumVarArray &x, Vec &x_k, IloEnv env, int &n);

GRBLinExpr dot_prod(Vec &grad, GRBVar *x, Vec &x_k, int &n);

#endif //DIPOA_MASTER_MILP_H
