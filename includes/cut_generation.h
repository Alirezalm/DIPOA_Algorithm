//
// Created by alireza on 10/2/20.
//

#ifndef DIPOA_CUT_GENERATION_H
#define DIPOA_CUT_GENERATION_H
#include "external_libs.h"
#include "results.h"

class CutStorage{
public:
    CutStorage(vector<Scalar> &obj_initial_storage, vector<Vec> &grad_initial_storage, vector<Vec> &x_initial_storage, vector<Scalar> &eig_init_storage);
    void add_cut_f(Scalar local_f, int current_iter);
    void  add_cut_grad(Vec &local_grad, int current_iter);
    void add_cut_x(Vec &local_x, int current_iter);
    void add_cut_eig(Scalar &local_eig, int current_iter);

//    friend Vec  master_milp(CutStorage &StoragePool,int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut,
//                            Scalar &lambda, double &elapsed_time, EventGen &event);
    friend Vec  master_milp_gurobi(CutStorage &StoragePool,int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut,
                            Scalar &lambda, double &elapsed_time, EventGen &event);

private:
    vector<Scalar> obj_value_storage;
    vector<Vec> grad_storage;
    vector<Vec> x_storage;
    vector<Scalar> eig_storage;

};

#endif //DIPOA_CUT_GENERATION_H
