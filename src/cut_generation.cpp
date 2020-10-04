//
// Created by alireza on 10/2/20.
//

#include "../includes/cut_generation.h"


CutStorage::CutStorage(vector<Scalar> &obj_initial_storage, vector<Vec> &grad_initial_storage,
                       vector<Vec> &x_initial_storage) {
    this->obj_value_storage = obj_initial_storage;
    this->grad_storage = grad_initial_storage;
    this->x_storage = x_initial_storage;
}

void CutStorage::add_cut_f(Scalar local_f, int current_iter) {

    obj_value_storage[current_iter] = local_f;
}

void CutStorage::add_cut_grad(Vec &local_grad, int current_iter) {

    grad_storage[current_iter] = local_grad;

}

void CutStorage::add_cut_x(Vec &local_x, int current_iter) {

    x_storage[current_iter] = local_x;

}


