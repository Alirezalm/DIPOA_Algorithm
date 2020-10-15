//
// Created by alireza on 21/09/2020.
//

#include "../includes/results.h"

Results::Results() {

    x_opt.setZero();
    obj_opt = 0;
    max_time = 0;
    max_iter = 0;
    exit_flag = -1;
}

const Vec &Results::getXOpt() const {
    return x_opt;
}

void Results::setXOpt(const Vec &xOpt) {
    x_opt = xOpt;
}

Scalar Results::getObjOpt() const {
    return obj_opt;
}

void Results::setObjOpt(Scalar objOpt) {
    obj_opt = objOpt;
}

double Results::getMaxTime() const {
    return max_time;
}

void Results::setMaxTime(double maxTime) {
    max_time = maxTime;
}

double Results::getMaxIter() const {
    return max_iter;
}

void Results::setMaxIter(double maxIter) {
    max_iter = maxIter;
}

int Results::getExitFlag() const {
    return exit_flag;
}

void Results::setExitFlag(int exitFlag) {
    exit_flag = exitFlag;
}

void Results::print() {
    cout << " "<< endl;
    cout << "DIPOA generated the following results:"<< endl;
    if (exit_flag == 1){
        cout << "status: optimal"<< endl;
    } else{
        cout << "status: sub-optimal"<< endl;
    }
    cout << "optimal objective function: " << obj_opt << endl;
    cout << "maximum number iterations: " << max_iter << endl;
    if (max_time < 1e3) {
        cout << "maximum time: " << max_time << " ms" << endl;
    }else{
        cout << "maximum time: " << max_time * 1e-3<< " seconds" << endl;
    }

}

SolverData::SolverData() {
    iter = 0;
    lb = -1e20;
    ub = 1e20;
    rel_err = (ub - lb)/(ub + 1e-8);
    abs_err = ub - lb;
    num_cut = 0;
    storage_size = 0;
    prime_iter = 0;
    mip_time = 0;
    nlp_time = 0;
}

void SolverData::print_status() {
// print the status
    std::cerr.precision(5);
    std::cerr.fill();
    std :: cerr.setf(std::ios::scientific);
    std::cerr.setf(std::ios::showpos);
    if (iter == 0){
        cout << " " << endl;
        cout << "Solver Status " << endl;
    }
    std :: cerr<<"iter: " << iter << " abs-err: " << abs_err << " rel-err: " << rel_err << " num-cuts: " <<
    num_cut << " nlp-time: "<< nlp_time << " ms"<< " mip-time: " << mip_time << " ms" <<endl;
}

EventGen::EventGen(vector<int> &event_storage) {
    old_gap = 1e10;
    current_gap = 1e20;
    threshold = 1e-2;
    this -> event_storage  = event_storage;
}

bool EventGen::is_generated() const {
    if ((old_gap - current_gap)/old_gap <= threshold){
        return true;
    }else{
        return false;
    }

}

