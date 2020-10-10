//
// Created by alireza on 21/09/2020.
//

#ifndef DIPOA_RESULTS_H
#define DIPOA_RESULTS_H
#include "external_libs.h"

class Results {

public:

    Results();
    void print();

    const Vec &getXOpt() const;

    void setXOpt(const Vec &xOpt);

    Scalar getObjOpt() const;

    void setObjOpt(Scalar objOpt);

    double getMaxTime() const;

    void setMaxTime(double maxTime);

    double getMaxIter() const;

    void setMaxIter(double maxIter);

    int getExitFlag() const;

    void setExitFlag(int exitFlag);

private:
    Vec x_opt;
    Scalar obj_opt;
    double max_time;
    double max_iter;
    int exit_flag;
};

class SolverData{

public:
    SolverData();
    void print_status();
    int iter;
    double lb;
    double ub;
    double rel_err;
    double abs_err;
    int num_cut;
    double storage_size;
    int prime_iter;
    double mip_time;
    double nlp_time;

};
#endif //DIPOA_RESULTS_H
