//
// Created by alireza on 10/2/20.
//

#ifndef DCCP_DCCP_H
#define DCCP_DCCP_H
#include "external_libs.h"
#include "rhadmm.h"

class  DCCP{

public:
    DCCP(ObjType &obj,  GradType &grad,  HessType &hess,  int &N,  int &kappa,  Scalar &M);

    Vec solve(Vec &delta,  int& rank);

    int getN() const;

    Scalar getKappa() const;

    Scalar getM() const;

private:
    int N;
    int kappa;
    Scalar M;
    ObjType obj;
    GradType grad;
    HessType hess;
};


#endif
