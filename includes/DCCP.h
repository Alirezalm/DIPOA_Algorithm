//
// Created by alireza on 10/2/20.
//

#ifndef DCCP_DCCP_H
#define DCCP_DCCP_H

#include "external_libs.h"
#include "results.h"
#include "rhadmm.h"

class DCCP {
/*
 * Distributed Cardinality Constrained Programming (DCCP) problem class.
 */
public:
    DCCP(ObjType &obj, GradType &grad, HessType &hess, int &N, int &kappa, Scalar &M,
         Scalar &lambda); // default constructor

    Results dipoa(Vec &delta, int &rank, bool display); //invoking DIPOA

    Vec sfp(Vec &x, int &rank);

    int getN() const;

    Scalar getKappa() const;

    Scalar getM() const;

    Scalar getLambda() const;

    void setLambda(Scalar lambda);
private:
    int N;
    int kappa;
    Scalar lambda;
    Scalar M;
    ObjType obj;
    GradType grad;
    HessType hess;


};


#endif
