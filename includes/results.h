//
// Created by alireza on 21/09/2020.
//

#ifndef DIPOA_RESULTS_H
#define DIPOA_RESULTS_H
#include "external_libs.h"

class Results {
public:
    Results(Vec &x, Scalar &f);

    const Vec &getX() const;

    Scalar getF() const;

private:
    Vec x;
    Scalar f;
};


#endif //DIPOA_RESULTS_H
