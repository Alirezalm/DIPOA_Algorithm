//
// Created by alireza on 21/09/2020.
//

#include "../includes/results.h"

Results::Results(Vec &x, Scalar &f):x(x), f(f) {}

const Vec &Results::getX() const {
    return x;
}

Scalar Results::getF() const {
    return f;
}
