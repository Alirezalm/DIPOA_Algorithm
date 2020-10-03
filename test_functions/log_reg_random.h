//
// Created by alireza on 21/09/2020.
//

#ifndef DIPOA_LOG_REG_RANDOM_H
#define DIPOA_LOG_REG_RANDOM_H
#include "../includes/external_libs.h"

ObjType log_reg_obj(const Mat &X, const Vec &y, const int &m, const Scalar &lambda){

    return [&X, &y, &m, &lambda] (Vec theta) -> Scalar {

        Scalar sum = 0; Scalar h, z;
        for (int i = 0; i < m; ++i) {
            z = X.row(i).dot(theta);
            h = 1 / ( 1 + exp(-z));

            if (h == 1) h = 1 - 1e-8;
            if (h == 0) h = 1e-8;

            sum += -y[i] * log(h) - (1 - y[i]) * log(1 - h);
        }
        return sum + lambda * 0.5 * theta.squaredNorm();

    };
}

GradType log_reg_grad(const Mat &X, const Vec &y, const Scalar &lambda) {

    return [&X, &y, &lambda](Vec theta) -> Vec {

        Vec d = -X * theta;
        Vec h = 1 / (1 + d.array().exp());

        return X.transpose() * (h - y) + lambda * theta;
    };
}


HessType log_reg_hess(const Mat &X, const Scalar &lambda) {

    return [&X, &lambda](Vec theta) -> Mat {

        Vec d = -X * theta;
        Vec h = 1 / (1 + d.array().exp());
        Vec _h = (1 - h.array());
        Vec _temp = (h.array() * _h.array());
//        Mat M = _temp.asDiagonal();
        const int m = _temp.size();
        const int n = X.cols();
        Mat M(m,n);
        for (int i = 0; i < m; ++i) {

            M.row(i) = _temp[i] * X.row(i);
        }
        Mat Q(n,n); Q.setIdentity();
        return X.transpose() * M + lambda * Q;
    };
}
#endif //DIPOA_LOG_REG_RANDOM_H
