//
// Created by alireza on 21/09/2020.
//

#include "../includes/unc_solver.h"
#include "eigen3/Eigen/IterativeLinearSolvers"


Scalar line_search(ObjType &obj_func, GradType &grad, Vec &step, Vec &x){
    Scalar t = 1, alpha = 1e-4, beta = 0.5;
    Scalar phi = obj_func(x + t *step), phi_prime_0 = grad(x).dot(step), phi0 = obj_func(x);

    int iter = 0; const int max_iter = 500;

    while ((phi > phi0 + alpha * t * phi_prime_0) && iter < max_iter){
        ++iter;
        t *= beta;
        phi = obj_func(x + t * step);
        if (iter == max_iter - 1) cout << "line search was not successful." << endl;
    }
    return t;

}


Vec conjugate_gradient(const Mat &A, const Vec &b, Vec x) {

    x.setZero();
    Vec r = b; Vec w;
    Scalar _rho = r.norm();
    Scalar rho = _rho * _rho, rho_old = rho, rho_old_ = 1;
    Scalar alpha; const Scalar b_norm = b.norm();
    Vec p;
    const int max_iter = 200;
    const Scalar eps = 1e-2;
    int iter = 1;
    while ((sqrt(rho) > eps * b_norm) && iter <= max_iter){

        if (iter > 1) rho_old_ = rho_old;
        rho_old = rho;

        if (iter == 1){
            p = r;
        }else{
            p = r + (rho_old/rho_old_) * p;
        }
        w = A * p;
        alpha = rho_old / (p.dot(w));
        x += alpha * p;
        r -= alpha * w;
        _rho = r.norm();
        rho = _rho * _rho;
        ++iter;

    }
    return x;
}

Vec truncated_newton(ObjType &obj_func, GradType &grad, HessType &hess, Vec x) {

    const int n = x.size();
    const int max_iter = 100;
    const Scalar eps = 1e-2;
    Scalar t = 1;
    Vec step(n,1);
    Scalar f, err = 10;
    Mat H(n, n);
    x.setZero();
    Vec g(n,1);
    int iter = 0;

    while ((err > eps)){
        ++iter;
//        cout << iter << " " <<err << " " << t <<endl;
//        f = obj_func(x);
        g = grad(x);
        H = hess(x);

        Eigen::ConjugateGradient<Mat> cg(H);
//        step = conjugate_gradient(H, -g, x);
         step = cg.solve(-g);
        t = line_search(obj_func, grad, step, x);
        x += t * step;

        err = g.norm();

    }
//    cout << err << endl;
    return x;
}

