//
// Created by alireza on 23/09/2020.
//

#include "../includes/rhadmm.h"

ObjType x_step_obj(ObjType &f, Vec &y, Vec &z, Scalar &rho) {

    return [&f, &y, &z, &rho](Vec x) -> Scalar {
        return f(x) + y.dot(x - z) + 0.5 * rho * (x - z).squaredNorm();
    };
}

GradType x_step_grad(GradType &grad, Vec &y, Vec &z, Scalar &rho) {

    return [&grad, &y, &z, &rho](Vec x) -> Vec {
        return grad(x) + y + rho * (x - z);
    };
}

HessType x_step_hess(HessType &hess, Scalar &rho) {

    return [&hess, &rho](Vec x) -> Mat {
        const int n = x.size();
        Mat _temp(n, n);
        _temp.setIdentity();
        return hess(x) + rho * _temp;
    };
}


Vec rhadmm(ObjType &obj, GradType &grad, HessType &hess, Vec &x, int &rank, Scalar &M, Vec delta) {

    const int max_iter = 500; // Maximum number of iterations
    const int n = x.size();
    Vec y(n, 1), z(n, 1), z_old(n, 1);
    x.setRandom();
    y.setZero();
    z.setZero();
    z_old.setZero();
    Scalar rho = 10;
    int exit_flag = 0;
    Vec x_rcv(n, 1), y_rcv(n, 1), z_rcv(n, 1);
    int max_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &max_nodes);
    MPI_Status status;
    ObjType f_x;
    GradType g_x;
    HessType Hess_x;
    Scalar eps = 1e-2;
    Scalar r = 1, s = 1; Scalar t = 1;
    Vec sum(n,1); sum.setZero();
    for (int i = 0; i < max_iter; ++i) {
        f_x = x_step_obj(obj, y, z, rho);
        g_x = x_step_grad(grad, y, z, rho);
        Hess_x = x_step_hess(hess, rho);
        x = truncated_newton(f_x, g_x, Hess_x, x);

        z_old = z;
//        MPI_Barrier(MPI_COMM_WORLD);
        sum = x + (1/rho) * y;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(sum.data(), z_rcv.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0){
            z_rcv.resize(n, 1);
            z = (1.0 / max_nodes) * z_rcv;
            z = (M * delta).cwiseMin((-M * delta).cwiseMax(z));
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(z.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        y +=  rho * (x - z);

        r = (x - z).squaredNorm();
        s = rho * (z_old - z).squaredNorm();
        MPI_Reduce(&r, &t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


//        if (rank == 0) cout<< "iter: " << i << " r: " << r << " s: " << s <<endl;
        if ((t <= eps) && (rho * sqrt(max_nodes) * s <= eps)){

            break;

        }


    }

    MPI_Barrier(MPI_COMM_WORLD);
    return x;
}

