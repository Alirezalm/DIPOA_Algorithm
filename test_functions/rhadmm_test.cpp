//
// Created by alireza on 10/2/20.
//

#include "../includes/rhadmm.h"
#include "log_reg_random.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    srand(clock());
    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    if (rank == 0) cout << "MPI IMPLEMENTATION OF ADMM ALGORITHM FOR " << total << " NODES" << endl;
    int m = 5000, n = 50;

    Mat X = Mat::Random(m, n);
    Vec theta = Vec::Random(n, 1);
    Vec y = Vec::Random(m, 1);

    for (int i = 0; i < m; ++i) {
        if (y[i] <= 0) {
            y[i] = 1.0;
        } else {
            y[i] = 0.0;
        }
    }
    Scalar lambda = 1e-4;
    ObjType obj_func = log_reg_obj(X, y, m, lambda);
    GradType grad_func = log_reg_grad(X, y, lambda);
    HessType hess_func = log_reg_hess(X, lambda);
    Vec init (n,1);
    init.setRandom();
    Scalar M = 0.01;
    Vec delta (n,1); delta.setOnes();
    Vec z = rhadmm(obj_func, grad_func, hess_func, init, rank, M, delta);
    MPI_Barrier(MPI_COMM_WORLD);
    cout << "NodeID: "<< rank <<" The final answer is: \n" << z.norm() << endl;
    MPI_Finalize();
    return 0;
}
