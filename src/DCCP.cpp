//
// Created by alireza on 10/2/20.
//
#include "../includes/DCCP.h"
#include "../includes/cut_generation.h"
#include "../includes/master_milp.h"

DCCP::DCCP( ObjType &obj,  GradType &grad,  HessType &hess,  int &N,  int &kappa,  Scalar &M){
    this->obj = obj;
    this->grad = grad;
    this->hess = hess;
    this->N = N;
    this->kappa = kappa;
    this->M = M;

}

Vec DCCP::solve(Vec &delta, int& rank) {

    int max_nodes;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &max_nodes);
    const int n = delta.size();
    Vec x(n,1); x.setZero();
    Scalar local_obj;
    Vec local_grad;
    Mat local_hess;

    const int max_iter = 1;
    vector<Scalar> obj_val_storage(max_iter * max_nodes);
    vector<Vec> grad_val_storage(max_iter * max_nodes);
    vector<Vec> x_val_storage(max_iter * max_nodes);

    CutStorage StoragePool(obj_val_storage, grad_val_storage, x_val_storage);

    for (int i = 0; i < max_iter; ++i) {

        x = rhadmm(obj, grad, hess, x, rank, M, delta);
        local_obj = obj(x);
        local_grad = grad(x);
        cout << "NodeID " << rank << " f: " << local_obj<< endl;
       if (rank != 0) {
           MPI_Send(&local_obj, 1, MPI_DOUBLE, 0, 50, MPI_COMM_WORLD);
           MPI_Send(local_grad.data(), n, MPI_DOUBLE, 0, 60, MPI_COMM_WORLD);
       }
       MPI_Barrier(MPI_COMM_WORLD);

       if (rank == 0) {
           StoragePool.add_cut_f(local_obj, i);
//           StoragePool.add_cut_grad(local_grad, i);
//           StoragePool.add_cut_x(x, i);
           Scalar f; Vec nabla_f(n,1);
           for (int j = 1; j < max_nodes; ++j) {
               MPI_Recv(&f, 1, MPI_DOUBLE, j, 50, MPI_COMM_WORLD, &status);
               MPI_Recv(nabla_f.data(), n, MPI_DOUBLE, j, 60, MPI_COMM_WORLD, &status);
               nabla_f.resize(n,1);

               StoragePool.add_cut_f(f, i + j);
//               StoragePool.add_cut_grad(nabla_f, i + j);
//               StoragePool.add_cut_x(x, i + j);
           }
           for (int j = 0; j < N ; ++j) {
               cout << "NodeID: " << rank << " " << StoragePool.obj_value_storage[j] << endl; //testing storage system
           }
//            master_milp(StoragePool, N, M, kappa, i);

       }
    }


    return x;
}

int DCCP::getN() const {
    return N;
}

Scalar DCCP::getKappa() const {
    return kappa;
}

Scalar DCCP::getM() const {
    return M;
};


