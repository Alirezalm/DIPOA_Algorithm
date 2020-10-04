//
// Created by Alireza on 10/2/20.
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
    int NumCut = 0;
    const int max_iter = 100;
    vector<Scalar> obj_val_storage(max_iter * max_nodes);
    vector<Vec> grad_val_storage(max_iter * max_nodes);
    vector<Vec> x_val_storage(max_iter * max_nodes);

    CutStorage StoragePool(obj_val_storage, grad_val_storage, x_val_storage);

    double lb = -1e5;
    double ub = 1e5; double _ub_temp;
    for (int i = 0; i < max_iter; ++i) {

//        cout << "NodeId: " << rank << " delta "<<delta << endl;
        x = rhadmm(obj, grad, hess, x, rank, M, delta);
        local_obj = obj(x);
        local_grad = grad(x);
//        cout << "NodeId: " << rank << " x "<< x << endl;


       if (rank != 0) {
           MPI_Send(&local_obj, 1, MPI_DOUBLE, 0, 50, MPI_COMM_WORLD);
           MPI_Send(local_grad.data(), n, MPI_DOUBLE, 0, 60, MPI_COMM_WORLD);
       }

       MPI_Barrier(MPI_COMM_WORLD);
       if (rank == 0) {
           StoragePool.add_cut_f(local_obj, i * N);
           StoragePool.add_cut_grad(local_grad, i* N);
           StoragePool.add_cut_x(x, i* N);

           Scalar f; Vec nabla_f(n,1);
           _ub_temp = local_obj;
           for (int j = 1; j < max_nodes; ++j) {
               MPI_Recv(&f, 1, MPI_DOUBLE, j, 50, MPI_COMM_WORLD, &status);
               MPI_Recv(nabla_f.data(), n, MPI_DOUBLE, j, 60, MPI_COMM_WORLD, &status);
               nabla_f.resize(n,1);
               StoragePool.add_cut_f(f, (i * N) + j);
               StoragePool.add_cut_grad(nabla_f, (i * N) + j);
               StoragePool.add_cut_x(x, (i * N) + j);
               _ub_temp += f;
           }
            ub = std :: min(ub, _ub_temp);
            delta = master_milp(StoragePool, max_nodes, M, kappa, i + 1, lb, NumCut);



            cout <<"Iter "<< i <<" lb " << lb << " ub " << ub << " error " << ub - lb << " TotalNumCut " << NumCut <<
            " CutPerNode " << NumCut/N << " TotalStorageSize " << ((i + 1) * (N + 2 * (N * n)) * 8) *1e-6 << " mb" << endl;
       }
        MPI_Bcast(delta.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
}