//
// Created by Alireza on 10/2/20.
//

#include "../includes/DCCP.h"
#include "../includes/cut_generation.h"
#include "../includes/master_milp.h"

DCCP::DCCP(ObjType &obj, GradType &grad, HessType &hess, int &N, int &kappa, Scalar &M, Scalar &lambda) {
    this->obj = obj;
    this->grad = grad;
    this->hess = hess;
    this->N = N;
    this->kappa = kappa;
    this->M = M;
    this->lambda = lambda;
}

Vec DCCP::dipoa(Vec &delta, int &rank) {
    double eps = 1e-4;
    int max_nodes;

    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &max_nodes);

    const int n = delta.size();
    Vec x(n, 1);
    x.setZero();

    Scalar local_obj;
    Vec local_grad;
    Scalar local_min_eig;
    Scalar f;
    Vec nabla_f(n, 1);
    Vec _x(n, 1);

    int NumCut = 0;
    const int max_iter = 300;
    int max_iter_rhadmm;
    // for initialization of the storage pool
    vector<Scalar> obj_val_storage(max_iter * max_nodes);
    vector<Scalar> eig_val_storage(max_iter * max_nodes);
    vector<Vec> grad_val_storage(max_iter * max_nodes);
    vector<Vec> x_val_storage(max_iter * max_nodes);
    CutStorage StoragePool(obj_val_storage, grad_val_storage, x_val_storage, eig_val_storage); // storage pool class

    double elapse_time;
    double master_time;

    double lb = -1e20;
    double ub = 1e20;
    double _ub_temp;

    double err;
    for (int i = 0; i < max_iter; ++i) {//main loop

        auto start = std::chrono::high_resolution_clock::now();
        x = rhadmm(obj, grad, hess, x, rank, M, delta, max_iter_rhadmm, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (rank == 0) elapse_time = duration.count();

        // computing local information
        local_obj = obj(x);
        local_grad = grad(x);
        local_min_eig = hess(x).eigenvalues().real().minCoeff();

        if (rank != 0) {

            MPI_Send(&local_obj, 1, MPI_DOUBLE, 0, 50, MPI_COMM_WORLD);
            MPI_Send(local_grad.data(), n, MPI_DOUBLE, 0, 60, MPI_COMM_WORLD);
            MPI_Send(&local_min_eig, 1 , MPI_DOUBLE, 0, 600, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            // storing data
            StoragePool.add_cut_f(local_obj, i * N);
            StoragePool.add_cut_eig(local_min_eig, i * N);
            StoragePool.add_cut_grad(local_grad, i * N);
            StoragePool.add_cut_x(x, i * N);
            _ub_temp = local_obj;

            for (int j = 1; j < max_nodes; ++j) { // storing data of other nodes

                MPI_Recv(&f, 1, MPI_DOUBLE, j, 50, MPI_COMM_WORLD, &status);
                MPI_Recv(nabla_f.data(), n, MPI_DOUBLE, j, 60, MPI_COMM_WORLD, &status);
                MPI_Recv(&local_min_eig, 1, MPI_DOUBLE, j, 600, MPI_COMM_WORLD, &status);

                nabla_f.resize(n, 1);
                StoragePool.add_cut_f(f, (i * N) + j);
                StoragePool.add_cut_eig(local_min_eig, (i * N) + j);
                StoragePool.add_cut_grad(nabla_f, (i * N) + j);
                StoragePool.add_cut_x(x, (i * N) + j);
                _ub_temp += f; // total primal objective function
            }
            ub = std::min(ub, _ub_temp); // updating the upper bound

            delta = master_milp(StoragePool, max_nodes, M, kappa, i + 1, lb, NumCut, lambda, master_time);
            err = (ub - lb)/(ub + 1e-8);
            // print the status
            std::cerr.precision(5);
            std::cerr.fill();
//            std :: cerr.setf(std::ios::scientific);
            std::cerr.setf(std::ios::showpos);
            std::cerr << "Iter " << i << " lb " << lb << " ub " << ub << " rel-err " << err << " abs-err " << ub - lb<< " NumCut "
                      << NumCut
                      << " StorageSize " << ((i + 1) * (N + 2 * (N * n)) * 8) * 1e-6 << " mb" <<
                      " PrimalIter " << max_iter_rhadmm << " MIPTime "
                      << master_time << "ms" << " NLPTime " << elapse_time << " ms" << endl;
        }

        MPI_Bcast(delta.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (err <= eps) {
            if(rank == 0) std :: cerr << "dipoa terminated successfully" << endl;
            break;
        }
    }
    x = rhadmm(obj, grad, hess, x, rank, M, delta, max_iter_rhadmm, false);
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

Scalar DCCP::getLambda() const {
    return lambda;
}

void DCCP::setLambda(Scalar lambda) {
    DCCP::lambda = lambda;
}

Vec DCCP::sfp(Vec &x, int &rank) {
    const int n = x.size();
    Vec delta(n,1); delta.setZero();
    int max_iter = 0;
    bool sfp = true;
    cout << "SFP is running ..." << endl;

    x = rhadmm(obj, grad, hess, x, rank, M, delta, max_iter, sfp);
    cout << "SFP converged " << endl;
    Vec warm_delta(n,1); warm_delta.setZero();
    Scalar _temp;


   x = x.array().abs();

   for (int i = 0; i < kappa; ++i) {

       _temp = x.maxCoeff();
       for (int j = 0; j < n ; ++j) {

           if (x[j] == _temp) {

               warm_delta[j] = 1;
               x[j] = -1;
               break;
           }

       }




   }
    MPI_Barrier(MPI_COMM_WORLD);
//   std :: cerr << " x:" << x << endl;
//   std :: cerr << " delta: "<<warm_delta << endl;

    return warm_delta;
}
