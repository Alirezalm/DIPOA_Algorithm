//
// Created by Alireza on 10/2/20.
//

#include "../includes/DCCP.h"
#include "../includes/cut_generation.h"
#include "../includes/master_milp.h"
#include <fstream>

DCCP::DCCP(ObjType &obj, GradType &grad, HessType &hess, int &N, int &kappa, Scalar &M, Scalar &lambda) {

    this->obj = obj;
    this->grad = grad;
    this->hess = hess;
    this->N = N;
    this->kappa = kappa;
    this->M = M;
    this->lambda = lambda;
}

Results DCCP::dipoa(Vec &delta, int &rank, bool display) {
    double eps = 1e-5;
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
    double max_time = 600;
    // for initialization of the storage pool
    vector<Scalar> obj_val_storage(max_iter * max_nodes);
    vector<Scalar> eig_val_storage(max_iter * max_nodes);
    vector<Vec> grad_val_storage(max_iter * max_nodes);
    vector<Vec> x_val_storage(max_iter * max_nodes);
    CutStorage StoragePool(obj_val_storage, grad_val_storage, x_val_storage, eig_val_storage); // storage pool class
    Results res;
    SolverData solver_status;
    vector<int> v(max_iter);
    EventGen event(v);
    double elapse_time;
    double master_time;

    double lb = -1e20;
    double ub = 1e20;
    double _ub_temp;
    int event_counter = 0;
    double err = 10;
    auto start_ccp = std::chrono::high_resolution_clock::now();
    std :: ofstream dipoa_data("solverdata.csv");
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
            MPI_Send(&local_min_eig, 1, MPI_DOUBLE, 0, 600, MPI_COMM_WORLD);
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
            delta = master_milp_gurobi(StoragePool, max_nodes, M, kappa, i, lb, NumCut, lambda, master_time, event);
            event.old_gap = err;
            err = (ub - lb) / (ub + 1e-8);
            event.current_gap = err;
            if (event.is_generated()){
                ++event_counter;
                event.event_storage[event_counter] = i;
                cout << "Total SOC : " << event_counter << endl;
                cout << "SOC added at iter : " << i + 1 << endl;

	    }

            // saving and printing the status
            solver_status.iter = i;
            solver_status.abs_err = ub - lb;
            solver_status.rel_err = (ub - lb)/(ub + 1e-8);
            solver_status.num_cut = NumCut;
            solver_status.nlp_time = duration.count();
            solver_status.mip_time = master_time;
            dipoa_data <<i <<", " <<  ub << ", "<< lb << endl;
            if (display){
                solver_status.print_status();
            }


        }

        MPI_Bcast(delta.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        auto end_ccp_iter = std::chrono::high_resolution_clock::now();
        auto duration_ccp_iter = std::chrono::duration_cast<std::chrono::seconds>(end_ccp_iter - start_ccp);
        if (rank == 0) cout << "Elapsed time: " << duration_ccp_iter.count() << endl;
      if (err <= eps) {
        //if(duration_ccp_iter.count() >= max_time){
            if (rank == 0) std::cerr << "dipoa terminated successfully" << endl;
//            x = rhadmm(obj, grad, hess, x, rank, M, delta, max_iter_rhadmm, false);
            res.setXOpt(x);
            res.setMaxIter(i);
            res.setExitFlag(1);
            res.setObjOpt(lb);

            break;
        }
    }
    auto end_ccp = std::chrono::high_resolution_clock::now();
    auto duration_ccp = std::chrono::duration_cast<std::chrono::milliseconds>(end_ccp - start_ccp);
    res.setMaxTime(duration_ccp.count());
    return res;
}

Vec DCCP::sfp(Vec &x, int &rank) {
    const int n = x.size();
    Vec delta(n, 1);
    delta.setZero();
    int max_iter = 0;
    bool sfp = true;
    cout << "SFP is running ..." << endl;

    x = rhadmm(obj, grad, hess, x, rank, M, delta, max_iter, sfp);
    cout << "SFP converged " << endl;
    Vec warm_delta(n, 1);
    warm_delta.setZero();
    Scalar _temp;


    x = x.array().abs();

    for (int i = 0; i < kappa; ++i) {

        _temp = x.maxCoeff();
        for (int j = 0; j < n; ++j) {

            if (x[j] == _temp) {

                warm_delta[j] = 1;
                x[j] = -1;
                break;
            }

        }

    }
   if(rank==0) cout << "WarmDelta " << warm_delta.sum() << endl;
    MPI_Barrier(MPI_COMM_WORLD);

    return warm_delta;
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


