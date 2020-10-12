//
// Created by alireza on 10/2/20.
//

#include "../includes/master_milp.h"

bool is_member(int key, vector<int> &v);
IloExpr quad_cut_expr(IloNumVarArray &x, Vec &x_k, IloEnv env, int &n) {
    IloExpr my_sum(env);
    for (int i = 0; i < n; ++i) {
        my_sum += IloPower((x[i] - x_k[i]), 2);
    }
    return my_sum;
}

Vec master_milp(CutStorage &StoragePool, int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut,
                Scalar &lambda, double &elapsed_time, EventGen &event) {

    int n = StoragePool.x_storage[0].size();
    IloEnv env;
    vector<IloNumVarArray> X(N);

    IloModel model(env);
    IloNumVarArray alpha(env, N, -1e5, 1e5, ILOFLOAT);

    for (int i = 0; i < N; ++i) {
        IloNumVarArray x(env, n, -1e5, 1e5, ILOFLOAT);
        X[i] = x;
    }
    IloNumVarArray z(env, n, -1e5, 1e5, ILOFLOAT);
    IloNumVarArray delta(env, n, 0.0, 1.0, ILOBOOL);
//
    IloObjective obj;
    IloExpr expr(env);

    for (int i = 0; i < N; ++i) {
        expr += alpha[i];
    }

    obj = IloMinimize(env, expr);

    model.add(obj);


    int soc_flag;

    int _iter_counter = 0;
    IloExpr foc(env);
    IloExpr soc(env);
    Scalar f_x;
    int k = -1;
    for (int j = 0; j < (current_iter + 1) * N; ++j) {
        k += 1;
        if(!event.event_storage.empty()){
            if ((_iter_counter != 0) && is_member(_iter_counter, event.event_storage)) {
                soc_flag = 1;
//                cout << "SOC is added at iter: " << _iter_counter << endl;
            } else{
                soc_flag = 0;
            }
        }else{
            soc_flag = 0;
        }

        f_x = StoragePool.obj_value_storage[j];
        foc = dot_prod(StoragePool.grad_storage[j], X[k], StoragePool.x_storage[j], env, n);
        soc = 0.5 * StoragePool.eig_storage[j] * quad_cut_expr(X[k], StoragePool.x_storage[j], env, n);
        model.add(alpha[k] >= f_x + foc + soc_flag * soc);
        if (k == N - 1) {
            k = -1;
            ++_iter_counter;
        }

    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < n; ++i) {
            model.add(IloAbs(X[j][i] - z[i]) == 0);
        }
    }

    for (int i = 0; i < n; ++i) {
        model.add(z[i] <= M * delta[i]);
        model.add(z[i] >= -M * delta[i]);
    }

    IloExpr sum_delta(env);
    for (int i = 0; i < n; ++i) {
        sum_delta += delta[i];
    }

    model.add(sum_delta <= kappa);
//
//

    IloCplex cplex(model);
//
    cplex.setParam(IloCplex::Param::MIP::Display, 0);
    cplex.setParam(IloCplex::Param::ParamDisplay, 0);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-5);

    NumCut = current_iter * N;
    auto start = std::chrono::high_resolution_clock::now(); // start measuring time
    cplex.solve();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    elapsed_time = duration.count();
    Vec _delta(n, 1);
    for (int i = 0; i < n; ++i) {
        _delta[i] = abs(cplex.getValue(delta[i]));
    }


    lb = cplex.getObjValue();
    env.end();
    return _delta;
}

IloExpr dot_prod(Vec &grad, IloNumVarArray &x, Vec &x_k, IloEnv env, int &n) {
    IloExpr sum(env);
    for (int i = 0; i < n; ++i) {
        sum += grad[i] * (x[i] - x_k[i]);
    }
    return sum;
}

bool is_member(int key, vector<int> &v){

    for (int i = 0; i < v.size(); ++i) {
        if (key == v[i]){

            return true;
        }
    }
    return false;
}