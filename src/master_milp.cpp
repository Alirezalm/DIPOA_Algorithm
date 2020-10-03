//
// Created by alireza on 10/2/20.
//

#include "../includes/master_milp.h"



Vec master_milp(CutStorage &StoragePool, int &N, Scalar &M, int &kappa, int &current_iter) {
    int n = StoragePool.x_storage[0].size();
    vector<IloNumVarArray> X(N);
    vector<IloNumVarArray> Delta(N);
    IloEnv env;

    IloModel model(env);
    IloNumVarArray alpha(env, N);

    for (int i = 0; i < N; ++i) {
        IloNumVarArray x(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
        X[i] = x;
    }
    IloNumVarArray z(env, n, -IloInfinity, IloInfinity, ILOFLOAT);
    IloNumVarArray delta(env, n, 0.0, 1.0, ILOBOOL);
    IloObjective obj;
    IloExpr expr(env);

    for (int i = 0; i < N; ++i) {
        expr += alpha[i];
    }

    obj = IloMinimize(env, expr);

    model.add(obj);


    IloExpr dot;
    for (int j = 0; j < current_iter * N; ++j) {
        dot = dot_prod(StoragePool.grad_storage[j], X[j], StoragePool.x_storage[j], n);
        model.add(alpha[j] >= StoragePool.obj_value_storage[j] + dot);
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < n; ++i) {
            model.add(X[j][i] == z[i]);
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

    model.add(sum_delta == kappa);


    IloCplex cplex(model);
    cplex.solve();
    Vec _delta(n,1);
    for (int i = 0; i < n; ++i) {
        _delta[i] = cplex.getValue(delta[i]);
    }
    env.end();

    cout << _delta << endl;
    return _delta;
}

IloExpr dot_prod(Vec &grad, IloNumVarArray &x, Vec &x_k, int &n) {
    IloExpr  sum;
    for (int i = 0; i < n; ++i) {
        sum += grad[i] * (x[i] - x_k[i]);
    }
    return sum;
}
