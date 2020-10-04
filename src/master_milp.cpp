//
// Created by alireza on 10/2/20.
//

#include "../includes/master_milp.h"



Vec master_milp(CutStorage &StoragePool, int &N, Scalar &M, int &kappa, int current_iter, double &lb, int &NumCut) {

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


    IloExpr dot;
    int k = -1;
    for (int j = 0; j < current_iter * N; ++j) {
            k += 1;
            dot = dot_prod(StoragePool.grad_storage[j], X[k], StoragePool.x_storage[j], env, n);
            model.add(alpha[k] >= StoragePool.obj_value_storage[j] + dot);
            if (k == N - 1) k = -1;

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

    model.add(sum_delta <= kappa);
//
//

    IloCplex cplex(model);
    cplex.setParam(IloCplex::Param::MIP::Display , 0);
    cplex.setParam(IloCplex::Param::ParamDisplay, 0);
    NumCut = cplex.getNrows();
    cplex.solve();
    Vec _delta(n,1);
    for (int i = 0; i < n; ++i) {
        _delta[i] = abs(cplex.getValue(delta[i]));
    }


   lb = cplex.getObjValue();
    env.end();
    return _delta;
}

IloExpr dot_prod(Vec &grad, IloNumVarArray &x, Vec &x_k, IloEnv env,  int &n) {
    IloExpr  sum(env);
    for (int i = 0; i < n; ++i) {
        sum += grad[i] * (x[i] - x_k[i]);
    }
    return sum;
}
