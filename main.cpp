/**
 * Distributed logistic regression test file for the DIPOA algorithm.
 * @author Alireza Olama
 * e-mail: alireza.lm69@gmail.com
 * @mainpage https://github.com/Alirezalm
 * @
 */
#include "includes/DCCP.h"
#include "test_functions/log_reg_random.h"

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    srand(clock());
    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &total); // get MPI size
    if (rank == 0) cout << "MPI IMPLEMENTATION OF ADMM ALGORITHM FOR " << total << " NODES" << endl;

    int m , n;
    if (rank == 0){
        cout << "number of rows? ";
        std::cin >> m;
        cout << "number of cols? ";
        std::cin >> n;
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "dataset info: " << endl;
        cout << "total number of rows and columns " << total * m << " x " << n << endl;
        cout << "total number of rows and columns per node: " << m << " x " << n << endl;
        cout << "total size: " << n * total * m * 8 * 1e-6 << " mb" << endl;
        cout << "total size per node: " << n * m * 8 * 1e-6 << " mb" << endl;
    }
    // Random data generation
    Mat X = Mat::Random(m, n);
    for (int i = 0; i < n; ++i) {
        X.col(i) = X.col(i) / X.col(i).norm();
    }
    Vec theta = Vec::Random(n, 1);
    Vec y = Vec::Random(m, 1);
    for (int i = 0; i < m; ++i) {
        if (y[i] <= 0) {
            y[i] = 1.0;
        } else {
            y[i] = 0.0;
        }
    }

    // problem parameters and functions
    Scalar M = 0.9;
    int kappa;
    if (rank == 0){
        cout << "How many non_zeros? ";
        std::cin >> kappa;
    }
    MPI_Bcast(&kappa, 1, MPI_INT, 0, MPI_COMM_WORLD);
    Scalar lambda = 0.5; //regularization parameter
    if (rank == 0){
        cout << "regularization param? ";
        std::cin >> lambda;
    }
    MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int sfp_flag;
    if (rank == 0){
        cout << "SFP? (0 or 1) ";
        std::cin >> sfp_flag;
    }
    MPI_Bcast(&sfp_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    
    ObjType obj_func = log_reg_obj(X, y, m, lambda); //logistic objective function
    GradType grad_func = log_reg_grad(X, y, lambda); // logistic gradient
    HessType hess_func = log_reg_hess(X, lambda); // logistic hessian

    Vec delta(n, 1); //initially feasible binary variables for OA algorithm
    delta.setZero();
    int N = total;
    // creating DCCP problem
    DCCP Problem(obj_func, grad_func, hess_func, N, kappa, M, lambda);
   if(sfp_flag){
    delta = Problem.sfp(theta, rank);
   }
    Results res = Problem.dipoa(delta, rank, true); // solving the problem

    if (rank == 0) res.print();
//    if (rank == 0) cout << hess_func(theta).eigenvalues().real().minCoeff() << endl;
    MPI_Finalize();
    return 0;
}
