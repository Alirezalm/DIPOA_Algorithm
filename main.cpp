/**
 * Distributed logistic regression test file for the DIPOA algorithm.
 * @author Alireza Olama
 * e-mail: alireza.lm69@gmail.com
 * @mainpage https://github.com/Alirezalm
 * @
 */
#include "includes/DCCP.h"
#include "test_functions/log_reg_random.h"
#include <random>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
//    srand(clock());
    int rank, total;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get MPI rank
    MPI_Comm_size(MPI_COMM_WORLD, &total); // get MPI size
    if (rank == 0) {
	    cout << "MPI IMPLEMENTATION OF DIPOA ALGORITHM FOR " << total << " NODES" << endl;
	    cout << "DIPOA is applied to Distributed Cardinality Constrained Programming (DCCP) problems" << endl;
	    cout << "General form: " << endl;
	    cout << "min_x \sum_i^N f_i(x)" << endl;
	    cout << "s.t. card(x) <= kappa" << endl;
	    cout << " " << endl;
	    cout << "This program implements the Distributed Sparse Logistic Regression (DSLR) problem" << endl;
    }
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
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);
    Mat X (m,n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            X(i,j) =  distribution(generator);
        }
    }


    for (int i = 0; i < n; ++i) {
    X.col(i) = X.col(i) / X.col(i).norm();
   }
    Vec theta(n,1);
    Vec y (m,1);

    for (int i = 0; i < n; ++i) {
        theta[i] =  distribution(generator);

    }
    for (int i = 0; i < m; ++i) {
        y[i] =  distribution(generator);
    }
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
//   cout << X.col(1).mean() << endl;
    MPI_Finalize();
    return 0;
}
