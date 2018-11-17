#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include<complex>
#include "Dense"
#include "Sparse"

using namespace Eigen;

SparseVector<double> SGS( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , double eps , int n_ite_max );

SparseVector<double> res_min( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , SparseVector<double> x0, double eps , int n_ite_max );

SparseVector<double> grad_conj( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , SparseVector<double> x0, double eps , int n_ite_max );

SparseMatrix<double> create_mat(const std::string name_file_read, bool sym);

void saveSolution(int it , int n_iter , double residu);
