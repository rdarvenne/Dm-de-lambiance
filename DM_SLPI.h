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


VectorXd SGS( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , double eps , int n_ite_max );

VectorXd res_min( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , VectorXd x0, double eps , int n_ite_max );

VectorXd grad_conj( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , VectorXd x0, double eps , int n_ite_max );

SparseMatrix<double> create_mat(const std::string name_file_read, bool sym);
