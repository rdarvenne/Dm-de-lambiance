#ifndef _Meth_Iterative_H

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <complex>
#include "Dense"
#include "Sparse"


class MethIterative
{
  protected:
    Eigen::VectorXd _x;  // solution itérée
    Eigen::VectorXd _r;  // vecteur résidu
    Eigen::VectorXd _b;  // vecteur du second membre
    Eigen::VectorXd _p;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _A; // matrice du systeme

  public:
    // constructeur à partir d'une matrice A dense donnée
    MethIterative();
    // destructeur
    ~MethIterative();
    // initialise une Matrice
    void MatrixInitialize(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A);
    // initialise les données
    void Initialize(Eigen::VectorXd x0, Eigen::VectorXd b);
    // initialise la matrice A venant d'un fichier
    //SparseMatrix<double> MatrixInitialize(const std::string name_file_read, bool sym);
    // exécute une itération
    virtual void Advance(Eigen::VectorXd z) = 0;
    // récupère le vecteur xk et le vecteur rk
    const Eigen::VectorXd & GetIterateSolution() const;
    const Eigen::VectorXd & GetResidu() const;
    const Eigen::VectorXd & Getp() const;
};

class ResiduMin : public MethIterative
{
  public:
    void Advance(Eigen::VectorXd z);
};

class GradientConj : public MethIterative
{
  public:
    void Advance(Eigen::VectorXd z);
};

class SGS : public MethIterative
{
  public:
    void Advance(Eigen::VectorXd z);
};

#define _Meth_Iterative_H
#endif
