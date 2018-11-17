#ifndef _Meth_Iterative_H

#include "DM_test.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

// constructeur par défaut
MethIterative::MethIterative()
{}

// Destructeur
MethIterative::~MethIterative()
{}


void MethIterative::MatrixInitialize(Matrix<double, Dynamic, Dynamic> A)
{
  _A.resize(A.size(), A.size());
  _A = A;
}

// initialisation des données
void MethIterative::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;                   // utile pour le GradientConj
}

void ResiduMin::Advance(VectorXd z)
{
  double alpha(0.);
  alpha = _r.dot(z)/z.dot(z);

  _x += alpha*_r;
  _r += - alpha*z;
}

const Eigen::VectorXd & MethIterative::GetIterateSolution() const
{
  return _x;
}

const Eigen::VectorXd & MethIterative::GetResidu() const
{
  return _r;
}


void GradientConj::Advance(VectorXd z)
{
  double alpha , gamma, stock_r;

  stock_r = _r.dot(_r);

  alpha   =  _r.dot(_r)/(z.dot(_p));
  _x +=  alpha*_p ;
  _r += - alpha*z ;
  gamma = _r.dot(_r)/stock_r;
  _p = _r + gamma*_p;
}

const VectorXd & MethIterative::Getp() const
{
  return _p;
}

void SGS::Advance(VectorXd z)
{

}


#define _Meth_Iterative_H
#endif
