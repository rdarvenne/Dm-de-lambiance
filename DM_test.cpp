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
  _A.resize(A.rows(), A.cols());
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

void SGS::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  
  Matrix<double, Dynamic, Dynamic> U, L, D;
  U.resize(_A.rows(), _A.cols()), L.resize(_A.rows(),_A.cols()), D.resize(_A.rows(),_A.cols());
  U = _A;
  L = _A;
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        U(i,j) = 0.;
      else if (j>i)
        L(i,j) = 0.;

      else
        D(i,i) = 1./_A(i,i);
    }
  }
  _M = L*D*U;
}

void SGS::Advance(VectorXd z)
{
  VectorXd y(_x.size()), w(_x.size());

  FullPivLU< Matrix<double, Dynamic, Dynamic> > lu1;
  lu1.compute(_M);
  y = lu1.solve(_b);

  FullPivLU< Matrix<double, Dynamic, Dynamic> > lu2;
  lu2.compute(_M);
  w = lu2.solve(z);

  _x += - w + y;
  _r = _b - _A*_x;

}


#define _Meth_Iterative_H
#endif
