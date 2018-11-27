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


void MethIterative::MatrixInitialize(SparseMatrix<double> A)
{
  _A.resize(A.rows(), A.cols());
  _A = A;
}

// initialisation des données
void GradientConj::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;                   // utile pour le GradientConj

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}

void ResiduMin::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_res_min.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}

void ResiduMin::Advance(VectorXd z)
{
  double alpha(0.);
  alpha = _r.dot(z)/z.dot(z);

  _x += alpha*_r;
  _r += - alpha*z;
}

const VectorXd & MethIterative::GetIterateSolution() const
{
  return _x;
}

const VectorXd & MethIterative::GetResidu() const
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

  _U.resize(_A.rows(), _A.cols()), _L.resize(_A.rows(),_A.cols());
  _U.setZero();
  _L.setZero();
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        _L.coeffRef(i,j) = _A.coeffRef(i,j);
      else if (j>i)
        _U.coeffRef(i,j) = _A.coeffRef(i,j);
      else
      {
        _L.coeffRef(i,j) = 1.;
        _U.coeffRef(i,j) = _A.coeffRef(i,j);
      }
    }
  }
  VectorXd y_bis(_A.cols());

  y_bis = GetSolTriangInf(_L, b);
  _y = GetSolTriangSup(_U, y_bis);
  cout << y_bis << endl;
  cout << "----------------" << endl;
  cout << _U*_y << endl;

}

void SGS::Advance(VectorXd z)
{
  VectorXd  w(_x.size()), w_bis(_x.size());

  // SparseLU< SparseMatrix<double> > lu1;
  // lu1.analyzePattern(_M) ;
  // lu1.factorize(_M);
  // y = lu1.solve(_b);

  w_bis = GetSolTriangInf(_L, z);
  w = GetSolTriangSup(_U, w_bis);


  _x += - w + _y;
  _r = _b - _A*_x;

}


// Écrit un fichier avec la solution au format Paraview
void MethIterative::saveSolution(int N ,string name_file , int n_iter , double residu)
{
  ofstream mon_flux; // Contruit un objet "ofstream"
  // name_file = ("/sol_"+to_string(N)+"_"+name_file+".txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);

  if(mon_flux)
  {
      mon_flux<<n_iter<<residu<<endl;
  }
  mon_flux.close();
}

void Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = v.size();

  vector< SparseVector<double> > _Vm ;
  SparseMatrix<double> _Hm;

  SparseVector<double> v1 , s1 ;
  s1.resize(v.size());
  v1 = v.sparseView();

  _Hm.resize( m+1 , m );
  _Hm.setZero();

  vector< SparseVector<double> > z;
  z.resize(v.size());
  _Vm.resize(m+1);
  _Vm[0]= v1/v1.norm();

  for (int j=0 ; j<m ; j++)
    {SparseVector<double> Av = A*_Vm[j];
      s1.setZero();
        for(int i=0 ; i<j ; i++)
        {
          _Hm.coeffRef(i,j) = Av.dot(_Vm[i]);
          s1 +=  _Hm.coeffRef(i,j)*_Vm[i];
        }
      z[j] = Av - s1;
      _Hm.coeffRef(j+1,j) = z[j].norm();

    if(_Hm.coeffRef(j+1,j) == 0.)
    {   break;}
    _Vm[j+1] = z[j]/_Hm.coeffRef(j+1,j);
    }
  //  cout<<"Hm = "<<_Hm <<endl;
}
/*
const vector< SparseVector<double> > & GetVm() const
{
  return _Vm;
}

const SparseMatrix<double> & GetHm() const
{
  return _Hm;
}
*/
/*
void Gmres::Advance(VectorXd z)
{
  SparseQR< SparseMatrix<double> > QR;
  QR.factorize(_Hm);
  QR.matrixQ()

}
*/
VectorXd GetSolTriangSup(SparseMatrix<double> U, VectorXd b)
{
  VectorXd solution(U.rows());

  for (int i=0; i<U.rows(); i++)
  {
    solution[U.rows()-i-1] = b[U.rows()-i-1];
    for (int j=U.rows()-i; j<U.rows(); j++)
    {
      solution[U.rows()-1-i] = solution[U.rows()-1-i] - U.coeffRef(U.rows()-1-i,j)*solution[j];
    }
    solution[U.rows()-1-i] = solution[U.rows()-1-i]/U.coeffRef(U.rows()-1-i,U.rows()-1-i);
  }
  return solution;
}

VectorXd GetSolTriangInf(SparseMatrix<double> L, VectorXd b)
{
  VectorXd solution(L.rows());

  for (int i=0; i<L.rows(); i++)
  {
    solution[i] = b[i];
    for (int j=0; j<i; j++)
    {
      solution[i] = solution[i] - L.coeffRef(i,j)*solution(j);
    }
    solution[i] = solution[i]/L.coeffRef(i,i);
  }
  return solution;
}

#define _Meth_Iterative_H
#endif
