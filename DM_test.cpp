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


///////////////////// CLASSE MERE ///////////////////////
void MethIterative::MatrixInitialize(SparseMatrix<double> A)
{
  _A.resize(A.rows(), A.cols());
  _A = A;
}

const VectorXd & MethIterative::GetIterateSolution() const
{
  return _x;
}

const VectorXd & MethIterative::GetResidu() const
{
  return _r;
}

const VectorXd & MethIterative::Getp() const
{
  return _p;
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


///////////////////// Gradient Conjugué ///////////////////////
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



///////////////////// Residu Minimum ///////////////////////
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



///////////////////// Gauss Seidel Symétrique ///////////////////////
void SGS::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;

  SparseMatrix<double> U, L, D;
  U.resize(_A.rows(), _A.cols()), L.resize(_A.rows(),_A.cols()), D.resize(_A.rows(),_A.cols());
  _U.resize(_A.rows(), _A.cols()), _L.resize(_A.rows(), _A.cols());
  U = _A;
  L = _A;
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        {U.coeffRef(i,j) = 0.;}
      else if (j>i)
        {L.coeffRef(i,j) = 0.;}
      else
      {D.coeffRef(i,i) = 1./_A.coeffRef(i,i);}
    }
  }
  _L = L*D;
  _U = U;

  VectorXd y_bis(_x.size());
  y_bis = GetSolTriangInf(_L, b);
  _y = GetSolTriangSup(_U, y_bis);

}

void SGS::Advance(VectorXd z)
{
  VectorXd  w(_x.size()), w_bis(_x.size());

  w_bis = GetSolTriangInf(_L, z);
  w = GetSolTriangSup(_U, w_bis);

  _x += - w + _y;
  _r = _b - _A*_x;

}


///////////////////// GMRes ///////////////////////
void Gmres::Initialize(VectorXd x0, VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;
  _beta = _r.norm();                   // utile pour le GradientConj

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
}


void Gmres::Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = v.size();

  _Vm.resize(m, m+1);

  vector< SparseVector<double> > Vm ;
  SparseVector<double> v1 , s1 ;
  s1.resize(v.size());
  v1 = v.sparseView();

  _Hm.resize( m+1 , m );
  _Hm.setZero();

  vector< SparseVector<double> > z;
  z.resize(v.size());
  Vm.resize(m+1);
  Vm[0]= v1/v1.norm();

  for (int j=0 ; j<m ; j++)
    {SparseVector<double> Av = A*Vm[j];
      s1.setZero();
        for(int i=0 ; i<j ; i++)
        {

          _Hm.coeffRef(i,j) = Av.dot(Vm[i]);
          s1 +=  _Hm.coeffRef(i,j)*Vm[i];

        }
      z[j] = Av - s1;
      _Hm.coeffRef(j+1,j) = z[j].norm();
    if(_Hm.coeffRef(j+1,j) == 0.)
    {   break;}
    Vm[j+1] = z[j]/_Hm.coeffRef(j+1,j);
  }

  for(int i=0; i<m; i++)
  {
    for (int j=0; j<m+1; j++)
    {_Vm.coeffRef(i,j) = Vm[j].coeffRef(i);}
  }

//
// for (int i=0; i<Vm.size(); i++)
// {
//     cout<<"Vm[i] = "<< Vm[i] <<endl;
//   }
//     cout<<"_Vm = "<< _Vm <<endl;
}


void Gmres::Givens(SparseMatrix<double> Hm)
{

// J'ai mis R et Q de taille carré...
  _Rm = Hm;
  //cout << Hm << endl;
  _Qm.resize(Hm.rows(), Hm.rows());
  //cout << "QM " << _Qm.rows() << _Qm.cols() << endl;
  _Qm.setIdentity();
  double c(0.), s(0.), u(0.), v(0.);
  SparseMatrix<double> Rotation_transposee(Hm.rows(), Hm.rows());

  for (int i=0; i<Hm.rows()-1; i++)
  {
    Rotation_transposee.setIdentity();
    c = _Rm.coeffRef(i,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    s = -_Rm.coeffRef(i+1,i)/sqrt(_Rm.coeffRef(i,i)*_Rm.coeffRef(i,i) + _Rm.coeffRef(i+1,i)*_Rm.coeffRef(i+1,i));
    Rotation_transposee.coeffRef(i,i) = c;
    Rotation_transposee.coeffRef(i+1,i+1) = c;
    Rotation_transposee.coeffRef(i+1,i) = -s;
    Rotation_transposee.coeffRef(i,i+1) = s;

    for (int j=i; j<Hm.cols(); j++)
    {
      u = _Rm.coeffRef(i,j);
      v = _Rm.coeffRef(i+1,j);
      _Rm.coeffRef(i,j) = c*u - s*v;
      _Rm.coeffRef(i+1,j) = s*u + c*v;
      //if (j == i)
      //{cout << "ici " << _Rm.coeffRef(i+1,j) << endl;}
    }

    _Qm = _Qm*Rotation_transposee;
  }
    // cout << "QR = "<< _Qm*_Rm << endl;
    // cout << "Hm = "<< _Hm << endl;
    // cout << "_vm = "<< _Vm << endl;
}


const SparseMatrix<double> & Gmres::GetVm() const
{
  return _Vm;
}

const SparseMatrix<double> & Gmres::GetHm() const
{
  return _Hm;
}

void Gmres::Advance(VectorXd z)
{
  VectorXd gm_barre(_Qm.rows()), gm(z.size()), y(z.size()), vect(_Qm.rows());
  SparseMatrix<double> Rm_pas_barre(z.size(), z.size());
  SparseMatrix<double> Vm;
  Vm.resize(z.size(), z.size());

  gm.setZero();
  gm_barre.setZero();
  vect.setZero();

  for (int i=0; i<_Qm.rows(); i++)
  {
    gm_barre[i] += _Qm.coeffRef(0,i);
    vect[i] = _Qm.coeffRef(i,z.size());
  }

  gm_barre = gm_barre*z.norm();
  for (int i=0; i<z.size(); i++)
  {
    gm[i] = gm_barre[i];
  }
  for (int i=0; i<z.size(); i++)
  {
    for (int j=0; j<z.size(); j++)
    {
      Rm_pas_barre.coeffRef(i,j) = _Rm.coeffRef(i,j);
      Vm.coeffRef(i,j) = _Vm.coeffRef(i,j);
    }
  }

  y = GetSolTriangSup(Rm_pas_barre, gm);
  // cout << "Qm = " << _Qm <<endl;
  // cout << "vect = " << vect <<endl;
  _x = _x + Vm*y;

  _r = gm_barre[z.size()]*_Vm*vect;
  _beta = abs(gm_barre[z.size()]);
    //cout << " r = " << _r << endl;
    cout << "gm+1" << gm_barre[_r.size()] << endl;
    cout <<"norme de r" << _r.norm() << endl;
  // cout << "juste après l'affectation de r dans advance" << endl;
  //
  // cout << "_vm = "<< _Vm << endl;
}
const double & Gmres::GetNorm() const
{
  return _beta;
}

/////////////////////Gradient conjugué préconditionné question 4d
void GradientConPrecond::Initialize(Eigen::VectorXd x0, Eigen::VectorXd b)
{
  _x = x0;
  _b = b;
  _r = _b - _A*_x;
  _p = _r;                   // utile pour le GradientConj

  ofstream mon_flux; // Contruit un objet "ofstream"
  string name_file = ("/sol_"+to_string(_x.size())+"_grad_conj_precond.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file,ios::out);
  
  _D.resize(_x.size(),_x.size()), _D_inv.resize(_x.size(),_x.size()), _E.resize(_x.size(),_x.size()), _F.resize(_x.size(),_x.size());
  _D.setZero(); _F.setZero(); _E.setZero();
  for (int i =0; i<_x.size(); i++)
  {
    _D.coeffRef(i,i) = _A.coeffRef(i,i);
    _D_inv.coeffRef(i,i) = 1./_A.coeffRef(i,i);

    for(int j = 0; j<_x.size(); j++)
    {
      if (j>i)
      {_F.coeffRef(i,j) = - _A.coeffRef(i,j);}

      else if (j<i)
      {_E.coeffRef(i,j) = - _A.coeffRef(i,j);}
    }

  }

}



void GradientConPrecond::Advance(Eigen::VectorXd z)
{
  double alpha , gamma, stock_r;

  stock_r = _r.dot(_r);

  alpha   =  _r.dot(_r)/(z.dot(_p));
  _x +=  alpha*_p ;
  _r += - alpha*z ;
  gamma = _r.dot(_r)/stock_r;
  _p = _r + gamma*_p;
}

///////////////////// Fonctions hors classe ///////////////////////
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
