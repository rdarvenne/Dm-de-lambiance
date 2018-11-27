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
  cout<<"bite dedans"<<endl;

  _r = _b - _A*_x;

    cout<<"cul dedans"<<endl;
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

  SparseMatrix<double> U, L, D;
  U.resize(_A.rows(), _A.cols()), L.resize(_A.rows(),_A.cols()), D.resize(_A.rows(),_A.cols());
  U = _A;
  L = _A;
  for (int i=0; i<_A.rows(); i++)
  {
    for (int j=0; j<_A.cols(); j++)
    {
      if (i>j)
        U.coeffRef(i,j) = 0.;
      else if (j>i)
        L.coeffRef(i,j) = 0.;

      else
        D.coeffRef(i,i) = 1./_A.coeffRef(i,i);
    }
  }
  _M = L*D*U;

  SparseLU< SparseMatrix<double>> lu1;

  lu1.analyzePattern(_M) ;
  lu1.factorize(_M);
  _y = lu1.solve(_b);

}

void SGS::Advance(VectorXd z)
{
  VectorXd  w(_x.size());

  // SparseLU< SparseMatrix<double> > lu1;
  // lu1.analyzePattern(_M) ;
  // lu1.factorize(_M);
  // y = lu1.solve(_b);

  SparseLU< SparseMatrix<double> > lu2;
  lu2.analyzePattern(_M) ;
  lu2.factorize(_M);
  w = lu2.solve(z);

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

SparseMatrix<double> MethIterative::create_mat(const string name_file_read, bool sym)
 {
   int N(0.);
  // Eigen::SparseMatrix<double> A ;
   ifstream mon_flux(name_file_read);

   string ligne, colonne, valeur;
   getline(mon_flux,ligne); //lit la première ligne qui ne nous intéresse pas

   mon_flux >> N; //lit le premier mot de la ligne 2 correspond au nombre de lignes

   int nonzero_elem;
   mon_flux >> colonne; //lit le nombre de colonnes (valeur stockée inutilement, je ne savais pas comment lire sans stocker...)
   mon_flux >> nonzero_elem; //lit le nombre d'élements non nuls

   // Définition de la la matrice A.
   _A.resize(N,N);
   vector<Triplet<double>> liste_elem;
   for (int i = 0; i < nonzero_elem; i++)
   {
     mon_flux >> ligne;
     mon_flux >> colonne;
     mon_flux >> valeur;

     int li = atoi(ligne.c_str());
     int col = atoi(colonne.c_str());
     double val = atof(valeur.c_str());

     liste_elem.push_back({li-1,col-1,val});  //atoi pour passer de string à int et atof idem avec double
     if ((colonne != ligne) && sym) // dans le cas d'une matrice symétrique seulement la moitié des éléments sont dans le fichier texte
     {
       liste_elem.push_back({col-1,li-1,val});
     }
   }
   _A.setFromTriplets(liste_elem.begin(),liste_elem.end());
   mon_flux.close();

   return _A;
}




void Arnoldi( SparseMatrix<double> A , VectorXd v)
{
  //dimension de l'espace
  int m = v.size();

  std::vector< Eigen::SparseVector<double> > _Vm ;
  Eigen::SparseMatrix<double> _Hm;

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
        for(int i=0 ; i<=j ; i++)
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

#define _Meth_Iterative_H
#endif
