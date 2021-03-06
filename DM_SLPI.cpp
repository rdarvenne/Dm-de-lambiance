#include "DM_SLPI.h"  // Inclure le fichier .h
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include<complex>
#include "Dense"
#include "Sparse"
using namespace std;
using namespace Eigen;


VectorXd SGS( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , double eps , int n_ite_max )
{
  VectorXd y;
  VectorXd diffxy;
  double s1(0.) , s2(0.) ;
  y.resize(x.size());
  diffxy.resize(x.size());

  for( int i = 0 ; i<x.size() ; i++)
      {
        y[i] = x[i];
        x[i] = y[i] + 2*eps;
        diffxy[i] = x[i] - y[i];
      }
  int n_ite(0.);

  while(diffxy.lpNorm<Infinity>() > eps && n_ite < n_ite_max)
    {
        for( int i = 0 ; i<x.size() ; ++i)
          {x[i] = y[i];}
        for( int i = 0 ; i<x.size() ; ++i)
          {
            s1=0.;
            for(int j =0 ; j< i ; ++j )
              {s1 += A(i,j)*y[j] ; }

            for(int j = i+1 ; j< x.size() ; ++j)
              {s1 += A(i,j)*x[j]; }

            y[i] = (b[i] - s1)/A(i,i);
           }

           n_ite = n_ite + 1 ;

           for( int i = 0 ; i<x.size() ; ++i)
           {diffxy[i] = x[i] - y[i];}
    }
    if(n_ite > n_ite_max)
      {cout << "Tolérance non atteinte"<<endl;}
  //   cout <<"n_ite = "<<n_ite<<endl;
    return x;
}


VectorXd res_min( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , VectorXd x0, double eps , int n_ite_max )
{
  VectorXd r , z ;
  r.resize(b.size());
  double alpha(0.);

  r = b - A*x0;
  int n_ite(0.);

  while(r.lpNorm<Infinity>() > eps && n_ite < n_ite_max)
  {
    z = A*r;
    alpha = r.dot(z)/z.dot(z);
    x = x + alpha*r;
    r = r - alpha*z;
    n_ite++;
  }
  return x;
}

VectorXd grad_conj( Matrix<double,Dynamic,Dynamic> A , VectorXd b , VectorXd x , VectorXd x0, double eps , int n_ite_max )
  {
    VectorXd p0 ,pold , r0 , pnew  , rold ,rnew ,xnew ,xold;
    int n_ite ;
    double alpha , beta ;

    p0.resize(x.size());
    r0.resize(x.size());
    pnew.resize(x.size());
    rold.resize(x.size());
    rold.resize(x.size());
    rnew.resize(x.size());
    xold.resize(x.size());
    xnew.resize(x.size());

    r0 = A*x0 - b;
    p0 = -r0 ;
    pold = p0 ;
    rold = r0 ;
    rnew = r0;
    n_ite = 0;

    while (rnew.norm() > eps && n_ite <= n_ite_max)
     {
       alpha   =  -pold.dot(rold)/(A*pold).dot(pold) ;
       xnew = xold + alpha*pold ;
       rnew = rold + alpha * A*pold;
       beta = rnew.dot(rnew)/rold.dot(rold);
       pnew = -rnew + beta*pold;
       xold = xnew;
       rold = rnew;
       pold = pnew;
       n_ite++;
     }
      if(n_ite > n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}
    //   cout <<"n_ite = "<<n_ite<<endl;
      return xnew;
  }


 SparseMatrix<double> create_mat(const string name_file_read, bool sym)
  {
    int N(0.);
    Eigen::SparseMatrix<double> A ;
    ifstream mon_flux(name_file_read);

    string ligne, colonne, valeur;
    getline(mon_flux,ligne); //lit la première ligne qui ne nous intéresse pas 

    mon_flux >> N; //lit le premier mot de la ligne 2 correspond au nombre de lignes

    int nonzero_elem;
    mon_flux >> colonne; //lit le nombre de colonnes (valeur stockée inutilement, je ne savais pas comment lire sans stocker...)
    mon_flux >> nonzero_elem; //lit le nombre d'élements non nuls

    // Définition de la la matrice A.
    A.resize(N,N);
    vector<Triplet<double>> liste_elem;
    for (int i = 0; i < nonzero_elem; i++)
    {
      mon_flux >> ligne;
      mon_flux >> colonne;
      mon_flux >> valeur;

      int li = atoi(ligne.c_str());
      int col = atoi(colonne.c_str());
      double val = atof(valeur.c_str());

      liste_elem.push_back({li-1,col-1,val});  //stoi pour passer de string à int et stod idem avec double
      if ((colonne != ligne) && sym) // dans le cas de cette matrice symétrique seulement la moitié des éléments sont référencés dans le fichier texte
      {
        liste_elem.push_back({col-1,li-1,val});
      }
    }
    A.setFromTriplets(liste_elem.begin(),liste_elem.end());
    mon_flux.close();

    return A;
}
