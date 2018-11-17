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

SparseVector<double> SGS( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , double eps , int n_ite_max )
{
  double s1(0.) ;
  int n = b.size();
  SparseVector<double> y(n) , diffxy(n) , r(n);
  
  for( int i = 0 ; i<x.size() ; i++)
      {
        y.coeffRef(i) = x.coeffRef(i);
        x.coeffRef(i) = y.coeffRef(i) + 2*eps;
        diffxy.coeffRef(i) = x.coeffRef(i) - y.coeffRef(i);
      }
  std::ofstream mon_flux; // Contruit un objet "ofstream"
  std::string name_file = ("sol_N"+to_string(n)+"_SGS.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file, ios::out);

  int n_ite(0.);
  r = b- A*x;

  if(mon_flux)
      {
        mon_flux<<n_ite<<" "<<r.norm()<<endl;
      }


  while(r.norm() > eps && n_ite < n_ite_max)
    {
        for( int i = 0 ; i<x.size() ; ++i)
          {x.coeffRef(i) = y.coeffRef(i);}

        for( int i = 0 ; i<x.size() ; ++i)
          {
            s1=0.;
            for(int j =0 ; j< i ; ++j )
              {s1 += A.coeffRef(i,j)*y.coeffRef(j); }

            for(int j = i+1 ; j< x.size() ; ++j)
              {s1 += A.coeffRef(i,j)*x.coeffRef(j); }

            y.coeffRef(i) = (b.coeffRef(i) - s1)/(A.coeffRef(i,i));
           }

           n_ite = n_ite + 1 ;
           cout <<n_ite<<endl;

           for( int i = 0 ; i<x.size() ; ++i)
           {diffxy.coeffRef(i) = x.coeffRef(i) - y.coeffRef(i);}

           r = b - A*y;

           if(mon_flux)
            {
              mon_flux<<n_ite<<" "<<r.norm()<<endl;
            }


    }
    if(n_ite >= n_ite_max)
      {cout << "Tolérance non atteinte"<<endl;}
     cout << "SGS a cv en " << n_ite<< " itérations"<< endl;

    return x;
}


SparseVector<double> res_min( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , SparseVector<double> x0, double eps , int n_ite_max )
{
//  SparseVector<double> r,z;

  //r.resize(b.size());
  double alpha(3.);
  int n = b.size();
  SparseVector<double> r(b - A*x0);
  //r = b - A*x0;
  int n_ite(0.);
  std::ofstream mon_flux; // Contruit un objet "ofstream"
  std::string name_file = ("sol_N"+to_string(n)+"_resmin.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file, ios::out);
  if(mon_flux)
         {
           mon_flux<<n_ite<<" "<<r.norm()<<endl;
         }
  while(r.norm() > eps && n_ite < n_ite_max)
  {
    SparseVector<double> z(A*r);
    alpha = r.dot(z)/z.dot(z);
    x = x + alpha*r;
    r = r - alpha*z;
    n_ite++;
    if(mon_flux)
         {
           mon_flux<<n_ite<<" "<<r.norm()<<endl;
         }
  }
   if(n_ite >= n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}
   cout << "le résidu minimum a cv en " << n_ite<< " itérations"<< endl;
   mon_flux.close();
  return x;
}


SparseVector<double> grad_conj( SparseMatrix<double> A , SparseVector<double> b , SparseVector<double> x , SparseVector<double> x0, double eps , int n_ite_max )
  {
    SparseVector<double> p0 ,pold  , pnew  , rold ,rnew ,xnew ,xold , r , z;
    int n_ite , n ;
    double alpha , beta ;
    n = x.size();
    p0.resize(n);
    pnew.resize(n);
    rold.resize(n);
    rold.resize(n);
    rnew.resize(n);
    xold.resize(n);
    xnew.resize(n);
    r.resize(n);
    z.resize(n);

    r = A*x0 - b;
    p0 = -r ;
    pold = p0 ;
    rold = r ;
    rnew = r;
    n_ite = 0;

    std::ofstream mon_flux; // Contruit un objet "ofstream"
    std::string name_file = ("sol_N"+to_string(n)+"_gradconj.txt");  //commande pour modifier le nom de chaque fichier
    mon_flux.open(name_file, ios::out);
    if(mon_flux)
         {
           mon_flux<<n_ite<<" "<<r.norm()<<endl;
         }

    while (r.norm() > eps && n_ite <= n_ite_max)
     {
       z = A*pold;
       alpha   =  -pold.dot(rold)/(z).dot(pold) ;
       xnew = xold + alpha*pold ;
       rnew = rold + alpha * z;
       beta = rnew.dot(rnew)/rold.dot(rold);
       pnew = -rnew + beta*pold;
       xold = xnew;
       rold = rnew;
       pold = pnew;
       r = b - A*xnew;
       n_ite++;
       if(mon_flux)
         {
           mon_flux<<n_ite<<" "<<r.norm()<<endl;
         }
     }
     cout << "le gradien conjugué a cv en " << n_ite<< " itérations"<< endl;
      if(n_ite >= n_ite_max)
        {cout << "Tolérance non atteinte"<<endl;}
    // cout <<"n_ite = "<<n_ite<<endl;
      mon_flux.close();
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

// Écrit un fichier avec la solution au format Paraview
void saveSolution(int it , int n_iter , double residu)
{
  std::ofstream mon_flux; // Contruit un objet "ofstream"
  std::string name_file = ("/sol_"+std::to_string(it)+"_gradconj.txt");  //commande pour modifier le nom de chaque fichier
  mon_flux.open(name_file, std::ios::out);

  if(mon_flux)
  {
      mon_flux<<n_iter<<residu<<endl;
  }
  mon_flux.close();
}
