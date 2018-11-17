
#include "DM_SLPI.h"  // Inclure le fichier .h
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include<complex>
#include "Dense"
#include "Sparse"

using namespace std;
using namespace Eigen;

int main()
{
//création d'une matrice de la forme souhaitée
int const n(10.);                                          //taille de la matrice
double alpha(2.);
SparseMatrix<double> In(n,n) , Bn(n,n) , An(n,n), BnTBn(n,n);


SparseVector<double> b(n) ,x(n) ,x1(n) ,x2(n);

In.setIdentity();

//création de Bn
for (int i =0 ; i<n ; i++)
  {
    for (int j =0 ; j<n ; j++)
      {float nombre = 0;
      srand(time(NULL));                             //fonction aléatoire basé sur le temps donc tous les coeffs de B sont égaux.
      nombre = (float)rand() / (float)RAND_MAX;      //Demander ou trouver mieux sur internet
      Bn.coeffRef(i,j) = nombre;}
  }
  cout <<Bn<<endl;

//Création de An
BnTBn = Bn.transpose()*Bn;
for (int i =0 ; i<n ; i++)
  {
    for (int j =0 ; j<n ; j++)
      {
        An.insert(i,j) =alpha*In.coeffRef(i,j) + BnTBn.coeffRef(i,j);

}
}
 cout <<"An = "<<endl<< An <<endl;

 //création de b
 for (int i =0 ; i<n ;i++)
  {b.coeffRef(i) = 1. +i;}
cout <<"    "<<endl;
cout <<"b = "<<endl<< b <<endl;
cout <<"    "<<endl;

// x = SGS( An , b ,  x , 0.01 , 200 );
// cout <<"x avec SGS = "<<endl << x <<endl;
// cout <<"    "<<endl;
// cout <<"Vérification en calculant An.x = "<<endl<< An*x<<endl;
//
// cout<<"---------------------------------------"<<endl;
// cout<<"Avec res min"<<endl;
// cout <<"    "<<endl;
//
SparseVector<double> x0;
x0.resize(x.size());
for( int i = 0 ; i < x.size() ; ++i)
  {x0.coeffRef(i) = 0.;}

x1 = res_min( An , b ,  x1 , x0 , 0.001 , 200 );   //ne fonctionne pas si l'on met autre chose que 0 dans x0
cout <<"  "<<endl;                                // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
cout <<"x avec res_min = "<<endl << x1 <<endl;
cout <<"    "<<endl;
cout << "A*x = "<<endl<<An*x1<<endl;
cout <<"    "<<endl;
//  cout<< An - Bn <<endl;



// //GRADIENT CONJUGUE//////////////////////////////////////////////////////////////////////////////
cout<<"---------------------------------------"<<endl;
cout<<"Avec le gradient conjugué"<<endl;
cout <<"    "<<endl;


x2 = grad_conj( An , b , x ,  x0,  0.1 , 200 );  //ne fonctionne pas si l'on met autre chose que 0 dans x0
cout <<"  "<<endl;
cout <<"x avec grad_conj = "<<endl << x2 <<endl;
cout <<"    "<<endl;
cout << "Vérification en calculant An*x = "<< endl << An*x2 <<endl;



//EN creux
// SparseMatrix<double> Sn ;
// cout <<"    "<<endl;
// cout <<"En creux = "<<endl;
// cout <<"    "<<endl;
// Sn = create_mat("test.mtx", false);
// cout <<"Sn = "<<Sn<<endl;

return 0;
}
