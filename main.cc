#include "Solveur.h"
#include <string>
#include "Dense"
#include <iostream>
#include <cmath>
using namespace std;
using namespace Eigen;

int main()
{
  int methode, matrice, N; //Taille Matrice An
  vector<double> norme; //Stocke les normes de résidu
  int kmax=10000; //Nombre d'itérations maximal
  double eps=0.001; //Précision / Condition d'arrêt

//Choix de la matrice à utiliser.
  cout<<"Quelle matrice souhaitez-vous tester?"<<endl;
 cout << "---------------------------------" << endl;
 cout << "bcsstk18 (1)" <<endl;
 cout << "fidapm37 (2)" <<endl;
 cout << "fs_760_3 (3)" <<endl;
 cout << "fs_541_4 (4)" <<endl;
 cout << "cas général (5)"<< endl;
 cin >> matrice ;

 Solveur *Solv(0);

 switch(matrice)
 {
   case 1:
   Solv = new Solveur("bcsstk18.mtx" , true);
   break;
   case 2:
   Solv = new Solveur("fidapm37.mtx");
   break;
   case 3:
   Solv = new Solveur("fs_760_3.mtx");
   break;
   case 4:
   Solv = new Solveur("s3rmt3m3.mtx");
   break;
   case 5:
   cout << "N= ?"<< endl;
   cin >> N;
   Solv = new Solveur(N);
   break;
   default:
   cout << "Choix impossible" << endl;
   exit(0);
 }

 //Choix du solveur linéaire à utiliser.
   cout << "Quel solveur souhaitez-vous utiliser?" <<endl;
   cout << "---------------------------------" << endl;
   cout << "Jacobi (1)" <<endl;
   cout << "GPO (2)" <<endl;
   cout << "Résidu Minimum (3)" <<endl;
   cout << "GMRes (4)" <<endl;
   cout << "Résidu Minimum préconditionné à gauche Jacobi (5)"<< endl;
   cout << "Résidu Minimum préconditionné à droite Jacobi (6)"<< endl;
   cout << "Résidu Minimum préconditionné à gauche SSOR (7)"<< endl;
   cout << "Résidu Minimum préconditionné à droite SSOR (8)"<< endl;
   cout << "Résidu Minimum préconditionné à droite SSORFlex (9)"<<endl;
   cout << "Résidu Minimum préconditionné à droite Auto (10)"<<endl;
   cin >> methode;

  switch(methode)
  {
    case 1:
      Solv->Jacobi(kmax,eps,norme);
    break;

    case 2:
      Solv->GPO(kmax, eps, norme);
    break;

    case 3:
      Solv->ResiduMinimum(kmax, eps, norme);
    break;

    case 4:
      Solv->GMRes(kmax, eps, norme);
    break;

    case 5:
      Solv->ResiduMinimumGaucheJacobi(kmax, eps, norme);
    break;

    case 6:
      Solv->ResiduMinimumDroiteJacobi(kmax, eps, norme);
    break;

    case 7:
      Solv->ResiduMinimumGaucheSSOR(kmax, eps, norme);
    break;

    case 8:
      Solv->ResiduMinimumDroiteSSOR(kmax, eps, norme);
    break;

    case 9:
      Solv->ResiduMinimumDroiteSSORFlex(kmax, eps, norme);
    break;

    case 10:
      Solv->ResiduMinimumDroiteAuto(kmax, eps, norme);
    break;

    default:
    cout << "Choix impossible. Recommencez svp" << endl;
    exit(0);
  }
  return 0;
}
