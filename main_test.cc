#include "DM_test.h"  // Inclure le fichier .h

using namespace std;
using namespace Eigen;

int main()
{
  int userChoiceMeth(0);
  int n_ite_max(200);
  double eps(0.01);
  int N;

  cout << "Veuillez choisir la méthode de résolution pour Ax=b:" << endl;
  cout << "1) Méthode du Résidu Minimum" << endl;
  cout << "2) Méthode du Gradient Conjugué" << endl;
  cout << "3) Méthode SGS" << endl;
  cin >> userChoiceMeth;

  // exemple d'une matrice 3x3
  Matrix<double, 3, 3>  In, Bn, An;
  In = MatrixXd::Identity(3,3);
  Bn = MatrixXd::Random(3,3);  //Matrice de coefficient aléatoire entre -1 et 1

  //création de Bn
  for (int i =0 ; i<3 ; i++)
  {
      for (int j =0 ; j<3 ; j++)
    {
      Bn(i,j) = abs(Bn(i,j));
    }
  }

  //Création de An
  for (int i =0 ; i<3 ; i++)
  {
    for (int j =0 ; j<3 ; j++)
    {
      An(i,j) = 2*In(i,j) + (Bn.transpose()*Bn)(i,j);
    }
  }
   cout <<"An = "<<endl<< An <<endl;

  //création de b
  VectorXd b(3);

  for (int i =0 ; i<3 ;i++)
  {
    b[i]=1.+i;
  }
  cout <<"    "<<endl;
  cout <<"b = "<<endl<< b <<endl;
  cout <<"    "<<endl;

  // creation de x0
  VectorXd x0(3);
  for( int i = 0 ; i < 3 ; ++i)
  {
    x0[i]=0.;
  }

  int n_ite(0);
  VectorXd z(3);

  // on construit le residu minimum
  MethIterative* MethIterate(0);

  switch(userChoiceMeth)
  {
    case 1:
      MethIterate = new ResiduMin();
      MethIterate->MatrixInitialize(An);

      MethIterate->Initialize(x0, b);

      while(MethIterate->GetResidu().lpNorm<Infinity>() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->GetResidu();
        MethIterate->Advance(z);
        n_ite++;
      }

      if(n_ite > 200)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec res_min = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;

      break;

    case 2:
      MethIterate = new GradientConj();
      MethIterate->MatrixInitialize(An);
      MethIterate->Initialize(x0, b);

      while(MethIterate->GetResidu().lpNorm<Infinity>() > eps && n_ite < n_ite_max)
      {
        z = An*MethIterate->Getp();
        MethIterate->Advance(z);
        n_ite++;
      }

      if (n_ite > 200)
        {cout << "Tolérance non atteinte"<<endl;}

      cout <<"  "<<endl;   // d'après doc internet si x0 est trop éloigné de x les résultats ne converge plus
      cout <<"x avec grad_conj = "<<endl << MethIterate->GetIterateSolution() <<endl;
      cout <<"    "<<endl;
      cout << An*MethIterate->GetIterateSolution()<<endl;
      cout <<"    "<<endl;

      break;

    default:
      cout << "Ce choix n'est pas possible" << endl;
      exit(0);
  }

  delete MethIterate;
  MethIterate = 0;


  return 0;
}
