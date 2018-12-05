#include "Solver.h"
#include <iostream>
#include <fstream>
#include "Dense"
#include "Sparse"
#include <cmath>
#include <iomanip>
using namespace std;
using namespace Eigen;

//____________________________________________________________
//Constructeur de la matrice A à partir d'un fichier.
Solveur::Solveur(const string name_file_read, bool sym)
{
  ifstream mon_flux(name_file_read);

  string ligne, colonne, valeur;
  getline(mon_flux,ligne); //lit la première ligne

  mon_flux >> _N; //lit le premier mot de la ligne 2 correspond au nombre de lignes

  int nonzero_elem;
  mon_flux >> nonzero_elem; //lit le nombre de colonnes (valeur stockée inutilement, je ne savais pas comment lire sans stocker...)
  mon_flux >> nonzero_elem; //lit le nombre d'élements non nuls

  // Définition de la la matrice A.
  _A.resize(_N,_N);
  vector<Triplet<double>> liste_elem;
  for (int i = 0; i < nonzero_elem; i++)
  {
    mon_flux >> ligne;
    mon_flux >> colonne;
    mon_flux >> valeur;
    liste_elem.push_back({stoi(ligne)-1,stoi(colonne)-1,stod(valeur)});
    if ((colonne != ligne) && sym) // dans le cas de cette matrice symétrique seulement la moitié des éléments sont référencés dans le fichier texte
    {
      liste_elem.push_back({stoi(colonne)-1,stoi(ligne)-1,stod(valeur)});
    }
  }
  _A.setFromTriplets(liste_elem.begin(),liste_elem.end());
  mon_flux.close();

  //Définition de b, sol, r
  _b.resize(_N);
  _sol.resize(_N);
  for (int i = 0; i < _N; i++)
  {
    _b[i] = 1.;  _sol[i] = 0.;
  }
  _r = _b - _A*_sol;
}




//____________________________________________________________
//Constructeur de la matrice An définie en question 1)
Solveur::Solveur(const int N):_N(N)
{
  // //Calcul de Bn
  // MatrixXd B(_N,_N), Id(_N,_N),M(_N,_N);
  // for (int i=0; i<_N;i++)
  // {
  //   for (int j=0; j<_N;j++)
  //   {
  //     B(i,j)=(rand() % 1000000001)/1.e9; // rand() donne un nombre entier aléatoire en 0 et 10^9, en divisant par 1.e9 on obtient un nombre entre 0 et 1, avec 9 chiffres significatifs
  //   }
  // }
  //
  // Id.setIdentity();
  // double alpha = 1.;
  // M = alpha*Id + B.transpose()*B;
  // _A=M.sparseView(); // Tous les algorithmes fonctionnent avec des matrices creuses
  //
  // _b.resize(_N);
  // _sol.resize(_N);
  // for (int i = 0; i < _N; i++)
  // {
  //   _b[i] = 1.;
  //   _sol[i] = 0.;
  // }
  //////////////////// debut du test///////////////
  SparseMatrix<double> In(_N,_N) , Bn(_N,_N) , An(_N,_N), BnTBn(_N,_N);
  In.setIdentity();
  _A.resize(_N,_N);
  double alpha(3*_N);

  for (int i =0 ; i<_N ; i++)
    {
      for (int j =0 ; j<_N ; j++)
        {
        double nb_alea = rand()/(double)RAND_MAX  ;
        Bn.coeffRef(i,j) = nb_alea;
      }
    }


  //Création de An
  BnTBn = Bn.transpose()*Bn;
  for (int i =0 ; i< _N; i++)
    {
      for (int j =0 ; j< _N; j++)
      {
        An.insert(i,j) =alpha*In.coeffRef(i,j) + BnTBn.coeffRef(i,j);
      }
    }

    _A = An;
    //////////////////////fin du test////////////////
    _b.resize(_N);
    _sol.resize(_N);
    for (int i = 0; i < _N; i++)
    {
      _b[i] = 1.;
      _sol[i] = 0.;
    }
  _r = _b - _A*_sol;
}




//______________________________________________________________________
// Construction des matrices D, E et F de Jacobi à partir de la matrice A.
void Solveur::InitialiseDEF()
{
  _D.resize(_N,_N);
  _E.resize(_N,_N);
  _F.resize(_N,_N);

  for (int i=0; i<_N; i++)
  {
    _D.insert(i,i)=_A.coeffRef(i,i);
  }
  _F = - _A.triangularView<StrictlyUpper>();
  _E = - _A.triangularView<StrictlyLower>();
}


//______________________________________________________________________
//Initialisation des matrices K=(D-wE) et G=(D-wF) pour omega = 0.5
void Solveur::InitialiseKG(double omega)
{
  _K=_D-omega*_E;
  _G=_D-omega*_F;
}


//______________________________________________________________________
//Initialisation des matrices K=(D-wE) et G=(D-wF) pour omega = 1.5
void Solveur::InitialiseK1G1(double omega1)
{
  _K1=_D-omega1*_E;
  _G1=_D-omega1*_F;
}



//______________________________________________________________________
//Méthode de Jacobi
void Solveur::Jacobi(const int kmax,const double eps, vector<double> norme) //norme stocke les normes des résidus
{
  //Définition des matrices D,E,F.
  Solveur::InitialiseDEF();

  //Inversion de D (diagonale)
  SparseMatrix<double> Dinv;
  Dinv.resize(_N,_N);
  for (int i=0; i<_N; i++)
  {
    Dinv.insert(i,i)=1./_D.coeffRef(i,i);
  }

  //Algorithme de Jacobi
  int k = 0;
  double nr = _r.norm();
  while (nr>eps && k<=kmax)
  {
    _sol=(_E+_F)*_sol;
    _sol = Dinv*(_sol+_b);
    _r = _b - _A*_sol;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  cout << "nombre d'itérations = " << k << endl;
  cout << "Norme résidu : " << nr << endl;
  Solveur::SaveSol("Jacobi.txt",norme);
}



//______________________________________________________________________
//Méthode du GP0
void Solveur::GPO(const int kmax, const double eps, vector<double> norme)
{
  VectorXd z; //Permet de stocker le produit A*r
  z.resize(_N);
  double nr = _r.norm();
  norme.push_back(nr);
  int k = 0;
  double alpha;

  //Algorithme du GPO
  while (nr>eps && k<=kmax)
  {
    z=_A*_r;
    alpha=_r.dot(_r)/(z.dot(_r));
    _sol=_sol+alpha*_r;
    _r=_r-alpha*z;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("GPO.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum
void Solveur::ResiduMinimum(const int kmax, const double eps, vector<double> norme)
{
  VectorXd z; //Permet de stocker le produit A*r
  z.resize(_N);
  double nr = _r.norm();
  norme.push_back(nr);
  int k = 0;

  //Algorithme du Résidu Minimu
  while (nr>eps && k<=kmax)
  {
    z=_A*_r;
    double alpha=z.dot(_r)/(z.dot(z));
    _sol+=alpha*_r;
    _r=_r-alpha*z;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimum.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum Préconditionnée à gauche avec Jacobi
void Solveur::ResiduMinimumGaucheJacobi(const int kmax, const double eps, vector<double> norme)
{
  VectorXd w, z, q;
  w.resize(_N);
  z.resize(_N);
  q.resize(_N);

  //Calcul des matrices D, E et F de Jacobi.
  Solveur::InitialiseDEF();

  //Résolution de Mq=r
  q = Solveur::InversionD(_r);

  double nr = _r.norm();
  norme.push_back(nr);
  int k =0;
  double alpha;

  //Algorithme de la méthode du Résidu Minimum préconditionné à gauche.
  while (nr>eps && k<=kmax)
  {
    w=_A*q;
    //Résolution de Mz=w
    z= Solveur::InversionD(w);
    alpha=q.dot(z)/(z.dot(z));
    _sol+=alpha*q;
    _r=_r-alpha*w;
    q=q-alpha*z;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimumGaucheJacobi.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum Préconditionnée à droite avec Jacobi
void Solveur::ResiduMinimumDroiteJacobi(const int kmax, const double eps, vector<double> norme)
{
  VectorXd w, z;
  w.resize(_N);
  z.resize(_N);
  int k=0;

  //Calcul des matrices D, E et F de Jacobi.
  Solveur::InitialiseDEF();

  double nr = _r.norm();
  norme.push_back(nr);
  double alpha = 0;

  //Algorithme de la méthode du Résidu Minimum préconditionné à droite.
  while (nr>eps && k<=kmax)
  {
    //Résolution de Mz=r
    z= Solveur::InversionD(_r);
    VectorXd w=_A*z;
    alpha=_r.dot(w)/(w.dot(w));
    _sol+=alpha*z;
    _r=_r-alpha*w;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimumDroiteJacobi.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum préconditionnée à gauche avec SSOR
void Solveur::ResiduMinimumGaucheSSOR(const int kmax, const double eps, vector<double> norme)
{
  VectorXd w, z, q;
  w.resize(_N);
  z.resize(_N);
  q.resize(_N);
  double omega = 0.5;
  //Solveurs de SimplicialLLTpour simplifier la résolution des systèmes futurs
  SimplicialLLT <SparseMatrix<double>> solver1, solver2;

  //Décomposition des matrices K et G.
  Solveur::InitialiseDEF();
  Solveur::InitialiseKG(omega);
  solver1.compute(_K);
  solver2.compute(_G);

  VectorXd x,y;
  x.resize(_N);
  y.resize(_N);

  y = solver1.solve(_r);
  x=_D*y;
  q = solver2.solve(x);(_N);

  double nr = _r.norm();
  norme.push_back(nr);

  int k = 0;
  double alpha;
  while (nr>eps && k<=kmax)
  {
    w=_A*q;
    y = solver1.solve(w);
    x=_D*y;
    z = solver2.solve(x);
    alpha=q.dot(z)/(z.dot(z));
    _sol+=alpha*q;
    _r=_r-alpha*w;
    q=q-alpha*z;
    nr = _r.norm();
    cout << fixed << setprecision(18) << nr << endl;
    norme.push_back(nr);
    k+=1;
  }
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimumGaucheSSOR.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum préconditionnée à droite avec SSOR
void Solveur::ResiduMinimumDroiteSSOR(const int kmax, const double eps, std::vector<double> norme)
{
  VectorXd x, y, w, z;
  x.resize(_N);
  y.resize(_N);
  w.resize(_N);
  z.resize(_N);
  double omega = 0.5;
  int k=0;

  SimplicialLLT <SparseMatrix<double>> solver1;
  SimplicialLLT <SparseMatrix<double>> solver2;

  Solveur::InitialiseDEF();
  Solveur::InitialiseKG(omega);
  solver1.compute(_K);
  solver2.compute(_G);

  double nr = _r.norm();
  double alpha;
  norme.push_back(nr);

  while (nr>eps && k<=kmax)
  {
    y =solver1.solve(_r);
    x=_D*y;
    z = solver2.solve(x);

    VectorXd w=_A*z;
    alpha=_r.dot(w)/(w.dot(w));
    _sol+=alpha*z;
    _r=_r-alpha*w;
    nr = _r.norm();
    cout << nr << endl;
    norme.push_back(nr);
    k+=1;
  }
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimumDroiteSSOR.txt",norme);
  cout << "Norme résidu : " << nr << endl;
  cout << "nbr itérations = " << k-1 << endl;
}



//______________________________________________________________________
//Méthode du Résidu Minimum FLEX Précondtionnée à droite avec SSOR
void Solveur::ResiduMinimumDroiteSSORFlex(const int kmax, const double eps, std::vector<double> norme)
{
  VectorXd x, y, w, z;
  x.resize(_N);
  y.resize(_N);
  w.resize(_N);
  z.resize(_N);
  double omega = 0.5;
  double omega1 = 1.5;
  int k=0;

  SimplicialLLT <SparseMatrix<double>> solver1, solver2, solver3, solver4;
  Solveur::InitialiseDEF();

  Solveur::InitialiseKG(omega);
  solver1.compute(_K);
  solver2.compute(_G);

  Solveur::InitialiseK1G1(omega1);
  solver3.compute(_K1);
  solver4.compute(_G1);

  double nr = _r.norm();
  double alpha;
  norme.push_back(nr);

  // On alterne pour chaque itération, o fait un préconditionnement SSOR avec omega=0.5, puis avec omega=1.5
  while (nr>eps && k<=kmax)
  {
    if (k%2==0)
    {
      y =solver1.solve(_r);
      x=_D*y;
      z = solver2.solve(x);
    }
    else
    {
      y =solver3.solve(_r);
      x=_D*y;
      z = solver4.solve(x);
    }
    VectorXd w=_A*z;
    alpha=_r.dot(w)/(w.dot(w));
    _sol+=alpha*z;
    _r=_r-alpha*w;
    nr = _r.norm();
    cout << "Norme résidu : " << nr << endl;
    cout << "nbr itérations = " << k-1 << endl;
    norme.push_back(nr);
    k+=1;
  }
  if (k>kmax)
    cout << "Tolérance non atteinte " << endl;
  Solveur::SaveSol("ResiduMinimumDroiteSSORFlex.txt",norme);
}



//______________________________________________________________________
//Méthode du Résidu Minimum Auto-Préconditionnée à droite.
void Solveur::ResiduMinimumDroiteAuto(const int kmax, const double eps, std::vector<double> norme)
{
  VectorXd x, w, z, r1;
  x.resize(_N);
  w.resize(_N);
  z.resize(_N);
  vector<double> norme1;
  int k=0;
  double alpha;


  double nr = _r.norm();
  norme.push_back(nr);

  while (nr>eps && k<=kmax)
  {
    z.setZero();
    r1=_r;
    //5 Premières itérations de ...
    for(int i=0;i<5;i++)
    {
      x=_A*r1;
      alpha=r1.dot(x)/(x.dot(x));
      z+=alpha*r1;
      r1=r1-alpha*x;
    }
    VectorXd w=_A*z;
    alpha=_r.dot(w)/(w.dot(w));
    _sol+=alpha*z;
    _r=_r-alpha*w;
    nr = _r.norm();
    norme.push_back(nr);
    k+=1;
  }
  if (k>kmax)
  {
    cout << "Tolérance non atteinte " << endl;
  }
    cout << "nombre d'itérations = " << k << endl;
    cout << "Norme résidu : " << nr << endl;
  Solveur::SaveSol("ResiduMinimumDroiteAuto.txt",norme);
}



//______________________________________________________________________
//Ecriture des normes des résidus à chaque itération dans un fichier.
void Solveur::SaveSol(string name_file, vector<double> norme)
{
  system(("rm ./" + name_file).c_str());
  ofstream mon_flux; // Contruit un objet "ofstream"
  mon_flux.open(name_file, ios::out); // Ouvre un fichier appelé name_file
  if(mon_flux) // Vérifie que le fichier est bien ouvert
  {
    for (int i = 0 ; i < norme.size() ; i++)
    mon_flux <<i<<" "<< norme[i] << endl; // Remplit le fichier
  }
  else // Renvoie un message d’erreur si ce n’est pas le cas
  {
    cout << "ERREUR: Impossible d’ouvrir le fichier." << endl;
  }
  mon_flux.close(); // Ferme le fichier
}



//______________________________________________________________________
//Inversion du système Dq=u avec D la diagonale de A.
VectorXd Solveur::InversionD(VectorXd u)
{
  VectorXd q;
  q.resize(u.size());
  for (int i=0; i<u.size(); i++)
  {
    q(i)=u(i)/_D.coeffRef(i,i);
  }

  return q;
}



//______________________________________________________________________
//Méthode du GMRes
void Solveur::GMRes( const int kmax, const double eps, vector<double> norme)
{
  int m=10;//Dimension de l'espace de Krylov.
  double beta = _r.norm();
  norme.push_back(beta);

  SparseVector<double> sol1, r1(_N), y(m), e1(m+1);
  sol1=_sol.sparseView();
  SparseMatrix<double> VM;

  //Vecteur e1 de la base canonique de R^(m+1)
  for (int i=1; i<m+1; i++)
  {e1.coeffRef(i)=0.;}
  e1.coeffRef(0)=1.;

  _Hm.resize(m+1,m);
  _Qm.resize(m+1,m+1);
  int k=0;

  //Algorithme de la méthode GMRes
  while(beta>eps && k<=kmax)
  {
    _Hm.setZero();
    VM=Solveur::Arnoldi(_r);
    y= Solveur::argmin(beta, m);
    sol1+= VM*y;
    _sol=VectorXd(sol1);
    _r=_b-_A*_sol;
    beta=_r.norm();
    k+=1;
    norme.push_back(beta);
  }
  //cout << "testyeah" << endl;

  cout <<"Norme du résidu :"<<beta<< endl;
  cout << "Nombre d'itérations :" << k << endl;
  if (k>kmax)
  {
    cout << "Tolérance non atteinte " << endl;
  }
  Solveur::SaveSol("GMRes.txt",norme);
}



//______________________________________________________________________
// Résolution de y =argmin de ||beta*e1-Hmy||
SparseVector<double> Solveur::argmin(double beta, int m)
{
  SparseVector<double> e1, y, hj;
  e1.resize(m+1);
  y.resize(m);
  hj.resize(m+1);

  //Vecteur e1 de la base canonique de R^(m+1)
  for (int i=1; i<m; i++)
  {e1.coeffRef(i)=0.;}
  e1.coeffRef(0)=1.;
  SparseVector<double> gm(m+1);

  //Decomp de Hm en QR avec l'algorithme de Givens
  vector<SparseVector<double>> Q(m+1);
  SparseMatrix<double> R(m+1,m);
  R=Solveur::Givens(m);

  //Calcul de gm=beta*Q(transposée)*e1
  gm=beta*_Qm.transpose()*e1;

  //Inversion de R pour R*y=gm
  y.coeffRef(m-1)=gm.coeffRef(m-1)/R.coeffRef(m-1,m-1);
  for (int i=m-2; i>=0; i--)
  {
    y.coeffRef(i)=gm.coeffRef(i);
    for (int j=i+1; j<m; j++)
    {y.coeffRef(i)-=R.coeffRef(i,j)*y.coeffRef(j);}
    y.coeffRef(i)=y.coeffRef(i)/R.coeffRef(i,i);
  }
  return y;
}



//______________________________________________________________________
//Méthode de décomposition de Hm sous forme QR de Givens ()
//On récupère R et l'attribut Qm est modifié.
SparseMatrix<double> Solveur::Givens(int m)
{
  SparseMatrix<double> R,G;
  R=_Hm;
  _Qm.setIdentity();
  G.resize(m+1,m+1); //matrice de transformation
  double c,s,norme;
  for(int j=0; j<m; j++)
  {
      G.setIdentity();
      c=_Hm.coeffRef(j,j);
      s=-_Hm.coeffRef(j+1,j);
      norme=sqrt(pow(c,2)+pow(s,2));
      c=c/norme;
      s=s/norme;
      G.coeffRef(j,j)=c;
      G.coeffRef(j+1,j+1)=c;
      G.coeffRef(j,j+1)=-s;
      G.coeffRef(j+1,j)=s;
      R=G*R;
    _Qm=_Qm*G.transpose();
  }
  return R;
}



//______________________________________________________________________
// Méthode d'Arnoldi utilisée dans GMRes
SparseMatrix<double> Solveur::Arnoldi(VectorXd v) // Arnoldi retourne la matrice VM (notée normalement Vm)
{
  int m;
  m=10;
  SparseVector<double> v1,w,z;
  v1=v.sparseView();
  vector<SparseVector<double>> Vm; //Stockage des vecteur vi
  Vm.resize(m);
  Vm[0]=v1/v1.norm();

  //Algorithme d'Arnoldi
  for (int j=0; j<m; j++)
  {
    w=_A*Vm[j];
    z=w;
    for (int i=0; i<=j; i++)
    {
      _Hm.coeffRef(i,j)=w.dot(Vm[i]);
      z=  z - _Hm.coeffRef(i,j)*Vm[i];
    }
    _Hm.coeffRef(j+1,j)=  (z).norm();
    if (_Hm.coeffRef(j+1,j)==0)
    {exit(0);}
    else
    {
      z=z/_Hm.coeffRef(j+1,j);
      if (j!=m-1)
      {Vm[j+1]=z;}
    }
  }
  //Passage de Vm de vector(SparseVector<double>) en SparseMatrix<double>.
  SparseMatrix<double> Mat_V(_N,m);
  for(int i=0; i<_N; i++)
  {
    for (int j=0; j<m; j++)
    {Mat_V.coeffRef(i,j)=Vm[j].coeffRef(i);}
  }
  return Mat_V;
}
