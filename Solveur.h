#include "Dense"
#include "Sparse"
#include <iostream>
#include <fstream>
class Solveur
{

  private:
    int _N;
    Eigen::SparseMatrix<double> _A, _D, _E, _F,_K, _G, _K1, _G1, _Hm, _Qm;
    Eigen::VectorXd _b;                  // Vecteur source
    Eigen::VectorXd _sol;                // Vecteur solution
    Eigen::VectorXd _r;                           // Résidu
  public:
    // Constructeurs
    Solveur(const std::string name_file_read, bool sym = 0);

    Solveur(const int N);

    void InitialiseDEF();

    void InitialiseKG(double omega);

    void InitialiseK1G1(double omega);

    void Jacobi(const int kmax, const double eps, std::vector<double> norme);

    void GPO(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimum(const int kmax, const double eps, std::vector<double> norme);

    Eigen::VectorXd InversionD(Eigen::VectorXd u);

    void ResiduMinimumGaucheJacobi(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimumDroiteJacobi(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimumGaucheSSOR(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimumDroiteSSOR(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimumDroiteSSORFlex(const int kmax, const double eps, std::vector<double> norme);

    void ResiduMinimumDroiteAuto(const int kmax, const double eps, std::vector<double> norme);

    void SaveSol(std::string name_file, std::vector<double> norme);

    void GradientOpti(Eigen::VectorXd x0, double epsilon, int kmax);

    void GMRes(const int kmax, const double eps, std::vector<double> norme);

    Eigen::SparseMatrix<double> Arnoldi(Eigen::VectorXd v);

    Eigen::SparseVector<double> argmin(double beta, int m);

    Eigen::SparseMatrix<double> Givens(int m);
};
