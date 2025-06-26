#ifndef COR_H
#define COR_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <boost/math/distributions/chi_squared.hpp>

#include "ptable.h"
#include "padjusttable.h"
#include "util.h"
#include "matrix.h"
#include "options.h"


class CorPearson {
public:
    CorPearson() {};

    ~CorPearson() {};

    // Returns the pearson correlation coefficient of each row between matrix X and matrix Y.
    // Note: If matrix Y is empty, the Pearson correlation will be computed between X and itself.
    static void parallelCalcCor(Matrix<double> &X, Matrix<double> &Y, double *result, int nthreads);

    static double calcCor(double *x, double *y, size_t n);

//    /**
//     * @brief Calculates the p-values for the given pearson correlation matrix in parallel.
//     *
//     * @param X Dynamic pointer to the correlation matrix data.
//     * @param size The size of the correlation matrix.
//     * @param ncols The number of columns in the raw data for calculating the correlation matrix.
//     * @param isSysmatric Boolean flag indicating whether the correlation matrix is symmetric.
//     * @param P Dynamic pointer to store the calculated p-values.
//     * @param ptable The PTable object containing the p-value table.
//     * @param nthreads The number of threads to use for parallel computation.
//     *
//     */
//    static void parallelCalcPvalue(const double *X, size_t size, size_t ncols, bool isSysmatric, double *P,
//                                   const PTable &ptable, int nthreads);


    static double calcPvalue(double r, double df, const PTable &ptable);

    /**
     * @brief Common calculation function that calculates the p-values for the given pearson correlations in parallel.
     *
     * @param X Dynamic pointer to the correlations data.
     * @param n The size of the correlations.
     * @param P Dynamic pointer to store the calculated p-values.
     * @param df The degrees of freedom, df = n - 2, n is the number of samples used to calculate the correlations.
     * @param nthreads The number of threads to use for parallel computation.
     */
//    static void parallelCommonCalcPvalue(const double *X, size_t n, double *P, double df, int nthreads);

    static double commonCalcPvalue(double r, double df, const boost::math::students_t &dist);

    // normalizes the elements of each row in matrix to have mean 0 and standard deviation 1 / sqrt(n).
    static void parallelPreprocessNormalize(Matrix<double> &X, int nthreads);

    static void preprocessNormalize(double *v, size_t n);
};

class CorSpearman {
public:
    CorSpearman() {};

    ~CorSpearman() {};

    // Returns the spearman correlation coefficient of each row between matrix X and matrix Y.
    // Note: If matrix Y is empty, the Pearson correlation will be computed between X and itself.
    static void parallelCalcCor(Matrix<double> &X, Matrix<double> &Y, double *result, int nthreads);

    static double calcCor(const double *x, const double *y, size_t n);

    /**
     * @brief Approximate calculation function that calculates the p-values for the given pearson correlations in parallel.
     *
     * @param X Dynamic pointer to the correlation matrix data.
     * @param n The size of the correlations.
     * @param P Dynamic pointer to store the calculated p-values.
     * @param ptable The PTable object containing the p-value table.
     * @param nthreads The number of threads to use for parallel computation.
     *
     */
//    static void parallelCalcPvalue(const double *X, size_t n, double *P, double df, const PTable &ptable, int nthreads);
};

struct KendallStat {
    double v0;
    double vt;
    double v1;
    double v2;
    double n1;
    double n2;

    KendallStat(double v0_, double vt_, double v1_, double v2_, double n1_, double n2_) :
            v0(v0_), vt(vt_), v1(v1_), v2(v2_), n1(n1_), n2(n2_) {
    }

    KendallStat() :
            v0(0), vt(0), v1(0), v2(0), n1(0), n2(0) {
    }

    void print(std::ostream &os) const {
        os << "v0 = " << v0 << ", vt = " << vt << ", v1 = " << v1 << ", v2 = " << v2 << ", n1 = " << n1 << ", n2 = "
           << n2 << std::endl;
    }
};

class CorKendall {
public:
    CorKendall() {};

    ~CorKendall() {};

    // Returns the kendall tau correlation coefficient of each row between matrix X and matrix Y.
    // Note: If matrix Y is empty, the Pearson correlation will be computed between X and itself.
    static void parallelCalcCor(const Matrix<double> &X, const Matrix<double> &Y, double *result, int nthreads);

    static std::pair<double, double> calcCor(const double *x, const double *y, size_t n);

    /**
     * @brief Calculates the p-value for the given kendall correlation.
     *
     * @param s The value of the correlation.
     * @param xstat The KendallStat for vector x used to calculate the correlation.
     * @param ystat The KendallStat for vector y used to calculate the correlation.
     * @param ptable The PTable object containing the p-value table.
     */
    static double calcPvalue(double s, const KendallStat &xstat, const KendallStat &ystat, const PTable &ptable);

    static double commonCalcPvalue(double s, const KendallStat &xstat, const KendallStat &ystat,
                                   const boost::math::normal_distribution<> &dist);

    // Returns the tie count of each row in matrix.
    static std::vector<KendallStat> parallelGetKendallStat(const Matrix<double> &X, int num_threads);

    static KendallStat getKendallStat(const std::vector<uint64_t> &ties, size_t n);

    static std::vector<uint64_t> getTies(const double *x, size_t n);

private:
    // This is a experimental function.
    // Sorts in place, returns the insertion sort inversions number between the input array.
    static uint64_t insertionSort(double *v, size_t n);

    static uint64_t merge(double *begin, double *mid, double *end);

    // Sorts in place, returns the merge sort inversions number between the input array.
    static uint64_t mergeSort(double *begin, double *end);

    // Return tie count of the input array.
    static uint64_t getMs(double *begin, double *end);

    // Sort in place, ordered by the first array.
    static void zipSort(double *x, double *y, size_t n);
};

/**
 * @brief Class for calculating phi coefficient (correlation for binary variables).
 * 
 * Phi coefficient measures the association between two binary variables.
 * It is equivalent to Pearson correlation for binary variables and ranges from -1 to 1.
 */
class CorPhi {
public:
    CorPhi() {};

    ~CorPhi() {};

    /**
     * @brief Calculates phi coefficient matrix between rows of matrices X and Y in parallel.
     * 
     * @param X Input matrix X
     * @param Y Input matrix Y (if empty, calculates phi between X and itself)
     * @param result Pointer to store the result matrix
     * @param nthreads Number of threads for parallel computation
     * @param threshold Threshold for binarization (default: 0.5)
     */
    static void parallelCalcCor(Matrix<double> &X, Matrix<double> &Y, 
                               double *result, int nthreads, double threshold = 0.5);

    /**
     * @brief Calculates phi coefficient between two vectors.
     * 
     * @param x First vector
     * @param y Second vector
     * @param n Length of vectors
     * @param threshold Threshold for binarization (default: 0.5)
     * @return Phi coefficient value
     */
    static double calcCor(const double *x, const double *y, size_t n, double threshold = 0.5);

    /**
     * @brief Calculates p-value for phi coefficient using chi-square test.
     * 
     * @param phi Phi coefficient value
     * @param n Sample size
     * @param ptable PTable object for p-value calculation
     * @return P-value
     */
    static double calcPvalue(double phi, size_t n, const PTable &ptable);

    /**
     * @brief Common calculation function for p-values using chi-square distribution.
     * 
     * @param phi Phi coefficient value
     * @param n Sample size
     * @param dist Chi-square distribution
     * @return P-value
     */
    static double commonCalcPvalue(double phi, size_t n, const boost::math::chi_squared &dist);

private:
    /**
     * @brief Binarizes continuous data using a threshold.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param threshold Threshold for binarization
     */
    static void binarize(double *data, size_t n, double threshold);

    /**
     * @brief Calculates phi coefficient from contingency table.
     * 
     * @param a Count of (1,1) pairs
     * @param b Count of (1,0) pairs
     * @param c Count of (0,1) pairs
     * @param d Count of (0,0) pairs
     * @return Phi coefficient value
     */
    static double calcPhiFromContingency(int a, int b, int c, int d);

    /**
     * @brief Creates contingency table from binary vectors.
     * 
     * @param x First binary vector
     * @param y Second binary vector
     * @param n Length of vectors
     * @return Contingency table [a, b, c, d]
     */
    static std::vector<int> createContingencyTable(const double *x, const double *y, size_t n);
};

#endif // COR_H