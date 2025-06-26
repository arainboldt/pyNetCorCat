#ifndef MUTUAL_INFO_H
#define MUTUAL_INFO_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "matrix.h"
#include "util.h"

/**
 * @brief Enumeration for different discretization methods used in mutual information calculation.
 */
enum class DiscretizationMethod {
    EQUAL_WIDTH,      ///< Equal-width binning
    EQUAL_FREQUENCY,  ///< Equal-frequency binning
    ADAPTIVE          ///< Adaptive binning based on data distribution
};

/**
 * @brief Statistics structure for mutual information calculation.
 * Contains information about discretization and entropy calculations.
 */
struct MutualInfoStat {
    std::vector<int> bins;           ///< Discretized values
    std::vector<int> histogram;      ///< Histogram of discretized values
    double entropy;                  ///< Entropy of the variable
    size_t n_samples;                ///< Number of samples
    int n_bins;                      ///< Number of bins used

    MutualInfoStat() : entropy(0.0), n_samples(0), n_bins(0) {}

    MutualInfoStat(const std::vector<int> &bins_, const std::vector<int> &hist_, 
                   double entropy_, size_t n_samples_, int n_bins_) :
            bins(bins_), histogram(hist_), entropy(entropy_), n_samples(n_samples_), n_bins(n_bins_) {}

    void print(std::ostream &os) const {
        os << "entropy = " << entropy << ", n_samples = " << n_samples 
           << ", n_bins = " << n_bins << std::endl;
    }
};

/**
 * @brief Class for calculating mutual information between variables.
 * 
 * Mutual information measures the mutual dependence between two random variables.
 * It quantifies the amount of information obtained about one random variable
 * through observing the other random variable.
 */
class MutualInfo {
public:
    MutualInfo() {};

    ~MutualInfo() {};

    /**
     * @brief Calculates mutual information matrix between rows of matrices X and Y in parallel.
     * 
     * @param X Input matrix X
     * @param Y Input matrix Y (if empty, calculates MI between X and itself)
     * @param result Pointer to store the result matrix
     * @param nthreads Number of threads for parallel computation
     * @param bins Number of bins for discretization (default: 10)
     * @param method Discretization method (default: EQUAL_WIDTH)
     */
    static void parallelCalcMI(Matrix<double> &X, Matrix<double> &Y, 
                              double *result, int nthreads, int bins = 10,
                              DiscretizationMethod method = DiscretizationMethod::EQUAL_WIDTH);

    /**
     * @brief Calculates mutual information between two vectors.
     * 
     * @param x First vector
     * @param y Second vector
     * @param n Length of vectors
     * @param bins Number of bins for discretization (default: 10)
     * @param method Discretization method (default: EQUAL_WIDTH)
     * @return Mutual information value
     */
    static double calcMI(const double *x, const double *y, size_t n, int bins = 10,
                        DiscretizationMethod method = DiscretizationMethod::EQUAL_WIDTH);

    /**
     * @brief Calculates mutual information with pre-computed statistics for efficiency.
     * 
     * @param x First vector
     * @param y Second vector
     * @param n Length of vectors
     * @param x_stat Pre-computed statistics for vector x
     * @param y_stat Pre-computed statistics for vector y
     * @param bins Number of bins for discretization
     * @param method Discretization method
     * @return Mutual information value
     */
    static double calcMIWithStats(const double *x, const double *y, size_t n,
                                 const MutualInfoStat &x_stat, const MutualInfoStat &y_stat,
                                 int bins, DiscretizationMethod method);

    /**
     * @brief Calculates p-values for mutual information using permutation tests.
     * 
     * @param mi_values Array of mutual information values
     * @param n_samples Number of samples used in MI calculation
     * @param n_permutations Number of permutations for significance testing
     * @param nthreads Number of threads for parallel computation
     * @return Array of p-values
     */
    static std::vector<double> calcPvalues(const double *mi_values, size_t n_values,
                                          size_t n_samples, int n_permutations, int nthreads);

    /**
     * @brief Preprocesses data by calculating statistics for each row in parallel.
     * This can be used to optimize repeated MI calculations.
     * 
     * @param X Input matrix
     * @param nthreads Number of threads for parallel computation
     * @param bins Number of bins for discretization
     * @param method Discretization method
     * @return Vector of MutualInfoStat objects for each row
     */
    static std::vector<MutualInfoStat> parallelGetMutualInfoStat(const Matrix<double> &X, 
                                                                 int nthreads, int bins,
                                                                 DiscretizationMethod method);

    /**
     * @brief Normalizes mutual information values to [0,1] range.
     * 
     * @param mi_values Array of mutual information values
     * @param n_values Number of values
     * @param normalization_method Method for normalization ("min_max", "z_score", "none")
     * @return Array of normalized values
     */
    static std::vector<double> normalizeMI(const double *mi_values, size_t n_values,
                                          const std::string &normalization_method = "min_max");

private:
    /**
     * @brief Discretizes continuous data into discrete bins.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param bins Number of bins
     * @param method Discretization method
     * @return Vector of discretized values
     */
    static std::vector<int> discretize(const double *data, size_t n, int bins,
                                      DiscretizationMethod method);

    /**
     * @brief Equal-width discretization.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param bins Number of bins
     * @return Vector of discretized values
     */
    static std::vector<int> equalWidthDiscretize(const double *data, size_t n, int bins);

    /**
     * @brief Equal-frequency discretization.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param bins Number of bins
     * @return Vector of discretized values
     */
    static std::vector<int> equalFreqDiscretize(const double *data, size_t n, int bins);

    /**
     * @brief Adaptive discretization based on data distribution.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param bins Number of bins
     * @return Vector of discretized values
     */
    static std::vector<int> adaptiveDiscretize(const double *data, size_t n, int bins);

    /**
     * @brief Calculates entropy of a discrete variable.
     * 
     * @param histogram Histogram of discrete values
     * @param total Total number of samples
     * @return Entropy value
     */
    static double calcEntropy(const std::vector<int> &histogram, size_t total);

    /**
     * @brief Calculates joint entropy of two discrete variables.
     * 
     * @param x_bins Discretized values of first variable
     * @param y_bins Discretized values of second variable
     * @param n Number of samples
     * @param bins Number of bins
     * @return Joint entropy value
     */
    static double calcJointEntropy(const std::vector<int> &x_bins, 
                                  const std::vector<int> &y_bins, size_t n, int bins);

    /**
     * @brief Creates histogram from discretized values.
     * 
     * @param bins Discretized values
     * @param max_bins Maximum number of bins
     * @return Histogram vector
     */
    static std::vector<int> createHistogram(const std::vector<int> &bins, int max_bins);

    /**
     * @brief Creates joint histogram from two sets of discretized values.
     * 
     * @param x_bins Discretized values of first variable
     * @param y_bins Discretized values of second variable
     * @param bins Number of bins
     * @return 2D joint histogram
     */
    static std::vector<std::vector<int>> createJointHistogram(const std::vector<int> &x_bins,
                                                              const std::vector<int> &y_bins,
                                                              int bins);

    /**
     * @brief Calculates mutual information statistics for a single vector.
     * 
     * @param data Input data array
     * @param n Length of data array
     * @param bins Number of bins
     * @param method Discretization method
     * @return MutualInfoStat object
     */
    static MutualInfoStat getMutualInfoStat(const double *data, size_t n, int bins,
                                           DiscretizationMethod method);

    /**
     * @brief Performs permutation test for significance testing.
     * 
     * @param x First vector
     * @param y Second vector
     * @param n Length of vectors
     * @param original_mi Original mutual information value
     * @param n_permutations Number of permutations
     * @param bins Number of bins
     * @param method Discretization method
     * @return P-value
     */
    static double permutationTest(const double *x, const double *y, size_t n,
                                 double original_mi, int n_permutations, int bins,
                                 DiscretizationMethod method);
};

#endif // MUTUAL_INFO_H 