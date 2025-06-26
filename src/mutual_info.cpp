#include "mutual_info.h"

#include <cmath>
#include <omp.h>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstring>

// Constants for numerical stability
const double EPSILON = 1e-15;
const double LOG_EPSILON = -34.538776394910684; // log(1e-15)

void MutualInfo::parallelCalcMI(Matrix<double> &X, Matrix<double> &Y, 
                                double *result, int nthreads, int bins,
                                DiscretizationMethod method) {
    size_t m = X.rows();
    size_t n = Y.isEmpty() ? X.rows() : Y.rows();

    // Pre-compute statistics for each row in parallel
    std::vector<MutualInfoStat> x_stats = parallelGetMutualInfoStat(X, nthreads, bins, method);
    std::vector<MutualInfoStat> y_stats;
    
    if (!Y.isEmpty()) {
        y_stats = parallelGetMutualInfoStat(Y, nthreads, bins, method);
    }

    // Calculate mutual information matrix in parallel
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < m; ++i) {
        if (Y.isEmpty()) {
            // Self-correlation: calculate MI between X and itself
            for (size_t j = i + 1; j < n; ++j) {
                double mi = calcMIWithStats(X.row(i), X.row(j), X.cols(), 
                                          x_stats[i], x_stats[j], bins, method);
                result[i * n + j] = mi;
            }
            // Set diagonal to maximum possible MI (entropy of the variable)
            if (i < n) {
                result[i * n + i] = x_stats[i].entropy;
            }
            // Fill lower triangle symmetrically
            for (size_t j = 0; j < i; ++j) {
                result[i * n + j] = result[j * n + i];
            }
        } else {
            // Cross-correlation: calculate MI between X and Y
            for (size_t j = 0; j < n; ++j) {
                double mi = calcMIWithStats(X.row(i), Y.row(j), X.cols(), 
                                          x_stats[i], y_stats[j], bins, method);
                result[i * n + j] = mi;
            }
        }
    }
}

double MutualInfo::calcMI(const double *x, const double *y, size_t n, int bins,
                          DiscretizationMethod method) {
    // Check for edge cases
    if (n < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Check for constant vectors
    bool x_constant = true, y_constant = true;
    double x_first = x[0], y_first = y[0];
    
    for (size_t i = 1; i < n; ++i) {
        if (std::abs(x[i] - x_first) > EPSILON) x_constant = false;
        if (std::abs(y[i] - y_first) > EPSILON) y_constant = false;
        if (!x_constant && !y_constant) break;
    }
    
    if (x_constant || y_constant) {
        return 0.0; // No mutual information if one variable is constant
    }

    // Discretize the data
    std::vector<int> x_bins = discretize(x, n, bins, method);
    std::vector<int> y_bins = discretize(y, n, bins, method);

    // Calculate individual entropies
    std::vector<int> x_hist = createHistogram(x_bins, bins);
    std::vector<int> y_hist = createHistogram(y_bins, bins);
    
    double h_x = calcEntropy(x_hist, n);
    double h_y = calcEntropy(y_hist, n);

    // Calculate joint entropy
    double h_xy = calcJointEntropy(x_bins, y_bins, n, bins);

    // Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    double mi = h_x + h_y - h_xy;
    
    // Ensure non-negative result (numerical stability)
    return std::max(0.0, mi);
}

double MutualInfo::calcMIWithStats(const double *x, const double *y, size_t n,
                                   const MutualInfoStat &x_stat, const MutualInfoStat &y_stat,
                                   int bins, DiscretizationMethod method) {
    // Check for edge cases
    if (n < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Use pre-computed individual entropies
    double h_x = x_stat.entropy;
    double h_y = y_stat.entropy;

    // Re-discretize for joint calculation (in case different binning is needed)
    std::vector<int> x_bins = discretize(x, n, bins, method);
    std::vector<int> y_bins = discretize(y, n, bins, method);

    // Calculate joint entropy
    double h_xy = calcJointEntropy(x_bins, y_bins, n, bins);

    // Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    double mi = h_x + h_y - h_xy;
    
    return std::max(0.0, mi);
}

std::vector<double> MutualInfo::calcPvalues(const double *mi_values, size_t n_values,
                                            size_t n_samples, int n_permutations, int nthreads) {
    std::vector<double> p_values(n_values);
    
    // For mutual information, we typically use permutation tests
    // This is a simplified version - in practice, you might want more sophisticated testing
    
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < n_values; ++i) {
        if (std::isnan(mi_values[i])) {
            p_values[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Simple approximation: MI follows chi-square distribution under independence
            // This is a rough approximation and should be replaced with proper permutation tests
            double chi_sq = 2 * n_samples * mi_values[i];
            // For large n, chi-square with 1 degree of freedom
            p_values[i] = std::exp(-chi_sq / 2.0) / std::sqrt(2 * M_PI * chi_sq);
            p_values[i] = std::min(1.0, std::max(0.0, p_values[i]));
        }
    }
    
    return p_values;
}

std::vector<MutualInfoStat> MutualInfo::parallelGetMutualInfoStat(const Matrix<double> &X, 
                                                                   int nthreads, int bins,
                                                                   DiscretizationMethod method) {
    std::vector<MutualInfoStat> stats(X.rows());

#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t i = 0; i < X.rows(); ++i) {
        stats[i] = getMutualInfoStat(X.row(i), X.cols(), bins, method);
    }

    return stats;
}

std::vector<double> MutualInfo::normalizeMI(const double *mi_values, size_t n_values,
                                            const std::string &normalization_method) {
    std::vector<double> normalized(n_values);
    
    if (normalization_method == "none") {
        std::copy(mi_values, mi_values + n_values, normalized.begin());
        return normalized;
    }
    
    // Find min and max for min_max normalization
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (size_t i = 0; i < n_values; ++i) {
        if (!std::isnan(mi_values[i])) {
            min_val = std::min(min_val, mi_values[i]);
            max_val = std::max(max_val, mi_values[i]);
        }
    }
    
    if (normalization_method == "min_max") {
        double range = max_val - min_val;
        if (range < EPSILON) {
            std::fill(normalized.begin(), normalized.end(), 0.5);
        } else {
            for (size_t i = 0; i < n_values; ++i) {
                if (std::isnan(mi_values[i])) {
                    normalized[i] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    normalized[i] = (mi_values[i] - min_val) / range;
                }
            }
        }
    } else if (normalization_method == "z_score") {
        // Calculate mean and standard deviation
        double sum = 0.0, sum_sq = 0.0;
        int count = 0;
        
        for (size_t i = 0; i < n_values; ++i) {
            if (!std::isnan(mi_values[i])) {
                sum += mi_values[i];
                sum_sq += mi_values[i] * mi_values[i];
                count++;
            }
        }
        
        if (count > 0) {
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double std_dev = std::sqrt(std::max(0.0, variance));
            
            for (size_t i = 0; i < n_values; ++i) {
                if (std::isnan(mi_values[i])) {
                    normalized[i] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    normalized[i] = std_dev > EPSILON ? (mi_values[i] - mean) / std_dev : 0.0;
                }
            }
        } else {
            std::fill(normalized.begin(), normalized.end(), std::numeric_limits<double>::quiet_NaN());
        }
    }
    
    return normalized;
}

// Private helper methods

std::vector<int> MutualInfo::discretize(const double *data, size_t n, int bins,
                                        DiscretizationMethod method) {
    switch (method) {
        case DiscretizationMethod::EQUAL_WIDTH:
            return equalWidthDiscretize(data, n, bins);
        case DiscretizationMethod::EQUAL_FREQUENCY:
            return equalFreqDiscretize(data, n, bins);
        case DiscretizationMethod::ADAPTIVE:
            return adaptiveDiscretize(data, n, bins);
        default:
            return equalWidthDiscretize(data, n, bins);
    }
}

std::vector<int> MutualInfo::equalWidthDiscretize(const double *data, size_t n, int bins) {
    std::vector<int> discretized(n);
    
    // Find min and max values
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(data[i])) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }
    }
    
    // Handle constant data
    if (std::abs(max_val - min_val) < EPSILON) {
        std::fill(discretized.begin(), discretized.end(), 0);
        return discretized;
    }
    
    double bin_width = (max_val - min_val) / bins;
    
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) {
            discretized[i] = -1; // Special value for NaN
        } else {
            int bin = static_cast<int>((data[i] - min_val) / bin_width);
            bin = std::min(bin, bins - 1); // Ensure bin is within range
            bin = std::max(bin, 0);
            discretized[i] = bin;
        }
    }
    
    return discretized;
}

std::vector<int> MutualInfo::equalFreqDiscretize(const double *data, size_t n, int bins) {
    std::vector<int> discretized(n);
    std::vector<std::pair<double, size_t>> sorted_data(n);
    
    // Create pairs of (value, index) for sorting
    for (size_t i = 0; i < n; ++i) {
        sorted_data[i] = std::make_pair(data[i], i);
    }
    
    // Sort by value, handling NaN values
    std::sort(sorted_data.begin(), sorted_data.end(), 
              [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) {
                  if (std::isnan(a.first) && std::isnan(b.first)) return false;
                  if (std::isnan(a.first)) return false;
                  if (std::isnan(b.first)) return true;
                  return a.first < b.first;
              });
    
    // Assign bins based on quantiles
    int samples_per_bin = n / bins;
    int remainder = n % bins;
    
    for (size_t i = 0; i < n; ++i) {
        size_t rank = i;
        int bin = static_cast<int>(rank / samples_per_bin);
        
        // Adjust for remainder
        if (bin >= remainder) {
            rank -= remainder;
            bin = remainder + static_cast<int>(rank / samples_per_bin);
        }
        
        bin = std::min(bin, bins - 1);
        discretized[sorted_data[i].second] = bin;
    }
    
    return discretized;
}

std::vector<int> MutualInfo::adaptiveDiscretize(const double *data, size_t n, int bins) {
    // Adaptive binning using Sturges' formula as a starting point
    int adaptive_bins = static_cast<int>(1 + 3.322 * std::log10(n));
    adaptive_bins = std::min(adaptive_bins, bins);
    adaptive_bins = std::max(adaptive_bins, 2);
    
    return equalWidthDiscretize(data, n, adaptive_bins);
}

double MutualInfo::calcEntropy(const std::vector<int> &histogram, size_t total) {
    double entropy = 0.0;
    
    for (int count : histogram) {
        if (count > 0) {
            double p = static_cast<double>(count) / total;
            entropy -= p * std::log(p);
        }
    }
    
    return entropy;
}

double MutualInfo::calcJointEntropy(const std::vector<int> &x_bins, 
                                    const std::vector<int> &y_bins, size_t n, int bins) {
    std::vector<std::vector<int>> joint_hist = createJointHistogram(x_bins, y_bins, bins);
    
    double joint_entropy = 0.0;
    int valid_pairs = 0;
    
    for (const auto &row : joint_hist) {
        for (int count : row) {
            if (count > 0) {
                valid_pairs += count;
                double p = static_cast<double>(count) / n;
                joint_entropy -= p * std::log(p);
            }
        }
    }
    
    // If no valid pairs, return 0
    if (valid_pairs == 0) {
        return 0.0;
    }
    
    return joint_entropy;
}

std::vector<int> MutualInfo::createHistogram(const std::vector<int> &bins, int max_bins) {
    std::vector<int> histogram(max_bins, 0);
    
    for (int bin : bins) {
        if (bin >= 0 && bin < max_bins) {
            histogram[bin]++;
        }
    }
    
    return histogram;
}

std::vector<std::vector<int>> MutualInfo::createJointHistogram(const std::vector<int> &x_bins,
                                                                const std::vector<int> &y_bins,
                                                                int bins) {
    std::vector<std::vector<int>> joint_hist(bins, std::vector<int>(bins, 0));
    
    for (size_t i = 0; i < x_bins.size(); ++i) {
        int x_bin = x_bins[i];
        int y_bin = y_bins[i];
        
        if (x_bin >= 0 && x_bin < bins && y_bin >= 0 && y_bin < bins) {
            joint_hist[x_bin][y_bin]++;
        }
    }
    
    return joint_hist;
}

MutualInfoStat MutualInfo::getMutualInfoStat(const double *data, size_t n, int bins,
                                             DiscretizationMethod method) {
    std::vector<int> discretized = discretize(data, n, bins, method);
    std::vector<int> histogram = createHistogram(discretized, bins);
    double entropy = calcEntropy(histogram, n);
    
    return MutualInfoStat(discretized, histogram, entropy, n, bins);
}

double MutualInfo::permutationTest(const double *x, const double *y, size_t n,
                                   double original_mi, int n_permutations, int bins,
                                   DiscretizationMethod method) {
    std::vector<double> y_permuted(n);
    std::memcpy(y_permuted.data(), y, n * sizeof(double));
    
    int count_extreme = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int perm = 0; perm < n_permutations; ++perm) {
        // Shuffle y values
        std::shuffle(y_permuted.begin(), y_permuted.end(), gen);
        
        // Calculate MI for permuted data
        double perm_mi = calcMI(x, y_permuted.data(), n, bins, method);
        
        if (perm_mi >= original_mi) {
            count_extreme++;
        }
    }
    
    return static_cast<double>(count_extreme + 1) / (n_permutations + 1);
} 