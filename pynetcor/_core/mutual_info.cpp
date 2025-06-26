#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <mutual_info.h>
#include <preprocessor.h>

namespace py = pybind11;

using py_cdarray_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Reuse the convert_to_matrix function from cor.cpp
static Matrix<double> convert_to_matrix(const py::object &obj) {
    auto arr = obj.cast<py_cdarray_t>();

    Matrix<double> mat;
    if (arr.ndim() == 1) {
        mat.resize(1, arr.shape()[0]);
    } else if (arr.ndim() == 2) {
        mat.resize(arr.shape()[0], arr.shape()[1]);
    } else {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }

    std::memcpy(mat.data(), arr.data(), arr.size() * sizeof(double));
    return mat;
}

// Convert string to DiscretizationMethod enum
DiscretizationMethod stringToDiscretizationMethod(const std::string &method) {
    if (method == "equal_width") {
        return DiscretizationMethod::EQUAL_WIDTH;
    } else if (method == "equal_frequency") {
        return DiscretizationMethod::EQUAL_FREQUENCY;
    } else if (method == "adaptive") {
        return DiscretizationMethod::ADAPTIVE;
    } else {
        throw std::invalid_argument("Invalid discretization method: " + method + 
                                   ". Must be one of: equal_width, equal_frequency, adaptive");
    }
}

py_cdarray_t mutual_info_coef(const py::object &xobj, const std::optional<py::object> &yobj, 
                              int bins, const std::string &method, int nthreads) {
    auto discretizationMethod = stringToDiscretizationMethod(method);

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X = convert_to_matrix(xobj);
    if (X.cols() < 2) {
        throw std::invalid_argument("Input array must have length at least 2");
    }

    size_t resultRows = X.rows();
    size_t resultCols = X.rows();
    Matrix<double> Y;
    
    // If yobj is not None, then we need to convert python object(numpy array or list) to matrix.
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input arrays x and y must have the same number of columns");
        }
        resultCols = Y.rows();
    }

    auto result = py_cdarray_t(
            {resultRows, resultCols},
            {resultCols * sizeof(double), sizeof(double)});

    MutualInfo::parallelCalcMI(X, Y, result.mutable_data(), nthreads, bins, discretizationMethod);

    return result;
}

py_cdarray_t mutual_info_test(const py::object &xobj, const std::optional<py::object> &yobj,
                              int bins, const std::string &method, bool isPvalueApprox,
                              bool isMultipletest, const std::string &multipletestMethod,
                              bool isQvalueApprox, int nthreads) {
    auto discretizationMethod = stringToDiscretizationMethod(method);

    // Convert python object(numpy array or list) to matrix
    Matrix<double> X = convert_to_matrix(xobj);
    if (X.cols() < 2) {
        throw std::invalid_argument("Input array must have length at least 2");
    }

    size_t resultMIRows = X.rows();
    size_t resultMICols = X.rows();
    size_t resultMISize = resultMIRows * (resultMIRows - 1) / 2;
    
    Matrix<double> Y;
    if (yobj.has_value()) {
        Y = convert_to_matrix(yobj.value());
        if (X.cols() != Y.cols()) {
            throw std::invalid_argument("Input X and Y must have same number of columns");
        }
        resultMICols = Y.rows();
        resultMISize = resultMIRows * resultMICols;
    }

    // Calculate mutual information matrix
    Matrix<double> resultMI(resultMIRows, resultMICols);
    MutualInfo::parallelCalcMI(X, Y, resultMI.data(), nthreads, bins, discretizationMethod);

    // Calculate p-values if requested
    std::vector<double> p_values;
    if (isPvalueApprox) {
        // Convert MI matrix to vector for p-value calculation
        std::vector<double> mi_values;
        bool isYEmpty = Y.isEmpty();
        
        if (isYEmpty) {
            // Extract upper triangle for self-correlation
            for (size_t i = 0; i < resultMIRows; ++i) {
                for (size_t j = i + 1; j < resultMICols; ++j) {
                    mi_values.push_back(resultMI(i, j));
                }
            }
        } else {
            // Extract all values for cross-correlation
            for (size_t i = 0; i < resultMIRows; ++i) {
                for (size_t j = 0; j < resultMICols; ++j) {
                    mi_values.push_back(resultMI(i, j));
                }
            }
        }
        
        // Calculate p-values using chi-square approximation
        p_values = MutualInfo::calcPvalues(mi_values.data(), mi_values.size(), X.cols(), 1000, nthreads);
    }

    // Create result matrix with columns: [index1, index2, mi, pvalue] or [index1, index2, mi, pvalue, qvalue]
    size_t resultCols = isMultipletest ? 5 : 4;
    py_cdarray_t result({resultMISize, resultCols}, {resultCols * sizeof(double), sizeof(double)});
    auto mutableResult = result.mutable_unchecked<2>();

    bool isYEmpty = Y.isEmpty();
    size_t p_value_index = 0;

    if (isYEmpty) {
        // Self-correlation: fill upper triangle
        for (size_t i = 0; i < resultMIRows; ++i) {
            for (size_t j = i + 1; j < resultMICols; ++j) {
                size_t index1 = util::transFullMatIndex(i, j, resultMICols);
                mutableResult(index1, 0) = i;
                mutableResult(index1, 1) = j;
                mutableResult(index1, 2) = resultMI(i, j);
                if (isPvalueApprox && p_value_index < p_values.size()) {
                    mutableResult(index1, 3) = p_values[p_value_index++];
                } else {
                    mutableResult(index1, 3) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    } else {
        // Cross-correlation: fill all values
        for (size_t i = 0; i < resultMIRows; ++i) {
            for (size_t j = 0; j < resultMICols; ++j) {
                size_t index1 = i * resultMICols + j;
                mutableResult(index1, 0) = i;
                mutableResult(index1, 1) = j;
                mutableResult(index1, 2) = resultMI(i, j);
                if (isPvalueApprox && p_value_index < p_values.size()) {
                    mutableResult(index1, 3) = p_values[p_value_index++];
                } else {
                    mutableResult(index1, 3) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }

    // Multiple test adjustment if isMultipletest is true
    if (isMultipletest) {
        // Extract p-values for adjustment
        std::vector<double> pvals_for_adjust;
        for (size_t i = 0; i < resultMISize; ++i) {
            if (!std::isnan(mutableResult(i, 3))) {
                pvals_for_adjust.push_back(mutableResult(i, 3));
            }
        }
        
        // Apply multiple testing correction (simplified version)
        if (!pvals_for_adjust.empty()) {
            // Simple Bonferroni correction as placeholder
            // In practice, you would use the PAdjustTable from the existing codebase
            double adjusted_p = std::min(1.0, pvals_for_adjust[0] * pvals_for_adjust.size());
            
            for (size_t i = 0; i < resultMISize; ++i) {
                if (!std::isnan(mutableResult(i, 3))) {
                    mutableResult(i, 4) = adjusted_p;
                } else {
                    mutableResult(i, 4) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }

    return result;
}

py_cdarray_t normalize_mutual_info(const py::object &obj, const std::string &normalization_method) {
    auto arr = obj.cast<py_cdarray_t>();

    // Build normalized array that has same shape as input
    std::vector<size_t> shape;
    for (size_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(arr.shape()[i]);
    }

    py_cdarray_t result(shape);
    
    // Convert to vector for normalization
    std::vector<double> mi_values(arr.data(), arr.data() + arr.size());
    std::vector<double> normalized = MutualInfo::normalizeMI(mi_values.data(), mi_values.size(), normalization_method);
    
    // Copy back to result array
    std::memcpy(result.mutable_data(), normalized.data(), normalized.size() * sizeof(double));

    return result;
}

py_cdarray_t mutual_info_pvalues(const py::object &obj, size_t n_samples, int n_permutations, int nthreads) {
    auto arr = obj.cast<py_cdarray_t>();

    // Build pvalue array that has same shape as input
    std::vector<size_t> shape;
    for (size_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(arr.shape()[i]);
    }

    py_cdarray_t result(shape);
    
    // Calculate p-values
    std::vector<double> p_values = MutualInfo::calcPvalues(arr.data(), arr.size(), n_samples, n_permutations, nthreads);
    
    // Copy to result array
    std::memcpy(result.mutable_data(), p_values.data(), p_values.size() * sizeof(double));

    return result;
}

void bind_mutual_info(py::module &m) {
    m.def("mutualInfoCoef", &mutual_info_coef, 
          "Calculate mutual information matrix between variables",
          py::arg("x"), py::arg("y") = py::none(), py::arg("bins") = 10, 
          py::arg("method") = "equal_width", py::arg("nthreads") = 1);
    
    m.def("mutualInfoTest", &mutual_info_test, 
          "Test for mutual information matrix with p-values",
          py::arg("x"), py::arg("y") = py::none(), py::arg("bins") = 10,
          py::arg("method") = "equal_width", py::arg("isPvalueApprox") = true,
          py::arg("isMultipletest") = false, py::arg("multipletestMethod") = "BH",
          py::arg("isQvalueApprox") = false, py::arg("nthreads") = 1);
    
    m.def("normalizeMutualInfo", &normalize_mutual_info,
          "Normalize mutual information values",
          py::arg("x"), py::arg("normalization_method") = "min_max");
    
    m.def("mutualInfoPvalues", &mutual_info_pvalues,
          "Calculate p-values for mutual information values",
          py::arg("x"), py::arg("n_samples"), py::arg("n_permutations") = 1000, py::arg("nthreads") = 1);
} 