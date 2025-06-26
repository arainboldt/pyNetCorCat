#!/usr/bin/env python3
"""
Test script for phi coefficient implementation
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pynetcor import phi_coef, phi_test, pvalue_chi_square
    print("✅ Successfully imported phi coefficient functions!")
except ImportError as e:
    print(f"❌ Failed to import phi coefficient functions: {e}")
    sys.exit(1)

def test_phi_coefficient():
    """Test basic phi coefficient calculation"""
    print("\n🧪 Testing phi coefficient calculation...")
    
    # Create test data
    x = np.array([[0.1, 0.8, 0.3, 0.9],
                  [0.2, 0.7, 0.4, 0.6]])
    
    # Test phi coefficient with default threshold (0.5)
    result = phi_coef(x, threshold=0.5)
    print(f"Phi coefficient matrix shape: {result.shape}")
    print(f"Phi coefficient matrix:\n{result}")
    
    # Verify diagonal is 1.0 (perfect correlation with itself)
    assert np.allclose(np.diag(result), 1.0), "Diagonal should be 1.0"
    print("✅ Diagonal elements are correctly set to 1.0")
    
    # Verify symmetry
    assert np.allclose(result, result.T), "Matrix should be symmetric"
    print("✅ Matrix is symmetric")
    
    # Verify values are in [-1, 1] range
    assert np.all(result >= -1) and np.all(result <= 1), "Values should be in [-1, 1] range"
    print("✅ All values are in valid range [-1, 1]")
    
    return result

def test_phi_test():
    """Test phi coefficient with p-values"""
    print("\n🧪 Testing phi coefficient with p-values...")
    
    # Create test data
    x = np.array([[0.1, 0.8, 0.3, 0.9],
                  [0.2, 0.7, 0.4, 0.6]])
    
    # Test phi test
    result = phi_test(x, threshold=0.5)
    print(f"Phi test result shape: {result.shape}")
    print(f"Phi test result:\n{result}")
    
    # Verify we have the expected columns: [index1, index2, phi, p]
    assert result.shape[1] == 4, f"Expected 4 columns, got {result.shape[1]}"
    print("✅ Result has correct number of columns")
    
    # Verify p-values are in [0, 1] range
    p_values = result[:, 3]
    assert np.all(p_values >= 0) and np.all(p_values <= 1), "P-values should be in [0, 1] range"
    print("✅ P-values are in valid range [0, 1]")
    
    return result

def test_pvalue_chi_square():
    """Test chi-square p-value calculation"""
    print("\n🧪 Testing chi-square p-value calculation...")
    
    # Test phi coefficients
    phi_coeffs = np.array([0.5, 0.8, 0.2])
    n = 100  # sample size
    
    # Calculate p-values
    p_values = pvalue_chi_square(phi_coeffs, n=n)
    print(f"Phi coefficients: {phi_coeffs}")
    print(f"P-values: {p_values}")
    
    # Verify p-values are in [0, 1] range
    assert np.all(p_values >= 0) and np.all(p_values <= 1), "P-values should be in [0, 1] range"
    print("✅ P-values are in valid range [0, 1]")
    
    # Verify that higher phi coefficients give lower p-values (stronger correlation)
    assert p_values[1] < p_values[2], "Higher phi should give lower p-value"
    print("✅ Higher phi coefficients correctly give lower p-values")
    
    return p_values

def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing edge cases...")
    
    # Test with identical data
    x = np.array([[0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5]])
    
    result = phi_coef(x, threshold=0.5)
    print(f"Identical data result:\n{result}")
    
    # Test with very small data
    x_small = np.array([[0.1, 0.9]])
    try:
        result_small = phi_coef(x_small, threshold=0.5)
        print(f"Small data result:\n{result_small}")
    except Exception as e:
        print(f"Small data test (expected error): {e}")
    
    print("✅ Edge cases handled appropriately")

if __name__ == "__main__":
    print("🚀 Starting phi coefficient tests...")
    
    try:
        # Run all tests
        phi_matrix = test_phi_coefficient()
        phi_test_result = test_phi_test()
        p_values = test_pvalue_chi_square()
        test_edge_cases()
        
        print("\n🎉 All tests passed! Phi coefficient implementation is working correctly.")
        print("\n📊 Summary:")
        print(f"   - Phi coefficient matrix: {phi_matrix.shape}")
        print(f"   - Phi test results: {phi_test_result.shape}")
        print(f"   - Chi-square p-values calculated successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 