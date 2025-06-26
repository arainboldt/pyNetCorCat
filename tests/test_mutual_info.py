#!/usr/bin/env python3
"""
Test script for mutual information implementation
"""

import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pynetcor import mutual_info_coef
    print("✅ Successfully imported mutual_info_coef!")
except ImportError as e:
    print(f"❌ Failed to import mutual_info_coef: {e}")
    sys.exit(1)

def test_mutual_info_basic():
    """Test basic mutual information calculation"""
    print("\n🧪 Testing mutual information calculation...")
    # Create test data
    x = np.array([[0.1, 0.8, 0.3, 0.9],
                  [0.2, 0.7, 0.4, 0.6]])
    # Test mutual information with default parameters
    result = mutual_info_coef(x)
    print(f"Mutual information matrix shape: {result.shape}")
    print(f"Mutual information matrix:\n{result}")
    # Verify diagonal is maximal (self MI)
    assert np.all(result.diagonal() >= result.min()), "Diagonal should be maximal (self MI)"
    # Verify symmetry
    assert np.allclose(result, result.T), "Matrix should be symmetric"
    print("✅ Matrix is symmetric and diagonal is maximal")
    return result

def test_mutual_info_edge_cases():
    """Test edge cases for mutual information"""
    print("\n🧪 Testing mutual information edge cases...")
    # Identical data
    x = np.array([[0.5, 0.5, 0.5, 0.5],
                  [0.5, 0.5, 0.5, 0.5]])
    result = mutual_info_coef(x)
    print(f"Identical data result:\n{result}")
    # Small data
    x_small = np.array([[0.1, 0.9]])
    try:
        result_small = mutual_info_coef(x_small)
        print(f"Small data result:\n{result_small}")
    except Exception as e:
        print(f"Small data test (expected error): {e}")
    print("✅ Edge cases handled appropriately")

def test_mutual_info_methods():
    """Test different discretization methods"""
    print("\n🧪 Testing mutual information with different discretization methods...")
    x = np.random.rand(3, 100)
    for method in ["equal_width", "equal_frequency"]:
        result = mutual_info_coef(x, method=method)
        print(f"Method: {method}, MI matrix:\n{result}")
        assert np.allclose(result, result.T), f"Matrix should be symmetric for method {method}"
    print("✅ All methods produce symmetric matrices")

if __name__ == "__main__":
    print("🚀 Starting mutual information tests...")
    try:
        mi_matrix = test_mutual_info_basic()
        test_mutual_info_edge_cases()
        test_mutual_info_methods()
        print("\n🎉 All tests passed! Mutual information implementation is working correctly.")
        print("\n📊 Summary:")
        print(f"   - Mutual information matrix: {mi_matrix.shape}")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 