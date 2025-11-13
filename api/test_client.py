#!/usr/bin/env python3
"""
Simple test client for the Hierarchical Data Simulator API.

Tests all major endpoints and downloads sample data.
"""

import requests
import json
import sys
from pathlib import Path

API_URL = "http://localhost:8000"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def test_health():
    """Test health check endpoint."""
    print_section("Testing Health Check")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Status: {data['status']}")
        print(f"✓ Service: {data['service']}")
        print(f"✓ Timestamp: {data['timestamp']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_quick_simulate():
    """Test quick simulation endpoint."""
    print_section("Testing Quick Simulation")
    
    try:
        payload = {
            "outcome_type": "continuous",
            "n_groups": 10,
            "size_range": [5, 15],
            "random_seed": 42
        }
        
        print(f"Request: {json.dumps(payload, indent=2)}")
        response = requests.post(f"{API_URL}/api/v1/simulate/quick", json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Simulation ID: {data['metadata']['simulation_id']}")
        print(f"✓ Outcome Type: {data['metadata']['outcome_type']}")
        print(f"✓ Total Observations: {data['metadata']['total_observations']}")
        print(f"✓ Number of Groups: {data['metadata']['n_groups']}")
        print(f"\nPreview (first 3 rows):")
        
        for i, row in enumerate(data['preview'][:3]):
            print(f"  Row {i+1}: {row}")
        
        return True
    except Exception as e:
        print(f"✗ Quick simulation failed: {e}")
        return False


def test_detailed_simulate():
    """Test detailed simulation endpoint."""
    print_section("Testing Detailed Simulation")
    
    try:
        payload = {
            "outcome_type": "binary",
            "link_function": "logit",
            "gamma_00": 0.0,
            "gamma_10": 0.5,
            "tau_00": 1.0,
            "tau_11": 0.3,
            "tau_01": 0.1,
            "n_groups": 15,
            "size_range": [10, 20],
            "predictor_range": [0.0, 1.0],
            "random_seed": 123
        }
        
        print(f"Request parameters:")
        print(f"  Outcome: {payload['outcome_type']}")
        print(f"  Link: {payload['link_function']}")
        print(f"  γ₀₀: {payload['gamma_00']}, γ₁₀: {payload['gamma_10']}")
        print(f"  τ₀₀: {payload['tau_00']}, τ₁₁: {payload['tau_11']}, τ₀₁: {payload['tau_01']}")
        
        response = requests.post(f"{API_URL}/api/v1/simulate/detailed", json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Simulation successful!")
        print(f"✓ Total Observations: {data['metadata']['total_observations']}")
        print(f"✓ Columns: {', '.join(data['columns'])}")
        
        # Count binary outcomes
        outcomes = [row['outcome'] for row in data['preview']]
        print(f"\nBinary outcome distribution in preview:")
        print(f"  0s: {outcomes.count(0)}")
        print(f"  1s: {outcomes.count(1)}")
        
        return True
    except Exception as e:
        print(f"✗ Detailed simulation failed: {e}")
        return False


def test_download():
    """Test data download in different formats."""
    print_section("Testing Data Download")
    
    output_dir = Path("test_downloads")
    output_dir.mkdir(exist_ok=True)
    
    payload = {
        "outcome_type": "count",
        "link_function": "log",
        "gamma_00": 1.5,
        "gamma_10": 0.5,
        "tau_00": 0.5,
        "tau_11": 0.2,
        "tau_01": 0.0,
        "dispersion": 1.0,
        "n_groups": 10,
        "size_range": [5, 10],
        "predictor_range": [0.0, 1.0],
        "random_seed": 456
    }
    
    formats = ["csv", "json", "excel"]
    success_count = 0
    
    for fmt in formats:
        try:
            print(f"\nDownloading {fmt.upper()}...", end=" ")
            response = requests.post(
                f"{API_URL}/api/v1/simulate/download/{fmt}",
                json=payload
            )
            response.raise_for_status()
            
            output_file = output_dir / f"test_data.{fmt}"
            output_file.write_bytes(response.content)
            
            file_size = output_file.stat().st_size
            print(f"✓ Saved to {output_file} ({file_size} bytes)")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print(f"\n{'✓' if success_count == len(formats) else '✗'} Downloaded {success_count}/{len(formats)} formats")
    return success_count == len(formats)


def test_defaults():
    """Test getting default parameters."""
    print_section("Testing Default Parameters")
    
    outcome_types = ["continuous", "binary", "count", "survival"]
    
    for outcome_type in outcome_types:
        try:
            response = requests.get(
                f"{API_URL}/api/v1/parameters/defaults/{outcome_type}"
            )
            response.raise_for_status()
            data = response.json()
            
            print(f"\n{outcome_type.upper()}:")
            defaults = data['default_parameters']
            print(f"  Link function: {defaults['link_function']}")
            print(f"  γ₀₀: {defaults['gamma_00']}, γ₁₀: {defaults['gamma_10']}")
            print(f"  τ₀₀: {defaults['tau_00']}, τ₁₁: {defaults['tau_11']}")
            
        except Exception as e:
            print(f"✗ Failed to get defaults for {outcome_type}: {e}")
            return False
    
    return True


def test_info():
    """Test outcome types info endpoint."""
    print_section("Testing Outcome Types Info")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/info/outcome-types")
        response.raise_for_status()
        data = response.json()
        
        print(f"\n✓ Found {len(data['outcome_types'])} outcome types:")
        for ot in data['outcome_types']:
            print(f"\n  {ot['type'].upper()}")
            print(f"    Description: {ot['description']}")
            print(f"    Link functions: {', '.join(ot['link_functions'])}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to get outcome types info: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  🎲 Hierarchical Data Simulator API Test Suite")
    print("=" * 60)
    print(f"\nTesting API at: {API_URL}")
    
    # Check if server is running
    try:
        requests.get(API_URL, timeout=2)
    except requests.exceptions.RequestException:
        print("\n❌ ERROR: API server is not running!")
        print(f"   Please start the server first:")
        print(f"   cd api && python main.py")
        sys.exit(1)
    
    # Run tests
    results = {
        "Health Check": test_health(),
        "Quick Simulation": test_quick_simulate(),
        "Detailed Simulation": test_detailed_simulate(),
        "Data Download": test_download(),
        "Default Parameters": test_defaults(),
        "Outcome Types Info": test_info(),
    }
    
    # Print summary
    print_section("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    print()
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} tests passed")
    print('=' * 60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
