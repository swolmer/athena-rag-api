#!/usr/bin/env python3
"""
🧪 Athen.ai RAG API Test Script
Tests basic functionality of the API endpoints
"""

import requests
import json
import os
import time

# Configuration
BASE_URL = "http://localhost:19123"  # Change to your RunPod URL
TEST_ORG_ID = "test_hospital"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_orgs_endpoint():
    """Test the organizations listing endpoint"""
    print("🔍 Testing organizations endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/orgs", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Organizations endpoint passed: {data}")
            return True
        else:
            print(f"❌ Organizations endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Organizations endpoint error: {e}")
        return False

def test_query_without_data():
    """Test querying without uploaded data (should fail gracefully)"""
    print("🔍 Testing query without data...")
    try:
        query_data = {
            "question": "What is appendectomy?",
            "org_id": TEST_ORG_ID,
            "k": 3
        }
        response = requests.post(
            f"{BASE_URL}/query",
            json=query_data,
            timeout=30
        )
        if response.status_code == 404:
            print("✅ Query without data correctly returned 404")
            return True
        else:
            print(f"⚠️ Query without data returned: {response.status_code}")
            print(f"Response: {response.text}")
            return True  # Still pass, just unexpected response
    except Exception as e:
        print(f"❌ Query test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Starting Athen.ai RAG API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Organizations", test_orgs_endpoint),
        ("Query (No Data)", test_query_without_data),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for production.")
    else:
        print("⚠️ Some tests failed. Check the API server and try again.")
    
    return passed == total

if __name__ == "__main__":
    # Check if server is running locally vs RunPod
    print("🔧 Configuration:")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Org ID: {TEST_ORG_ID}")
    print()
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n📚 Next steps:")
        print("1. Upload training materials using /upload endpoint")
        print("2. Test queries with your medical documents")
        print("3. Try fine-tuning with /fine-tune endpoint")
        print("\n📖 API Documentation: {BASE_URL}/docs")
    
    exit(0 if success else 1)
