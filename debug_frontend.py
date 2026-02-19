#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("=== Testing API endpoints directly ===")

# Test statistics
print("\n1. Testing /api/statistics:")
resp = client.get('/api/statistics')
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")

# Test cases
print("\n2. Testing /api/cases:")
resp2 = client.get('/api/cases')
print(f"Status: {resp2.status_code}")
cases = resp2.json()
print(f"Cases count: {len(cases)}")

if cases:
    case = cases[0]
    print(f"First case keys: {list(case.keys())}")
    print(f"First case: {case}")
    
    # Check for required fields
    required_fields = ['id', 'img_path', 'ocr_text', 'gt_text', 'confidence', 'timestamp']
    missing = [f for f in required_fields if f not in case]
    if missing:
        print(f"❌ Missing fields: {missing}")
    else:
        print("✅ All required fields present")

print("\n=== Testing case correction endpoint ===")
resp3 = client.post('/api/cases/1/correct', json={"corrected_text": "test"})
print(f"Correction status: {resp3.status_code}")
print(f"Correction response: {resp3.json()}")
