#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("Testing /api/statistics:")
resp = client.get('/api/statistics')
print(f"Status: {resp.status_code}")
print(f"Data: {resp.json()}")

print("\nTesting /api/cases:")
resp2 = client.get('/api/cases')
print(f"Status: {resp2.status_code}")
print(f"Cases count: {len(resp2.json())}")
if resp2.json():
    print(f"First case: {resp2.json()[0]}")
else:
    print("No cases found")
