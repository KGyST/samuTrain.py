import requests
import json

def test_samutrain_system():
    """Test the complete samuTrain V2 system"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing samuTrain V2 System")
    print("=" * 40)
    
    # Test creating multiple cases with different confidence levels
    test_cases = [
        {'image_path': 'data/test_high_conf.png', 'ocr_text': 'High confidence text', 'confidence': 0.95, 'is_failset': False},
        {'image_path': 'data/test_low_conf.png', 'ocr_text': 'Low confidence text', 'confidence': 0.3, 'is_failset': True},
        {'image_path': 'data/test_medium_conf.png', 'ocr_text': 'Medium confidence text', 'confidence': 0.7, 'is_failset': False}
    ]
    
    print("\n📝 Creating test cases...")
    for i, case in enumerate(test_cases, 1):
        r = requests.post(f"{base_url}/api/cases", json=case)
        case_id = r.json().get("case_id")
        print(f"  Case {i}: Status {r.status_code}, ID: {case_id}")
    
    # Test statistics
    print("\n📊 System Statistics:")
    stats = requests.get(f"{base_url}/api/statistics").json()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test filtering
    print("\n🔍 Case Filtering:")
    all_cases = requests.get(f"{base_url}/api/cases").json()
    failset_cases = requests.get(f"{base_url}/api/cases?failset_only=true").json()
    
    print(f"  All cases count: {len(all_cases)}")
    print(f"  Failset only count: {len(failset_cases)}")
    
    # Test case correction
    if all_cases:
        case_id = all_cases[0]['id']
        print(f"\n✏️  Testing case correction for ID {case_id}")
        correction_data = {'corrected_text': 'Corrected text'}
        r = requests.post(f"{base_url}/api/cases/{case_id}/correct", json=correction_data)
        print(f"  Correction status: {r.status_code}")
        print(f"  Response: {r.json()}")
    
    print("\n✅ System test completed!")
    print(f"🌐 UI available at: {base_url}")
    print(f"📚 API docs at: {base_url}/docs")

if __name__ == "__main__":
    test_samutrain_system()
