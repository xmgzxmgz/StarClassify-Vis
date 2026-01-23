import time
import requests
import json
import sys
from pathlib import Path

BASE_API = "http://localhost:8000/api"
BASE_WEB = "http://localhost:8080"

def wait_for_health(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_API}/health")
            if r.status_code == 200:
                print("✅ Backend is healthy")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print("Waiting for backend...", end="\r")
    print("\n❌ Backend health check timed out")
    sys.exit(1)

def test_web_accessibility():
    try:
        r = requests.get(BASE_WEB)
        if r.status_code == 200 and "<html" in r.text.lower():
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend check failed: {r.status_code}")
    except Exception as e:
        print(f"❌ Frontend check error: {e}")

def test_api_flow():
    # 1. Upload and Analyze
    csv_path = Path("datasets/sdss_like_small.csv")
    if not csv_path.exists():
        print(f"❌ Test dataset not found: {csv_path}")
        return

    payload = {
        "datasetName": "Test Auto Run",
        "targetColumn": "class",
        "featureColumns": ["u", "g", "r", "i", "z"],
        "testSize": 0.2,
        "randomState": 42,
        "modelType": "gaussian_nb",
        "gnbParams": {"varSmoothing": 1e-9}
    }

    print("🚀 Starting analysis run...")
    try:
        files = {'file': ('sdss_like_small.csv', open(csv_path, 'rb'), 'text/csv')}
        r = requests.post(
            f"{BASE_API}/runs",
            data={'payload': json.dumps(payload)},
            files=files
        )
        if r.status_code != 200:
            print(f"❌ Analysis failed: {r.text}")
            return
        
        result = r.json()
        run_id = result['id']
        acc = result['metrics']['accuracy']
        print(f"✅ Analysis complete. Run ID: {run_id}, Accuracy: {acc}")

        # 2. List Runs
        r = requests.get(f"{BASE_API}/runs")
        if r.status_code == 200:
            total = r.json()['total']
            print(f"✅ List runs success. Total runs: {total}")
        else:
            print(f"❌ List runs failed: {r.status_code}")

        # 3. Get Run Detail
        r = requests.get(f"{BASE_API}/runs/{run_id}")
        if r.status_code == 200:
            print("✅ Get run detail success")
        else:
            print(f"❌ Get run detail failed: {r.status_code}")

    except Exception as e:
        print(f"❌ API test error: {e}")

if __name__ == "__main__":
    print("Waiting for services to start...")
    wait_for_health()
    test_web_accessibility()
    test_api_flow()
