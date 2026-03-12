import http.server
import json
import urllib.parse
import os
import mimetypes

# 2-space indentation. English comments.
class TestHandler(http.server.SimpleHTTPRequestHandler):
  def do_GET(self):
    # Parse query parameters
    parsed_path = urllib.parse.urlparse(self.path)
    query_params = urllib.parse.parse_qs(parsed_path.query)
    
    if self.path.startswith('/api/cases'):
      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.end_headers()
      # Test case data for frontend display
      test_data = [{
        "id": 1,
        "img_path": "64_case/01000a.bin.png",
        "ocr_text": "Sample OCR Prediction",
        "model_prediction": "Sample OCR Prediction",
        "gt_text": "Ground Truth Sample Text",
        "confidence": 0.85,
        "timestamp": "2026-03-02T14:50:00",
        "is_corrected": False,
        "is_failset": False
      }]
      self.wfile.write(json.dumps(test_data).encode())
    
    elif self.path.startswith('/api/statistics'):
      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.end_headers()
      # Test statistics for frontend
      stats_data = {
        "total_cases": 1,
        "failset_cases": 0,
        "avg_confidence": 0.85,
        "recent_activity": 1
      }
      self.wfile.write(json.dumps(stats_data).encode())
    
    elif self.path.startswith('/api/training/status'):
      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.end_headers()
      # Test training status
      training_data = {
        "is_training": False,
        "training_status": "idle",
        "session_id": None,
        "current_epoch": 0,
        "db_state": None
      }
      self.wfile.write(json.dumps(training_data).encode())
    
    elif self.path.startswith('/static/'):
      # Serve static files (images) from data directory
      file_path = self.path[len('/static/'):]  # Remove /static/ prefix
      data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
      full_path = os.path.join(data_dir, file_path)
      
      if os.path.exists(full_path) and os.path.isfile(full_path):
        # Determine content type
        content_type, _ = mimetypes.guess_type(full_path)
        if not content_type:
          content_type = 'application/octet-stream'
        
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        with open(full_path, 'rb') as f:
          self.wfile.write(f.read())
      else:
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'File not found')
    
    else:
      # Serve static files for frontend
      return super().do_GET()

  def do_POST(self):
    if self.path == '/api/cases':
      self.send_response(200)
      self.send_header('Content-Type', 'application/json')
      self.send_header('Access-Control-Allow-Origin', '*')
      self.end_headers()
      # Mock successful case creation
      response = {
        "success": True,
        "case_id": 999,
        "message": "Case created successfully with ID 999"
      }
      self.wfile.write(json.dumps(response).encode())
    else:
      self.send_response(404)
      self.end_headers()

if __name__ == "__main__":
  os.chdir('src/static')
  print("Test server: http://localhost:8000")
  print("Endpoints available:")
  print("  GET /api/cases - Returns test cases")
  print("  GET /api/statistics - Returns test statistics")
  print("  GET /api/training/status - Returns training status")
  print("  POST /api/cases - Creates new test case")
  http.server.HTTPServer(('localhost', 8000), TestHandler).serve_forever()
