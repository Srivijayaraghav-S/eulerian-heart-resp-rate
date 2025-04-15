import requests
import cv2
from urllib.request import urlopen

IP = "192.168.116.119"
PORT = "8080"
URL = f"http://{IP}:{PORT}"

# Test basic connection
try:
    response = urlopen(f"{URL}/", timeout=3)
    print(f"✅ Can reach phone at {URL}")
    print(f"Response: {response.read(100)}")  # First 100 bytes
except Exception as e:
    print(f"❌ Connection failed: {e}")