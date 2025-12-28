import requests

url = "http://127.0.0.1:5000/predict"

with open("slice_008.npy", "rb") as f:
    files = {"file": f}
    r = requests.post(url, files=files)

print(r.json())
