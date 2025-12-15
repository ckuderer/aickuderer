import requests
import json

url = f"https://addressvalidation.googleapis.com/v1:validateAddress?key=AIzaSyD-TG31OkgHEmZYK6L9HOw66hSDQugWAao"

payload = {
    "address": {
        "regionCode": "US",
        "locality": "Mountain View",
        "addressLines": ["1600 Amphitheatre Pkwy"]
    }
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
