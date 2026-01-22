import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():
    file_id = '1xYasjU52whXMLT5MtF7RCPQkV66993oR'
    dest_folder = os.path.join(os.path.dirname(__file__), 'model-weights')
    os.makedirs(dest_folder, exist_ok=True)
    destination = os.path.join(dest_folder, 'yolov3-wider_16000.weights')
    if not os.path.exists(destination):
        print(f"Downloading weights to {destination}...")
        download_file_from_google_drive(file_id, destination)
        print("Download complete.")
    else:
        print("Weights file already exists.")

if __name__ == "__main__":
    main()
