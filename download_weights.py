import os
import gdown

# Google Drive file ID
file_id = "1xYasjU52whXMLT5MtF7RCPQkV66993oR"
output_dir = "model-weights"
output_file = os.path.join(output_dir, "yolov3-wider_16000.weights")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download the file
url = f"https://drive.google.com/uc?id={file_id}"
print(f"Downloading weights to {output_file} ...")
gdown.download(url, output_file, quiet=False)

print("Download complete.")