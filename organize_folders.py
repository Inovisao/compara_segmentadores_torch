import os
import shutil

# Define source directory and target directories
# SOURCE_DIR = "./data/dobras/fold_1"
SOURCE_DIR = "./data_1024_UCDB_UPS/all/imagens/"
UCDB_DIR = "./data_1024_UCDB_UPS/ucdb/"
UPS_DIR = "./data_1024_UCDB_UPS/ups/"

# Create target directories if they don't exist
os.makedirs(UCDB_DIR, exist_ok=True)
os.makedirs(UPS_DIR, exist_ok=True)

# Move files starting with UCDB to UCDB_DIR
for file_name in os.listdir(SOURCE_DIR):
    if file_name.startswith("UCDB_") and file_name.endswith(".jpg"):
        shutil.move(
            os.path.join(SOURCE_DIR, file_name), os.path.join(UCDB_DIR, file_name)
        )

# Move files starting with UPS to UPS_DIR
for file_name in os.listdir(SOURCE_DIR):
    if file_name.startswith("UPS_") and file_name.endswith(".jpg"):
        shutil.move(
            os.path.join(SOURCE_DIR, file_name), os.path.join(UPS_DIR, file_name)
        )

print("Files moved successfully!")
