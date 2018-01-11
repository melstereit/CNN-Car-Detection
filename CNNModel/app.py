import subprocess
from carmodel import Model

process = subprocess.Popen("./data/get_datasets.sh", shell=True, stdout=subprocess.PIPE)
process.wait()
print("Download images - this might take a while")
if process.returncode == 0:
    model = Model()
    model.preprocess_and_build_model()
else:
    print("Could not download files")
