import subprocess
from carmodel import Model

print("Download images - this might take a while")
process = subprocess.Popen("./data/get_datasets.sh", shell=True, stdout=subprocess.PIPE)
process.wait()
print("Finished")
if process.returncode == 0:
    model = Model()
    model.preprocess_and_build_model()
else:
    print("Could not download files")
