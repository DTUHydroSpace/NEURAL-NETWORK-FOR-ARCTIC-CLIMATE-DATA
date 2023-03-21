# NEURAL NETWORK FOR ARCTIC CLIMATE DATA

To run this repository two options are available either using `Docker` or manually setting up the repository
## Docker
Execute to build and run the docker image (requires installation of Docker) `docker compose up`.
Jupyter lab will be lauched and can be accessed on port 8888.

## Manual
1. Install Anaconda environment using
`conda env create -f environment.yml`
This will also install pip packages from `requirements.txt`.
2. Activate environment using `conda activate env`
3. Install PyTorch using
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
4. Download all the required data using. This requires `wget` ([wget](https://eternallybored.org/misc/wget/) for windows users).
`./download_data.sh`
5. Run jupyter lab
`jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=src --allow-root`

## Link to data
The following is only necessary if the `download_data.sh` script does not work
Download the data from [here](https://drive.google.com/drive/folders/1Qfv0EYKHhM5AfTFoUkNyMb4QjftiQdIY?usp=share_link) and put the files into the `data` folder.
