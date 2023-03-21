FROM continuumio/miniconda3

# Download data
COPY download_data.sh ./
RUN mkdir -p /src/data
RUN download_data.sh

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Make RUN commands use the new environment:
RUN echo "conda activate env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

WORKDIR /src
COPY /src /src

# The code to run when container is started:
COPY entrypoint.sh ./
EXPOSE 8888
ENTRYPOINT ["./entrypoint.sh"]