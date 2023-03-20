FROM continuumio/miniconda3

# Download data
RUN mkdir -p /app/data
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rBhPbCBlUtyrVKoyVejpEPZA3FMtUHuz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rBhPbCBlUtyrVKoyVejpEPZA3FMtUHuz" -O app/data/regions.geojson && rm -rf /tmp/cookies.txt
RUN wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m4ORKcqtJYEnYcxkKiVSf5J-s9HvUBje' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1m4ORKcqtJYEnYcxkKiVSf5J-s9HvUBje" -O app/data/GHRSST.nc && rm -rf /tmp/cookies.txt
RUN wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JAE8SdJMiRtxD5HX5ywaalq6uGF2Ht--' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JAE8SdJMiRtxD5HX5ywaalq6uGF2Ht--" -O app/data/TSprofiles_intp.tar.gz && rm -rf /tmp/cookies.txt
RUN wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18896U_N1jnMhkSNUf7EJLs2dmopoVwEi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18896U_N1jnMhkSNUf7EJLs2dmopoVwEi" -O app/data/RTopo-2.0.1_30sec_bedrock_topography.nc && rm -rf /tmp/cookies.txt
# RUN wget -q -O app/data/RTopo-2.0.1_30sec_bedrock_topography.nc https://hs.pangaea.de/Maps/RTopo-2.0.1/RTopo-2.0.1_30sec_bedrock_topography.nc

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Make RUN commands use the new environment:
RUN echo "conda activate env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

WORKDIR /app
COPY /app /app

# The code to run when container is started:
COPY entrypoint.sh ./
EXPOSE 8888
ENTRYPOINT ["./entrypoint.sh"]