# Caffe built for CPU operation
FROM bvlc/caffe:cpu

# Ensure we can run notebooks
RUN pip install --upgrade pip \
 && pip install jupyter

# This is the Caffe pre-trained LeNet model we are exploring
RUN /opt/caffe/scripts/download_model_binary.py /opt/caffe/models/bvlc_googlenet

# Include all of the notebooks
COPY *.ipynb /home/jovyan/

COPY images /home/jovyan/images/

WORKDIR /home/jovyan

EXPOSE 8888

# Note that we starte in background and have a default access token (login password) of 'demo'
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='demo'

