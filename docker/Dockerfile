#Dockerfile based on https://github.com/rapidsai/gpuci-build-environment/tree/enh-miniconda-cuda-df &
# https://github.com/rapidsai/build/blob/branch-0.14/generated-dockerfiles/ubuntu18.04-devel.Dockerfile

ARG FROM_IMAGE=gpuci/miniconda-cuda
ARG CUDA_VERSION=10.1
ARG CUDA_VER=${CUDA_VERSION}
ARG CUDA_TYPE=devel
ARG LINUX_VERSION=ubuntu18.04
ARG PYTHON_VERSION=3.6
FROM ${FROM_IMAGE}:${CUDA_VERSION}-${CUDA_TYPE}-${LINUX_VERSION}

# Define arguments
ARG ENV_NAME=falconcv

# Set environment
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/opt/conda/envs/${ENV_NAME}/lib
ENV PATH=${PATH}:/conda/bin

# Add a condarc for override settings
COPY docker/.condarc /conda/.condarc

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install nano -y \
    && apt-get install protobuf-compiler -y

WORKDIR /${ENV_NAME}

COPY . .

RUN source activate base \
    && conda env create -f environment.yml \
    && conda clean -afy \
    && sed -i 's/conda activate base/conda activate `${ENV_NAME}`/g' ~/.bashrc \
    && chmod -R ugo+w /opt/conda \
    && source activate ${ENV_NAME} \
    && python setup.py install #Installs FalconCV on the env

RUN mv docker/.run_in_falconcv.sh /.run_in_falconcv

# Jupyter notebooks port
EXPOSE 8888
# Dask Scheduler Bokeh port
EXPOSE 8787
EXPOSE 8786

ENTRYPOINT [ "/usr/bin/tini", "--", "/.run_in_falconcv"]

CMD [ "/bin/bash" ]