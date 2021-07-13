ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-base
LABEL maintainer="morris.chris.m@gmail.com"

RUN apt update
RUN apt install -y wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O miniconda.sh && \
/bin/bash miniconda.sh -b -p /opt/conda
ENV PATH="/opt/conda/bin:${PATH}"

ENV SIAMESE_PATH=/usr/local/siamese_network
RUN mkdir ${SIAMESE_PATH}
COPY . ${SIAMESE_PATH}/
RUN cd ${SIAMESE_PATH}

RUN sed -i --expression s/cudatoolkit=11.0/cudatoolkit=${CUDA_VERSION}/g ${SIAMESE_PATH}/environment.yml

RUN conda env create -f ${SIAMESE_PATH}/environment.yml

EXPOSE 9183-9193

WORKDIR ${SIAMESE_PATH}/nbs
# CMD ["source", "activate", "siamese"]
# CMD ["jupyter", "notebook", "--no-browser", "--port=9183"]