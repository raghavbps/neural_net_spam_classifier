FROM tensorflow/tensorflow:2.3.0rc0-jupyter

RUN pip3 install sagemaker-training

ENV PATH="/opt/ml/code:${PATH}"

COPY /source_dir /opt/ml/code
WORKDIR /opt/ml/code
RUN pip3 install -r requirements.txt
ENV SAGEMAKER_PROGRAM train.py