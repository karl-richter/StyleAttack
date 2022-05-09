ARG PYTHON_VERSION=3.7
FROM python:${PYTHON_VERSION}-stretch
RUN \
  apt-get update && \
  apt-get install -y build-essential libsnappy-dev openjdk-8-jre

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
ENV PYTHONPATH "/workspaces/playground"
WORKDIR /workspaces/playground
CMD ["/bin/bash"]