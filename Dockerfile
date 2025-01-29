FROM ubuntu:20.04

LABEL MAINTAINER splathottam@anl.gov
ARG PYTHON_VERSION=3.12.7

ENV container docker
ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.utf8
ENV MAKEFLAGS -j4

RUN mkdir /app
WORKDIR /app

# DEPENDENCIES needed for Pandas
#===========================================
RUN apt-get update -y && \
    apt-get install -y \
	gcc \
	make \
	wget \
	zlib1g-dev \
	libffi-dev \
	libssl-dev \
	libbz2-dev \
	liblzma-dev

# INSTALL PYTHON
#===========================================
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --with-ensurepip=install --enable-shared && make && make install && \
    ldconfig && \
    ln -sf python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel


RUN mkdir /home/powerdatapipeline
WORKDIR /home/powerdatapipeline

COPY requirements.txt . 
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY datapipeline datapipeline 
COPY config config
COPY utilities utilities
COPY examples examples

# CLEAN UP
#===========================================
RUN rm -rf /app/Python-$PYTHON_VERSION.tgz

RUN apt-get purge -y gcc make wget zlib1g-dev libffi-dev libssl-dev && \
    apt-get autoremove -y

RUN echo 'Power Data Pipeline Developement continer image succesfully built'
