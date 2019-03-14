FROM ubuntu:16.04
MAINTAINER Anastasia Haswani <my website>

RUN apt-get update

RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib plotly sklearn

RUN cat /etc/lsb-release

#RUN mkdir -p /app/

#COPY ./analyse.py /app/
#COPY ./wdbc.data /app/
#COPY ./test_noheader.txt /app/
#COPY ./header_5.txt /app/

#WORKDIR /app
#ENV name test_noheader.txt
#ENV header header_5.txt
ENTRYPOINT ["python3","./analyse.py"]
CMD ["breast_cancer"]




