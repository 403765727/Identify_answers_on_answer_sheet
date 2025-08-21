FROM python:3.10-slim

#COPY ./sources.list /etc/apt/
#RUN apt-get update && apt-get install -y \
#    libopencv-dev python3-opencv libzbar0

#FROM python:3.8.8
ENV LANG "en_US.utf8"
RUN mkdir -p /home/app
WORKDIR /home/app
COPY . /home/app
RUN chmod +x start.sh
RUN pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# RUN pip3 install -r requirements.txt -i http://mirrors.dg.com/pypi/simple --trusted-host mirrors.dg.com

ENTRYPOINT ["./start.sh"]
