# This a tensorflow Dockerfile
# Version 1.15

# Base images 基础镜像

# 构建tensorflow 环境
 FROM ubuntu
#
MAINTAINER LJH
#
# # 安装python 和 pip
 RUN apt update \
         && apt install -y python3-pip python3-dev vim \
         && pip3 install --upgrade pip \
         && pip3 install tensorflow==1.15 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com \ 
         && ln -s /usr/bin/python3 /usr/bin/python \
         && ln -s /usr/bin/pip3 /usr/bin/pip
#EXPOSE 映射端口
EXPOSE 87

#CMD 运行以下命令

