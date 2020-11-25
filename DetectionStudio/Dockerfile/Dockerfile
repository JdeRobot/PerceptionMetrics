FROM ubuntu:18.04

RUN apt-get update && apt-get upgrade -y

RUN apt install -y build-essential git cmake rapidjson-dev libssl-dev python-dev python-numpy \
    libboost-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev

RUN apt install -y libgoogle-glog-dev libyaml-cpp-dev qt5-default libqt5svg5-dev wget unzip vim python-pip

RUN pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow \
    && pip install keras \
    && pip install torch

RUN cd ~ \
    && wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip \
    && unzip opencv.zip \
    && unzip opencv_contrib.zip \
    && mv opencv-4.2.0 opencv \
    && mv opencv_contrib-4.2.0 opencv_contrib \
    && cd ~/opencv \
    && mkdir build \
    && cd build \
    && cmake -D WITH_QT=ON -D WITH_GTK=OFF -D ENABLE_FAST_MATH=1 -D WITH_CUBLAS=1 .. \
    && make -j4 \
    && make install

RUN cd ~ \
    && git clone https://github.com/JdeRobot/DetectionStudio \
    && cd DetectionStudio/DetectionStudio \
    && mkdir build && cd build \
    && cmake .. \
    && make -j4

RUN mkdir /root/volume \
    && mkdir /root/volume/datasets \
    && mkdir /root/volume/datasets/cfg \
    && mkdir /root/volume/datasets/eval \
    && mkdir /root/volume/datasets/names \
    && mkdir /root/volume/datasets/weights

COPY ./appConfig.yml /root/DetectionStudio/DetectionStudio/build/DatasetEvaluationApp/appConfig.yml
WORKDIR /root/DetectionStudio/DetectionStudio/build/DatasetEvaluationApp

CMD ["./DatasetEvaluationApp"]
