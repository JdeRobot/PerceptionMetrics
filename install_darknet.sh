#!/bin/sh
if [ -n "$(ls -A $HOME/darknet/build)" ];
 then
        # We're using a cached version of our OpenCV build
        cd $HOME/darknet/build;
        sudo make install
 else
        # No OpenCV cache – clone and make the files
        rm -rf $HOME/darknet;
	cd $HOME
	git clone https://github.com/JdeRobot/darknet.git
	cd darknet
        mkdir build && cd build
        cmake -DUSE_GPU=OFF ..
        make -j2
        sudo make install
 fi


