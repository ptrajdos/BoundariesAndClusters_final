#!/usr/bin/env bash


#####
#
#For building javabridge from sources 
#
######

#export CC=gcc
#export CXX=g++

export CFLAGS="-L${JAVA_HOME}/lib/server -ljvm"
export CXXFLAGS="-L${JAVA_HOME}/lib/server -ljvm"

export C_INCLUDE_PATH=$JAVA_HOME/include:$JAVA_HOME/include/freebsd
export LD_LIBRARY_PATH=$JAVA_HOME/lib/:$JAVA_HOME/lib/server
export JAVA_VERSION=17
