#-------------------------------------------------
#
# Project created by QtCreator 2016-03-04T17:56:44
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Qt_HelloWorld
TEMPLATE = app

# C++11 support
CONFIG += c++11


SOURCES += main.cpp\
        mainwindow.cpp

# cuda sources
CUDA_SOURCES += reduce.cu

# project dir and outputs
PROJECT_DIR = $$system(pwd)
OBJECTS_DIR = $$PROJECT_DIR/Obj
DESTDIR = ../bin

# Path to cuda SDK install
CUDA_SDK = /usr/local/cuda-7.0/samples
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda-7.0
# GPU architecture
CUDA_ARCH = sm_30
# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# include paths
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$CUDA_SDK/common/inc/
INCLUDEPATH += $$CUDA_SDK/../shared/inc/
# lib dirs
QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/lib
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
# libs - note than i'm using a x_86_64 machine
LIBS += -lcudart -lcudadevrt
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

HEADERS  += mainwindow.h \
    reduce.h

FORMS    += mainwindow.ui

DISTFILES +=
