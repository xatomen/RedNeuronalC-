cmake_minimum_required(VERSION 3.28)
project(RedNeuronalC__)

set(CMAKE_CXX_STANDARD 17)

add_executable(RedNeuronalC__ main.cpp
        PerceptronMulticapa.cpp
        PerceptronMulticapa.h)
