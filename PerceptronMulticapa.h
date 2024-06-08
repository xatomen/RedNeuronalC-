//
// Created by jorge on 8/06/24.
//

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <time.h>
#include <cmath>

#ifndef REDNEURONALC___PERCEPTRONMULTICAPA_H
#define REDNEURONALC___PERCEPTRONMULTICAPA_H

class Perceptron{
public:
    Perceptron::Perceptron(size_t numero_de_entradas, double bias = 1.0);
    double ejecutar(std::vector<double> datos_de_entrada);          //Recibe información y devuelve el resultado correspondiente
    void establecer_pesos(std::vector<double> pesos_iniciales);     //Establecer el peso de cada dato para la salida en la suma ponderada
    double funcion_de_activacion(double x);                         //Controlará cuando se active la neurona, permite modelar relaciones complejas no lineales entre entradas y salidas
    double sigmoide(double x);                                      //Transforma los valores de entrada entre 0 y 1
    double bias;                                                    //Sesgo -> suma ponderada de las entradas antes de que sean procesadas por la función de activación
    std::vector<double> pesos;
};
class PerceptronMulticapa {
public:
    std::vector<std::vector<Perceptron> > red;
    std::vector<size_t> capas;
    std::vector<std::vector<double>> valores;
    double bias;
    void establecer_pesos(std::vector<std::vector<std::vector<double>>> pesos_iniciales);
    std::vector<double> ejecutar(std::vector<double> datos_de_entrada);

    double retro(std::vector<double> x, std::vector<double> y);
    std::vector<std::vector<double>> d;
    double eta;
};


#endif //REDNEURONALC___PERCEPTRONMULTICAPA_H
