//
// Created by jorge on 8/06/24.
//

#include "PerceptronMulticapa.h"

double frand(){
    return (2.0*(double)rand() / RAND_MAX) - 1.0;
}

Perceptron::Perceptron(size_t numero_de_entradas){
    this->bias = bias;
    pesos.resize(numero_de_entradas+1);
    generate(pesos.begin(), pesos.end(), frand);
}

void Perceptron::establecer_pesos(std::vector<double> pesos_iniciales) {
    pesos = pesos_iniciales;
}

double Perceptron::ejecutar(std::vector<double> datos_de_entrada) {
    //Entrada de un perceptrón simple
    datos_de_entada.push_back(bias);
    double suma_ponderada = std::inner_product(
            datos_de_entrada.begin(),
            datos_de_entrada.end(),
            pesos.begin(),
            (double)0.0
            );
    return sigmoide(suma_ponderada);
}

double Perceptron::sigmoide(double x){
    return 1.0/(1.0 + exp(-x));
}

/*--------------------------------*/

PerceptronMulticapa::PerceptronMulticapa(std::vector<size_t> capas, double bias){
    this->capas = capas;    //Cantidad de neuronas de cada capa
    this->bias = bias;      //Sesgo

    //Inicializar red neuronal multicapa
    for(size_t i=0; i < capas.size(); i++){
        valores.push_back(std::vector<double>(capas[i], 0.0));
        red.push_back(std::vector<Perceptron>());

        if(i > 0){
            for(size_t j=0; j < capas[i]; j++){
                red[i].push_back(Perceptron(capas[i-1], bias));
            }
        }
    }
}

void PerceptronMulticapa::establecer_pesos(std::vector<std::vector<std::vector<double>> pesos_iniciales){
    for(size_t i=0; i < pesos_iniciales.size(); i++){                             //Capas de la red
        for(size_t j=0; j < pesos_iniciales[i].size(); j++){                      //Neuronas de cada capa
            red[i+1][j].establecer_pesos(pesos_iniciales[i][j]);    //Pesos asociados a las conexiones de cada neurona con las neuronas de la capa anterior
        }
    }
}

std::vector<double> PerceptronMulticapa::ejecutar(std::vector<double> datos_de_entrada) {
    valores[0] = datos_de_entrada;
    for(size_t i=1; i<red.size(); i++){
        for(size_t j=0; j<capas[i]; j++){
            valores[i][j] = red[i][j].ejecutar(valores[i-1]);
        }
    }
    return valores.back();
}
