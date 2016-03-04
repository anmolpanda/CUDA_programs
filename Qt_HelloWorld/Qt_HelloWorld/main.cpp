#include "mainwindow.h"
#include <QApplication>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "reduce.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();


    // cuda part
    std::default_random_engine dre;
    std::uniform_real_distribution<float> di(-1.1503654,1.15236543645);
    float sum {0.0};
    // vector of float of size 2^20
    std::vector<float> data_vec(1024 * 1024);
    for(auto& ele: data_vec){
        ele = di(dre);
        sum += ele;
    }
    std::cout << "CPU result: " << std::fixed <<  std::setprecision(10) << sum << "\n";

    float gpu_sum {};
    reduce_add(data_vec, &gpu_sum, true);
    // end cuda part


    return a.exec();
}
