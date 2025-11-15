//
// Created by andrew on 23/09/2025.
//

#include "Sphere.h"

double Sphere::evaluate(const Eigen::VectorXd& params) const {
    int n = params.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow(params[i], 2);
    }
    return sum;
}