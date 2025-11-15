//
// Created by andrew on 21/09/2025.
//

#include "Michalewicz.h"
#include <cmath>

double Michalewicz::evaluate(const Eigen::VectorXd& params) const {
    int n = params.size();
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double xi = params[i];
        sum += std::sin(xi) * std::pow(std::sin((i + 1) * xi * xi / M_PI), 2 * this->m);
    }
    return -sum;
}
