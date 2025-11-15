//
// Created by andrew on 21/09/2025.
//

#include "Zakharov.h"
#include <Eigen/Dense>
#include <cmath>

double Zakharov::evaluate(const Eigen::VectorXd& params) const {
    int n = params.size();
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; ++i) {
        double xi = params[i];
        sum1 += xi * xi;
        sum2 += 0.5 * (i + 1) * xi;
    }

    return sum1 + sum2 * sum2 + std::pow(sum2, 4);
}
