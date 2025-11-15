//
// Created by andrew on 21/09/2025.
//

#include "Griewank.h"
#include <cmath>

double Griewank::evaluate(const Eigen::VectorXd& params) const {
    double sum = 0.0;
    double prod = 1.0;
    int n = params.size();

    for (int i = 0; i < n; ++i) {
        sum += params[i] * params[i] / 4000.0;
        prod *= std::cos(params[i] / std::sqrt(i + 1));
    }

    return sum - prod + 1.0;
}
