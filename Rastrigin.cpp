//
// Created by andrew on 21/09/2025.
//

#include "Rastrigin.h"
#include <cmath>

double Rastrigin::evaluate(const Eigen::VectorXd& params) const {
    double A = 10.0;
    int n = params.size();
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        double xi = params[i];
        sum += xi*xi - A * std::cos(2 * M_PI * xi);
    }

    return A * n + sum;
}
