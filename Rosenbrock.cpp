//
// Created by andrew on 21/09/2025.
//

#include "Rosenbrock.h"

#include <cmath>

double Rosenbrock::evaluate(const Eigen::VectorXd& params) const {
    double sum = 0.0;
    int n = params.size();

    for (int i = 0; i < n - 1; ++i) {
        double xi = params[i];
        double xnext = params[i + 1];
        sum += 100.0 * std::pow(xnext - xi*xi, 2) + std::pow(1 - xi, 2);
    }

    return sum;
}
