//
// Created by andrew on 21/09/2025.
//

#include "Levy.h"
#include <cmath>

double Levy::evaluate(const Eigen::VectorXd& params) const {
    int n = params.size();
    Eigen::VectorXd w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 1 + (params[i] - 1) / 4.0;
    }

    double term1 = std::sin(M_PI * w[0]) * std::sin(M_PI * w[0]);
    double sum = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        double wi_minus1 = w[i] - 1.0;
        sum += wi_minus1 * wi_minus1 * (1 + 10 * std::sin(M_PI * w[i] + 1) * std::sin(M_PI * w[i] + 1));
    }
    double termN = (w[n - 1] - 1.0) * (w[n - 1] - 1.0) * (1 + std::sin(2 * M_PI * w[n - 1]) * std::sin(2 * M_PI * w[n - 1]));

    return term1 + sum + termN;
}
