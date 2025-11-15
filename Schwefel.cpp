//
// Created by andrew on 21/09/2025.
//

#include "Schwefel.h"
#include <cmath>

double Schwefel::evaluate(const Eigen::VectorXd& params) const {
    double sum = 0.0;
    const int n = params.size();
    for (int i = 0; i < n; ++i) {
        sum += params[i] * std::sin(std::sqrt(std::abs(params[i])));
    }
    return 418.9829 * n - sum;
}
