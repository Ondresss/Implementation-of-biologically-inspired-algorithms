//
// Created by andrew on 21/09/2025.
//

#include "Ackley.h"

double Ackley::evaluate(const Eigen::VectorXd& params) const {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2 * M_PI;

    long d = params.size();
    double sum1 = params.squaredNorm();
    double sum2 = (c * params.array()).cos().sum();

    double term1 = -a * std::exp(-b * std::sqrt(sum1 / d));
    double term2 = -std::exp(sum2 / d);

    return term1 + term2 + a + std::exp(1.0);
}
