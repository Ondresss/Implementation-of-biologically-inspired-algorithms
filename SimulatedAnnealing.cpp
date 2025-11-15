//
// Created by andrew on 26/09/2025.
//

#include "SimulatedAnnealing.h"


void SimulatedAnnealing::run(std::shared_ptr<Function> f, int noIterations) {
    this->f = f;
    auto current = this->generateRandomSolution();
    double f_curr = f->evaluate(current);
    double T = this->tInit;

    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    while (T > this->tMin) {
        auto neighbor = this->generateNeighbor(current);
        double f_next = f->evaluate(neighbor);

        if (f_next < f_curr) {
            current = neighbor;
            f_curr = f_next;
        } else {
            double delta = f_next - f_curr;
            double acceptProb = std::exp(-delta / T);
            if (uniform(this->rng) < acceptProb) {
                current = neighbor;
                f_curr = f_next;
            }
        }
        T *= this->alpha;
    }

    this->bestResult = std::make_tuple(current, f_curr);
}
