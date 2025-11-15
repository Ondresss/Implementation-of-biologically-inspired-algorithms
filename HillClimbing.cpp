//
// Created by andrew on 22/09/2025.
//

#include "HillClimbing.h"
#include <random>

void HillClimbing::run(std::shared_ptr<Function> f, int noIterations) {
    this->f = f;
    auto Xb = this->generateRandomSolution();
    double f_xb = f->evaluate(Xb);

    for (int i = 0; i < noIterations; i++) {
        Eigen::VectorXd bestNeighbor = Xb;
        double f_bestNeighbor = f_xb;
        bool best = false;
        for (int j = 0; j < this->noNeighbors; j++) {
            auto neighbor = this->generateNeighbor(Xb);
            double f_xs = this->f->evaluate(neighbor);
            if (f_xs < f_bestNeighbor) {
                f_bestNeighbor = f_xs;
                bestNeighbor = neighbor;
                best = true;
            }
        }
        if (best) {
            f_xb = f_bestNeighbor;
            Xb = bestNeighbor;
            lastResults.push_back({bestNeighbor, f_bestNeighbor, true});
        }

    }
    this->f_best = f_xb;
    this->bestResult = std::make_tuple(Xb,f_xb);
}