//
// Created by andrew on 21/09/2025.
//

#include "BlindSearch.h"

void BlindSearch::run(std::shared_ptr<Function> f, int noIterations) {
    this->f = f;
    this->f_best = INFINITY;
    for (int i = 0; i < noIterations; i++) {
        Eigen::VectorXd Xb = this->generateRandomSolution();
        double val = f->evaluate(Xb);
        bool best = false;
        if (val < this->f_best) {
            this->f_best = val;
            this->parameters = Xb;
            best = true;
        }
        lastResults.push_back({Xb, val, best});
    }
    this->bestResult = std::make_tuple(this->parameters,this->f_best);
}
