//
// Created by andrew on 26/09/2025.
//

#pragma once
#include "Solution.h"
#include <random>

#include "HillClimbing.h"

class SimulatedAnnealing : public HillClimbing {
public:
    SimulatedAnnealing(int dimension,
                 const std::tuple<int,int>& bounds,
                 int noNeighbors,
                 double sigma, double tMin, double tInit,double alpha)
    : HillClimbing(dimension, bounds,noNeighbors,sigma), tMin(tMin), tInit(tInit), alpha(alpha) {}
    void run(std::shared_ptr<Function> f,int noIterations) override;
    std::string getName() override { return this->name; }
private:
    double tMin;
    double tInit;
    double alpha;
    const std::string name = "Simulated Annealing";

};

