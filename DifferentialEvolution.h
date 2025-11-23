//
// Created by andrew on 11/10/2025.
//

#pragma once
#include "Solution.h"
#include <ranges>
#include <random>
class DifferentialEvolution : public Solution {
public:
    DifferentialEvolution(int dimension,const std::tuple<int,int>& bounds,int NP,int F,int CR) : Solution(dimension,bounds),noIndividuals(NP)
    ,mutationConstant(F),crossoverRange(CR),rng(std::random_device()()),uniformIndices(0,NP-1),uniformDimension(0,dimension),probability(0.0f,1.0f) {}
    void visualize() override;
    void run(std::shared_ptr<Function> f, int noIterations) override;
    std::string getName() override { return "DE"; }
    double getBestSolution() override;
private:
    int noIndividuals;
    int mutationConstant;
    int crossoverRange;
    std::mt19937 rng;
    std::vector<Eigen::VectorXd> historyBestIndividuals;
    std::uniform_int_distribution<int> uniformIndices;
    std::uniform_int_distribution<int> uniformDimension;
    std::uniform_real_distribution<double> probability;
    std::vector<Eigen::VectorXd> initPopulation;

    void generateInitialPopulation();
    std::tuple<int,int,int> getRandomIndices(int excludedIndex);
};

