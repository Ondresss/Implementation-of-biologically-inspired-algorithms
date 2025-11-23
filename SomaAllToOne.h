//
// Created by andrew on 24/10/2025.
//
#pragma once
#include "Solution.h"
#include <random>
#include <ranges>
#include <matplot/matplot.h>
#include <thread>
class SomaAllToOne : public Solution {
public:
    struct Individual {
        double fitness;
        Eigen::VectorXd parameters;
    };
    SomaAllToOne(int dimension,const std::tuple<int,int>& bounds,int popSize,double step,double PRT,double pathLength) :
    Solution(dimension,bounds),popSize(popSize),step(step),PRT(PRT),pathLength(pathLength),rng(std::random_device{}()) {}

    void run(std::shared_ptr<Function> f, int noIterations) override;
    void visualize() override;
    std::string getName() override { return "SOMA       "; };
    double getBestSolution() override;
private:
    int popSize;
    double step;
    double PRT;
    double pathLength;
    std::mt19937 rng;
    std::vector<Individual> initialPopulation;
    std::vector<std::vector<Individual>> historyPopulations;
    std::vector<Individual> foundLeaders;
    Eigen::VectorXd generatePRTVector();
    Individual findLeader();
    void generateInitialPopulation();
    void clampParameters(Eigen::VectorXd& params) const;
};

