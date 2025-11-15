//
// Created by andrew on 18/10/2025.
//

#pragma once
#include <random>

#include "Solution.h"
#include <ranges>
#include <thread>

class ParticleSwarmOptimization : public Solution {
public:
    struct Particle {
        Eigen::VectorXd parameters;
        Eigen::VectorXd pBest;
        Eigen::VectorXd velocity;
    };
    ParticleSwarmOptimization(int dimension,const std::tuple<int,int>& bounds, int popSize,int c1,int c2,int minVel,int maxVel)
    : Solution(dimension,bounds), popSize(popSize)
    ,learningConstants(std::make_tuple(c1,c2)),minMaxVelocity(std::make_tuple(minVel,maxVel)),rng(std::random_device()()),r1Random(0,1) {
        std::srand(std::time(0));
    }
    void run(std::shared_ptr<Function> f, int noIterations) override;
    void visualize() override;
    std::string getName() override { return "ParticleSwarmOptimization"; };
private:
    int popSize;
    std::tuple<int,int> learningConstants;
    std::tuple<int,int> minMaxVelocity;
    std::shared_ptr<Particle> currentBestParticle;
    std::mt19937 rng;
    std::uniform_real_distribution<double> r1Random;
    Eigen::VectorXd calculateVelocity(std::shared_ptr<Particle> particle,int i,int mMax, const Eigen::VectorXd& gBest);
    std::vector<std::shared_ptr<Particle>> initPopulation;
    std::vector<std::vector<std::shared_ptr<Particle>>> historyPopulation;
    void generateInitialPopulation();
    std::shared_ptr<Particle> selectBestIndividual();
    void clampParametersVelocity(Eigen::VectorXd& params);
    void clampParameters(Eigen::VectorXd& params) const;


};

