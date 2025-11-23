//
// Created by andrew on 15/11/2025.
//


#include <random>
#include <thread>
#include "Solution.h"
#include "matplot/matplot.h"
class TeachingLearningBasedOptimization : public Solution {
public:
    struct Individual {
        Eigen::VectorXd parameters;
        double fitness;
    };
    TeachingLearningBasedOptimization(int dimension,const std::tuple<int,int>& bounds,int popSize)
    : Solution(dimension,bounds), populationSize(popSize),rng(std::random_device{}()) {}
    void run(std::shared_ptr<Function> f, int noIterations) override;
    void visualize() override;
    [[nodiscard]] const Individual& findBestIndividual() const;
    std::string getName() override { return "TLBO"; }
    double getBestSolution() override;

private:
    int populationSize;
    std::vector<Individual> population;
    std::vector<std::vector<Individual>> history;
    std::mt19937 rng;

    void clampParameters(Eigen::VectorXd& params_) const;
    [[nodiscard]] Eigen::VectorXd getMeanVector() const;
    void runTeacherPhase();
    void runLearnerPhase();
    void initializePopulation();

    int findNextStudent(int exclude);
};
