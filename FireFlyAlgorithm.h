//
// Created by andrew on 09/11/2025.
//

#pragma once
#include <random>
#include <matplot/matplot.h>
#include "Solution.h"
#include <thread>
class FireFlyAlgorithm : public Solution {
public:
    struct FireFly {
        double fitness;
        Eigen::VectorXd parameters;
        Eigen::VectorXd randomVector;
    };
    struct Parameters {
        double alpha;
        double beta0;
        double gamma;
    };
    void run(std::shared_ptr<Function> f, int noIterations) override;
    void visualize() override;
    FireFlyAlgorithm(int dimension,const std::tuple<int,int>& bounds,const int noFireFlies,const Parameters& params)
    : Solution(dimension,bounds), noFireFlies(noFireFlies), params(params),rng(std::random_device{}()) {}

    std::string getName() override { return "FireFlyAlgorithm"; };
private:
    int noFireFlies;
    Parameters params;
    std::vector<FireFly> fireFlies;
    std::vector<std::vector<FireFly>> fireFliesHistory;
    std::mt19937_64 rng;
    void initialiseFireFlies();
    const FireFly& findBestFireFly();
    [[nodiscard]] double calculateDistance(const FireFly& f, const FireFly& s) const;
    void updateFireFlyPosition(FireFly &f, const Eigen::VectorXd& direction,double beta);
    void clampParameters(Eigen::VectorXd& params_) const;
};



