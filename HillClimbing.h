#pragma once
#include "Solution.h"
#include <random>
#include <iostream>
class HillClimbing : public Solution {
public:
    HillClimbing(int dimension,
                 const std::tuple<int,int>& bounds,
                 int noNeighbors,
                 double sigma)
        : Solution(dimension, bounds),
          noNeighbors(noNeighbors),
          sigma(sigma),
          rng(std::random_device{}()),
          normal_dist(0.0, sigma) {}

    void run(std::shared_ptr<Function> f, int noIterations) override;

    inline Eigen::VectorXd generateNeighbor(const Eigen::VectorXd& x) {
        Eigen::VectorXd neighbor = x;

        auto [boundsY, boundsX] = this->f->getOfficialBounds();
        const auto& [y_min, y_max] = boundsY;
        const auto& [x_min, x_max] = boundsX;

        for (int i = 0; i < x.size(); ++i) {
            neighbor[i] += normal_dist(rng);

            if (i == 0)
                neighbor[i] = std::clamp(neighbor[i], x_min, x_max);
            else if (i == 1)
                neighbor[i] = std::clamp(neighbor[i], y_min, y_max);
            else {
                auto [lower, upper] = this->bounds;
                neighbor[i] = std::clamp(neighbor[i], (double)lower, (double)upper);
            }
        }

        return neighbor;
    }

    std::string getName() override { return this->name; }
protected:
    int noNeighbors;
    double sigma;
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
    const std::string name = "Hill climbing";
};
