//
// Created by andrew on 21/09/2025.
//

#pragma once
#include <tuple>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "Function.h"

class Solution {
public:
    Solution(int dimension,const std::tuple<int,int>& bounds) : dimension(dimension), bounds(bounds),parameters(Eigen::VectorXd::Zero(dimension)),f_best(INFINITY) {}
    virtual ~Solution() {}
    virtual void visualize();
    virtual void run(std::shared_ptr<Function> f,int noIterations) = 0;
    virtual std::string getName() = 0;
    [[nodiscard]] double getBest() const {return this->f_best;}
    [[nodiscard]] std::tuple<Eigen::VectorXd,double> getBestResult() const { return this->bestResult; }
protected:
    Eigen::VectorXd generateRandomSolution() {
        auto [boundsY, boundsX] = this->f->getOfficialBounds();
        const auto& [y_min, y_max] = boundsY;
        const auto& [x_min, x_max] = boundsX;

        Eigen::VectorXd Xb(dimension);

        if (dimension == 2) {
            // x₁
            Xb(0) = x_min + ((double)rand() / RAND_MAX) * (x_max - x_min);
            // x₂
            Xb(1) = y_min + ((double)rand() / RAND_MAX) * (y_max - y_min);
        }
        else {
            // fallback: stejný rozsah pro všechny dimenze
            auto [lower, upper] = this->bounds;
            for (int i = 0; i < dimension; ++i)
                Xb(i) = lower + ((double)rand() / RAND_MAX) * (upper - lower);
        }

        return Xb;
    }

    int dimension;
    std::tuple<int,int> bounds;
    Eigen::VectorXd parameters;
    std::shared_ptr<Function> f;
    std::vector<std::tuple<Eigen::VectorXd,double,bool>> lastResults;
    double f_best;
    std::tuple<Eigen::VectorXd,double> bestResult;

};
