//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"
#include <Eigen/Dense>

class Rosenbrock : public Function {
public:
    Rosenbrock(const std::string& name) : Function(name) {}
    Rosenbrock(const std::string& name, double step) : Function(name, step) {}
    ~Rosenbrock() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(
        std::make_tuple(-10.0, 10.0),
    std::make_tuple(-6.0, 6.0)  // boundsX2// boundsX1
);

    }
};