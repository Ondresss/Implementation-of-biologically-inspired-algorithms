//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"
#include <Eigen/Dense>


class Griewank : public Function {
public:
    Griewank(const std::string& name) : Function(name) {}
    Griewank(const std::string& name, double step) : Function(name, step) {}
    ~Griewank() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(std::make_tuple(-10, 10), std::make_tuple(-10, 10));
    }
};