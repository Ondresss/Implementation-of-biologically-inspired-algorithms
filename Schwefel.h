//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"
#include <Eigen/Dense>

class Schwefel : public Function {
public:
    Schwefel(const std::string& name) : Function(name) {}
    Schwefel(const std::string& name, double step) : Function(name, step) {}
    ~Schwefel() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(std::make_tuple(-500.0, 500.0), std::make_tuple(-500.0, 500.0));
    }

};