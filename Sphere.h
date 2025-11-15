//
// Created by andrew on 23/09/2025.
//

#pragma once
#include "Function.h"


class Sphere : public Function {
public:
    Sphere(const std::string& name) : Function(name) {}
    Sphere(const std::string& name, double step) : Function(name, step) {}
    ~Sphere() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(std::make_tuple(-6.0, 6.0), std::make_tuple(-6.0, 6.0));
    }
};