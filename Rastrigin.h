//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"
#include <Eigen/Dense>

class Rastrigin : public Function {
public:
    Rastrigin(const std::string& name) : Function(name) {}
    Rastrigin(const std::string& name, double step) : Function(name, step) {}
    ~Rastrigin() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(
            std::make_tuple(-5.0, 5.0),  // osa Y
            std::make_tuple(-5.0, 5.0)   // osa X (min < max)
        );
    }

};