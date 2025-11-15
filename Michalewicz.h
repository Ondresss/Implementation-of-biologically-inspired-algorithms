//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"
#include <Eigen/Dense>


class Michalewicz : public Function {
public:
    Michalewicz(const std::string& name,int m) : Function(name),m(m) {}
    Michalewicz(const std::string& name, double step,int m) : Function(name, step), m(m) {}
    ~Michalewicz() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {
        return std::make_tuple(std::make_tuple(0,M_PI), std::make_tuple(0,M_PI));
    }
private:
    int m;
};
