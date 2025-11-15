//
// Created by andrew on 21/09/2025.
//

#pragma once
#include "Function.h"


class Ackley : public Function {
public:
    Ackley(const std::string& name) : Function(name) {}
    Ackley(const std::string& name,const double step) : Function(name,step) {}
    ~Ackley() override = default;
    double evaluate(const Eigen::VectorXd& params) const override;
    inline std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const override {   return std::make_tuple(std::make_tuple(-40, 40), std::make_tuple(-40, 40));}
};

