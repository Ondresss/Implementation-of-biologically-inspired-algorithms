//
// Created by andrew on 21/09/2025.
//



#pragma once
#include <Eigen/Dense>
#include <cmath>
class Function {
public:
    explicit Function(std::string name) : name(std::move(name)),step(1.0f) {};
    explicit Function(std::string name,const double step) : name(std::move(name)),step(step) {};
    virtual ~Function() = default;
    [[nodiscard]] virtual double evaluate(const Eigen::VectorXd& params) const = 0;
    [[nodiscard]] const std::string& getName() const { return name; }
    double getStep() const { return step; }
    virtual std::tuple<std::tuple<double,double>, std::tuple<double,double>> getOfficialBounds() const = 0;
private:
    std::string name;
    double step;

};

