//
// Created by andrew on 01/11/2025.
//
#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include <functional>
#include <matplot/matplot.h>
class AntColonyOptimization {
public:
    struct Point {
        double x, y;
    };
    struct AntColonyParams {
        double alpha;
        double beta;
        double rho;
        double Q;
    };
    struct Ant {
        int currentCity;
        int startCity;
        std::vector<int> path;
        std::vector<int> visited;
        double fitness;
    };
    AntColonyOptimization(const std::vector<Point>& cities,int noAnts,AntColonyParams params)
    : cities(cities),noAnts(noAnts),params(params),rng(std::random_device()()) {}
    void run(int noIterations);
    void visualize() const;
private:
    std::function<double(int,int)> distance = [this](int fCityIndex, int sCityIndex) -> double {
        Point fCity = this->cities.at(fCityIndex);
        Point sCity = this->cities.at(sCityIndex);
        return std::sqrt(std::pow(sCity.x - fCity.x, 2) + std::pow(sCity.y - fCity.y, 2));
    };
    std::vector<Point> cities;
    int noAnts;
    AntColonyParams params;
    std::mt19937 rng;
    std::vector<std::vector<double>> tau;
    std::vector<Ant> ants;
    std::vector<std::vector<Ant>> antHistory;
    void intializeTau();
    void intializeAnts();
    std::vector<int> getUnvisitedCities(const Ant& ant);
    [[nodiscard]] std::vector<double> getProbabilities(const Ant& a,const std::vector<int>& unvisitedCities) const;
    [[nodiscard]] double calculateTotalFitness(const Ant& ant) const;
    void evaporatePheromones();
    void antInfluencePheromones();
    [[nodiscard]] const Ant& findBestAnt(const std::vector<Ant>& pAnts) const;
};
