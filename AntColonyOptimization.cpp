//
// Created by andrew on 01/11/2025.
//

#include "AntColonyOptimization.h"

#include <thread>

void AntColonyOptimization::intializeAnts() {
    this->ants.clear();
    this->ants.resize(this->noAnts);
    for (int i = 0; i < this->noAnts; ++i) {
        this->ants.at(i) = Ant(-1,-1,{},{},-1);
        this->ants.at(i).path.reserve(this->cities.size());
        this->ants.at(i).visited.resize(this->cities.size(),false);
    }
}
std::vector<int> AntColonyOptimization::getUnvisitedCities(const Ant& ant) {
    std::vector<int> ret;
    for (int i = 0; i < ant.visited.size(); ++i) {
        if (!ant.visited.at(i)) ret.push_back(i);
    }
    return ret;
}
std::vector<double> AntColonyOptimization::getProbabilities(const Ant& a,const std::vector<int>& unvisitedCities) const {
    std::vector<double> ret;
    ret.resize(unvisitedCities.size());
    double sum = 0.0;
    for (size_t i = 0; i < unvisitedCities.size(); ++i) {
        double tauVal = this->tau.at(a.currentCity).at(unvisitedCities.at(i));
        double etaVal = 1.0 / this->distance(a.currentCity,unvisitedCities.at(i));
        double val = std::pow(tauVal,this->params.alpha) * std::pow(etaVal,this->params.beta);
        ret.at(i) = val;
        sum += val;
    }
    for (auto& prop : ret) prop /= sum;
    return ret;
}

double AntColonyOptimization::calculateTotalFitness(const Ant &ant) const {
    double totatDistance = 0.0;
    for (int i = 0; i < ant.path.size() - 1; ++i) {
        totatDistance += distance(ant.path.at(i),ant.path.at(i+1));
    }
    return totatDistance;
}

void AntColonyOptimization::evaporatePheromones() {
    for (auto& r : this->tau) {
        for (auto& val : r) {
            val *= (1 - this->params.rho);
        }
    }
}

void AntColonyOptimization::antInfluencePheromones() {
    for (const auto& a : this->ants) {
        double contribution = this->params.Q / a.fitness;

        for (int i = 0; i < a.path.size() - 1; ++i) {
            this->tau[a.path.at(i)][a.path.at(i+1)] += contribution;
            this->tau[a.path.at(i+1)][a.path.at(i)] += contribution;
        }
    }
}

const AntColonyOptimization::Ant& AntColonyOptimization::findBestAnt(const std::vector<Ant>& pAnts) const {
    auto bestCriterium = [this](const Ant& Al,const Ant& Ar) -> bool {
            return Al.fitness < Ar.fitness;
    };
    auto bestAnt = std::min_element(pAnts.begin(),pAnts.end(),bestCriterium);
    return *bestAnt;
}

void AntColonyOptimization::run(int noIterations) {
    std::uniform_int_distribution<> dist(0, static_cast<int>(this->cities.size() - 1));
    this->intializeTau();
    this->intializeAnts();
    int iter = 0;
    while (iter < noIterations) {
        for (auto& a : this->ants) {
            a.path.clear();
            std::fill(a.visited.begin(), a.visited.end(), false);
        }
        for (auto& a : this->ants) {
            int cityIndex = dist(this->rng);
            a.currentCity = cityIndex;
            a.visited.at(cityIndex) = true;
            a.path.push_back(cityIndex);
            while (std::find(a.visited.begin(), a.visited.end(), false) != a.visited.end()) {
                std::vector<int> unvisitedCities = this->getUnvisitedCities(a);
                std::vector<double> probabilites = this->getProbabilities(a,unvisitedCities);
                std::discrete_distribution<int> dist(probabilites.begin(),probabilites.end());
                int nextCity = unvisitedCities[dist(this->rng)];
                a.currentCity = nextCity;
                a.visited.at(nextCity) = true;
                a.path.push_back(nextCity);
            }
            a.path.push_back(a.path.front());
            a.fitness = this->calculateTotalFitness(a);
        }
        this->evaporatePheromones();
        this->antInfluencePheromones();
        iter++;
        this->antHistory.push_back(this->ants);
    }

}

void AntColonyOptimization::visualize() const {
    using namespace matplot;
    auto plotRoute2D = [this](const std::vector<int>& route) -> std::pair<std::vector<double>, std::vector<double>> {
        std::vector<double> xs, ys;
        for (int i = 0; i < route.size(); ++i) {
            xs.push_back(this->cities[route[i]].x);
            ys.push_back(this->cities[route[i]].y);
        }
        return std::make_pair(xs, ys);
    };

    auto f = figure();
    f->name("TSP Solution with Ant Colony Optimization");
    auto ax = f->add_axes();
    hold(ax,true);
    std::vector<double> cityX, cityY;
    for (const auto& c : this->cities) {
        cityX.push_back(c.x);
        cityY.push_back(c.y);
    }
    for (const auto& a : this->antHistory) {
        cla(ax);
        auto s = ax->scatter(cityX, cityY, 20.0);
        s->marker_style("o");
        s->marker_size(15);
        s->marker_face_color("red");

        for (const auto& na : a) {
            const auto& [xcoords, ycoords] = plotRoute2D(na.path);
            plot(xcoords, ycoords, "-")->color("gray").line_width(1);
        }

        const Ant& currentBestAnt = this->findBestAnt(a);
        const auto& [xcoords, ycoords] = plotRoute2D(currentBestAnt.path);
        plot(xcoords, ycoords, "-o")->line_width(2).color("b");

        title("Ant fitness: " + std::to_string(currentBestAnt.fitness));
        std::this_thread::sleep_for(std::chrono::duration<double>(0.8));
    }

}

void AntColonyOptimization::intializeTau() {
    std::size_t n = this->cities.size();
    this->tau.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        this->tau[i].resize(n,1.0f);
    }
}
