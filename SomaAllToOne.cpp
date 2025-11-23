#include "SomaAllToOne.h"
#include <random>
#include <algorithm>
#include <iostream>

void SomaAllToOne::run(std::shared_ptr<Function> f, int noIterations) {
    std::cout << "SOMA running" << std::endl;
    this->f = f;

    // Vyčistit předchozí běhy
    this->initialPopulation.clear();
    this->foundLeaders.clear();
    this->historyPopulations.clear();

    const int noMigrations = noIterations;
    this->generateInitialPopulation();

    int m = 0;
    while (m < noMigrations) {
        Individual leader = this->findLeader();
        this->foundLeaders.push_back(leader);

        for (auto &x : this->initialPopulation) {
            const auto &leaderParams = leader.parameters;
            for (double t = 0.0; t <= this->pathLength + 1e-12; t += this->step) {
                Eigen::VectorXd PrtVector = this->generatePRTVector();
                Eigen::VectorXd newParameters = x.parameters;

                for (int d = 0; d < this->dimension; ++d) {
                    newParameters(d) = x.parameters(d) + t * (leaderParams(d) - x.parameters(d)) * PrtVector(d);
                }

                this->clampParameters(newParameters);

                double newFitness = this->f->evaluate(newParameters);
                if (newFitness < x.fitness) {
                    x.fitness = newFitness;
                    x.parameters = newParameters;
                }
            }
        }

        this->historyPopulations.push_back(this->initialPopulation);

        std::cout << "SOMA iteration: " << m << " best: " << this->getBestSolution() << std::endl;
        ++m;
    }

    std::cout << "SOMA ended" << std::endl;
}

double SomaAllToOne::getBestSolution() {
    if (this->initialPopulation.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    // použijem uloženou fitness hodnotu místo opětovného volání evaluate
    const auto &leaderIt = std::min_element(this->initialPopulation.begin(),
                                            this->initialPopulation.end(),
                                            [](const Individual &a, const Individual &b) {
                                                return a.fitness < b.fitness;
                                            });
    return leaderIt->fitness;
}

Eigen::VectorXd SomaAllToOne::generatePRTVector() {
    Eigen::VectorXd result(this->dimension);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < this->dimension; ++i) {
        result(i) = (dist(this->rng) < this->PRT) ? 1.0 : 0.0;
    }
    return result;
}

SomaAllToOne::Individual SomaAllToOne::findLeader() {
    if (this->initialPopulation.empty()) {
        throw std::runtime_error("findLeader(): initialPopulation is empty");
    }

    auto it = std::min_element(this->initialPopulation.begin(),
                               this->initialPopulation.end(),
                               [](const Individual &a, const Individual &b) {
                                   return a.fitness < b.fitness;
                               });
    return *it; // vrátíme kopii leadera
}

void SomaAllToOne::generateInitialPopulation() {
    this->initialPopulation.clear();
    this->initialPopulation.reserve(this->popSize);

    for (int i = 0; i < this->popSize; ++i) {
        Eigen::VectorXd params = this->generateRandomSolution();
        double fitness = this->f->evaluate(params);
        Individual ind;
        ind.parameters = params;
        ind.fitness = fitness;
        this->initialPopulation.push_back(std::move(ind));
    }
}

void SomaAllToOne::clampParameters(Eigen::VectorXd &params) const {
    auto [boundsX, boundsY] = this->f->getOfficialBounds();
    const auto& [xmin, xmax] = boundsX;
    const auto& [ymin, ymax] = boundsY;

    for (int j = 0; j < this->dimension; ++j) {
        if (j == 0)
            params(j) = std::clamp(params(j), xmin, xmax);
        else
            params(j) = std::clamp(params(j), ymin, ymax);
    }
}

void SomaAllToOne::visualize() {
    using namespace matplot;
    auto [boundsX, boundsY] = this->f->getOfficialBounds();
    const auto& [xmin, xmax] = boundsX;
    const auto& [ymin, ymax] = boundsY;

    int N = 30; // počet bodů na ose
    auto x_lin = linspace(xmin, xmax, N);
    auto y_lin = linspace(ymin, ymax, N);

    auto [X, Y] = meshgrid(x_lin, y_lin);

    auto Z = transform(X, Y, [this](double x, double y) {
        return this->f->evaluate(Eigen::Vector2d{x, y});
    });

    auto f = figure();
    f->name(this->f->getName() + " " + this->getName());
    auto ax = f->add_axes();
    hold(ax,true);

    auto surf_plot = ax->surf(X, Y, Z);
    surf_plot->edge_color("none");
    surf_plot->face_alpha(0.6);

    std::vector<double> px, py, pz;
    auto scatter_p = ax->scatter3(px, py, pz);
    scatter_p->marker_style("o");
    scatter_p->marker_size(15);
    scatter_p->marker_face_color("green");
    for (size_t i = 0; i < this->historyPopulations.size(); ++i) {
        px.clear(); py.clear(); pz.clear();
        for (auto& individual : this->historyPopulations[i]) {
            px.push_back(individual.parameters(0));
            py.push_back(individual.parameters(1));
            pz.push_back(this->f->evaluate(individual.parameters));
        }
        scatter_p->x_data(px);
        scatter_p->y_data(py);
        scatter_p->z_data(pz);

        title("Iteration " + std::to_string(i + 1));
        f->draw();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.9));
    }


    auto bestIndividual = this->findLeader();

    std::vector<double> bx, by, bz;
    bx.push_back(bestIndividual.parameters(0));
    by.push_back(bestIndividual.parameters(1));
    bz.push_back(bestIndividual.fitness);

    auto scatter_b = ax->scatter3(bx, by, bz);
    scatter_b->marker_style("o");
    scatter_b->marker_size(15);
    scatter_b->marker_face_color("red");

    ax->xlabel("X");
    ax->ylabel("Y");
    ax->zlabel("Z");
    ax->grid(true);

}
