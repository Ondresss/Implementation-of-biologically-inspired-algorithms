//
// Created by andrew on 24/10/2025.
//

#include "SomaAllToOne.h"

#include <random>

void SomaAllToOne::run(std::shared_ptr<Function> f, int noIterations) {
    this->f = f;
    int m = 0;
    const int noMigrations = noIterations;
    this->generateInitialPopulation();
    this->historyPopulations.clear();
    while (m < noMigrations) {
        for (const auto& p : std::views::enumerate(this->initialPopulation)) {
            auto& x = std::get<1>(p);
            Individual& leader = this->findLeader();
            this->foundLeaders.push_back(leader);
            const auto& leaderParams = leader.parameters;
            for (double t = 0; t < this->pathLength + 1e-12; t+=this->step) {
                auto PrtVector = this->generatePRTVector();
                Eigen::VectorXd newParameters = Eigen::VectorXd::Zero(this->dimension);
                for (int d = 0; d < this->dimension;++d) {
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
        m++;
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
Eigen::VectorXd SomaAllToOne::generatePRTVector() {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(this->dimension);
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < this->dimension; ++i) {
        if (dist(this->rng) < this->PRT) {
            result(i) = 1;
        } else {
            result(i) = 0;
        }
    }
    return result;
}

SomaAllToOne::Individual& SomaAllToOne::findLeader() {
    Individual& leader = *std::min_element(this->initialPopulation.begin(), this->initialPopulation.end(),[this](const auto& x,const auto& y) {
        return this->f->evaluate(x.parameters) < this->f->evaluate(y.parameters);
    });
    return leader;
}

void SomaAllToOne::generateInitialPopulation() {
    for (int i = 0; i < this->popSize;++i) {
        auto params = this->generateRandomSolution();
        Individual ind = {this->f->evaluate(params),params};
        this->initialPopulation.push_back(ind);
    }
}

void SomaAllToOne::clampParameters(Eigen::VectorXd& params) const {
    auto [boundsX, boundsY] = this->f->getOfficialBounds();
    const auto& [xmin, xmax] = boundsX;
    const auto& [ymin, ymax] = boundsY;

    for (int j = 0; j < this->dimension; ++j) {
        if (j == 0)
            params(j) = std::clamp(params(j), xmin, xmax);
        else if (j == 1)
            params(j) = std::clamp(params(j), ymin, ymax);
    }
}