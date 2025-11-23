//
// Created by andrew on 11/10/2025.
//

#include "DifferentialEvolution.h"

#include <bits/this_thread_sleep.h>

#include "matplot/matplot.h"

void DifferentialEvolution::visualize() {
    using namespace matplot;
    const auto& [boundsX2, boundsX1] = this->f->getOfficialBounds();
    const auto& [x2_min, x2_max] = boundsX2;  // osa Y
    const auto& [x1_min, x1_max] = boundsX1;  // osa X

    int N = 50;
    auto x_lin = linspace(x1_min, x1_max, N);
    auto y_lin = linspace(x2_min, x2_max, N);


    auto [X, Y] = meshgrid(x_lin, y_lin);

    auto Z = transform(X, Y, [this](double x, double y) {
        return this->f->evaluate(Eigen::Vector2d{x, y});
    });


    auto fig = figure();
    fig->name(f->getName() + " " + this->getName());
    auto ax = fig->add_axes();
    hold(ax,true);

    auto surf_plot = ax->surf(X, Y, Z);
    surf_plot->edge_color("gray");
    surf_plot->edge_color("none");
    surf_plot->face_alpha(0.6);

    std::vector<double> vx, vy, vz;

    double bestFitness = std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestIndividual;

    for (const auto& ind : initPopulation) {
        double val = f->evaluate(Eigen::Vector2d{ind(0), ind(1)});

        vx.push_back(ind(0));
        vy.push_back(ind(1));
        vz.push_back(val);

        if (val < bestFitness) {
            bestFitness = val;
            bestIndividual = ind;
        }
    }


    // vykreslení populace
    auto scatter_pop = ax->scatter3(vx, vy, vz);
    scatter_pop->marker_style("o");
    scatter_pop->marker_size(10);
    scatter_pop->marker_face_color("red");

    // vykreslení historie best jedinců
    std::vector<double> hx, hy, hz;
    for (const auto& ind : historyBestIndividuals) {
        hx.push_back(ind(0));
        hy.push_back(ind(1));
        hz.push_back(f->evaluate(ind));
    }
    auto scatter_history = ax->scatter3(hx, hy, hz);
    scatter_history->marker_style("o");
    scatter_history->marker_size(15);        // střední velikost pro předchozí body
    scatter_history->marker_face_color("green");

    // zvýraznění posledního nejlepšího jedince
    const auto& lastBest = historyBestIndividuals.back();
    auto scatter_last = ax->scatter3({bestIndividual(0)}, {bestIndividual(1)}, {bestFitness});
    scatter_last->marker_style("o");
    scatter_last->marker_size(25);          // větší
    scatter_last->marker_face_color("lime"); // jasně zelená, odlišná



    ax->xlabel("X");
    ax->ylabel("Y");
    ax->zlabel("Z");
    ax->grid(true);

}



double DifferentialEvolution::getBestSolution() {
    auto bestSolution = std::min_element(this->initPopulation.begin(), this->initPopulation.end(),[this](const auto& x, const auto& y) {
        return this->f->evaluate(x) < this->f->evaluate(y);
    });
    double bestFitness = this->f->evaluate(*bestSolution);
    return bestFitness;
}
void DifferentialEvolution::run(std::shared_ptr<Function> f, int noIterations) {
    std::cout << "DE running.." << std::endl;
    this->f = f;
    this->generateInitialPopulation();
    int g = 0;

    while (g < noIterations) {
        auto currentBestIndividual = std::min_element(
            initPopulation.begin(), initPopulation.end(),
            [this](const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
                return this->f->evaluate(a) < this->f->evaluate(b);
            });
        this->historyBestIndividuals.push_back(*currentBestIndividual);

        std::vector<Eigen::VectorXd> newPopulation;
        newPopulation.reserve(this->noIndividuals);

        for (int i = 0; i < this->initPopulation.size(); ++i) {
            const auto& x = initPopulation[i];
            const auto& [r1, r2, r3] = this->getRandomIndices(i);

            Eigen::VectorXd mutatingVector =
                (initPopulation[r1] - initPopulation[r2]) * this->mutationConstant + initPopulation[r3];

            Eigen::VectorXd trialVector = Eigen::VectorXd::Zero(mutatingVector.size());
            int j_rnd = this->uniformDimension(this->rng);

            for (int j = 0; j < this->dimension; ++j) {
                double prob = this->probability(this->rng);
                if (prob < this->crossoverRange || j == j_rnd) {
                    trialVector(j) = mutatingVector(j);
                } else {
                    trialVector(j) = x(j);
                }
            }

            auto [boundsY, boundsX] = this->f->getOfficialBounds();
            const auto& [ymin, ymax] = boundsY;
            const auto& [xmin, xmax] = boundsX;

            for (int j = 0; j < this->dimension; ++j) {
                if (j == 0)
                    trialVector(j) = std::clamp(trialVector(j), xmin, xmax);
                else if (j == 1)
                    trialVector(j) = std::clamp(trialVector(j), ymin, ymax);
                else
                    trialVector(j) = std::clamp(trialVector(j), xmin, xmax);
            }

            double f_u = f->evaluate(trialVector);
            double f_x = f->evaluate(x);

            if (f_u <= f_x)
                newPopulation.push_back(trialVector);
            else
                newPopulation.push_back(x);
        }

        // Nahrazení populace
        this->initPopulation = std::move(newPopulation);
        std::cout << "DE iteration: " << g << std::endl;
        g++;
    }
    std::cout << "DE ended " << std::endl;
}


void DifferentialEvolution::generateInitialPopulation() {
    this->initPopulation.clear();
    this->initPopulation.reserve(this->noIndividuals);
    for (int i = 0; i < this->noIndividuals; ++i)
        this->initPopulation.push_back(this->generateRandomSolution());
}

std::tuple<int, int, int> DifferentialEvolution::getRandomIndices(int excludedIndex) {
    int r1, r2, r3 = 0;
    do {
        r1 = this->uniformIndices(this->rng);
    } while (r1 == excludedIndex);

    do {
        r2 = this->uniformIndices(this->rng);
    } while (r2 == excludedIndex || r2 == r1);

    do {
        r3 = this->uniformIndices(this->rng);
    } while (r3 == excludedIndex || r3 == r1 || r3 == r2);


    return std::make_tuple(r1, r2, r3);
}
