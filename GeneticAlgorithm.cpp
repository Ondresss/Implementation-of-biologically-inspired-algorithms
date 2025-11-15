//
// Created by andrew on 05/10/2025.
//

#include "GeneticAlgorithm.h"

#include <matplot/matplot.h>
#include <thread>
#include <chrono>


void GeneticAlgorithmTSP::run(int noIterations) {
    this->initializePopulation();
    for (int g = 0; g < noIterations; ++g) {
        std::vector<std::vector<int>> newPop = population;
        std::vector<double> newFit = fitness;

        for (int i = 0; i < NP; ++i) {
            int a = i;
            int b;
            do { b = distInt(rng); } while (b == a);

            auto child = this->orderCrossover(population[a], population[b]);

            if (uniformDist(rng) < 0.3)
                child = mutate(child);

            double f_child = this->evaluateRoute(child);
            if (f_child < fitness[a]) {
                newPop[a] = child;
                newFit[a] = f_child;

                if (f_child < f_best) {
                    f_best = f_child;
                    Eigen::VectorXd routeVec(child.size());
                    for (int k = 0; k < child.size(); ++k)
                        routeVec[k] = static_cast<double>(child[k]);
                    bestResult = {routeVec, f_child};
                    historyRoutes.push_back(routeVec);
                }
            }
        }

        population = newPop;
        fitness = newFit;
    }
}

void GeneticAlgorithmTSP::visualize() const {
    using namespace matplot;
    auto plotRoute2D = [this](const Eigen::VectorXd& route) {
        std::vector<double> xs, ys;
        for (int i = 0; i < route.size(); ++i) {
            int idx = static_cast<int>(route[i]);
            xs.push_back(cities[idx].x);
            ys.push_back(cities[idx].y);
        }
        xs.push_back(cities[static_cast<int>(route[0])].x);
        ys.push_back(cities[static_cast<int>(route[0])].y);
        return std::make_pair(xs, ys);
    };

    auto f = figure(true);
    f->size(1500, 500);

    // 1️⃣ počáteční nejlepší
    const auto& [routeF, valF] = this->firstBestRoute;
    subplot(1, 3, 0);
    auto [xsF, ysF] = plotRoute2D(routeF);
    plot(xsF, ysF, "-o")->line_width(2).color("b");
    title("First Best | Fitness: " + std::to_string(valF));
    xlabel("First Best | Fitness: " + std::to_string(valF));
    const auto& [routeB, valB] = this->bestResult;
    subplot(1, 3, 1);
    auto [xsB, ysB] = plotRoute2D(routeB);
    plot(xsB, ysB, "-o")->line_width(2).color("r");
    xlabel("First Best | Fitness: " + std::to_string(valB));

    subplot(1, 3, 2);
    title("Evolution of Best Route");
    xlabel("X");
    ylabel("Y");
    grid(on);

    auto ax = gca();
    hold(ax,true);

    for (size_t i = 0; i < historyRoutes.size(); ++i) {
        cla(ax); // vymazat subplot
        auto [xs, ys] = plotRoute2D(historyRoutes[i]);
        plot(xs, ys, "-o")->line_width(2).color("g");
        title("Iteration " + std::to_string(i + 1));
        f->draw(); // 👈 vynutí aktualizaci grafu (bez zavření okna)
        std::this_thread::sleep_for(std::chrono::duration<double>(0.4));
    }

    matplot::show();

}


