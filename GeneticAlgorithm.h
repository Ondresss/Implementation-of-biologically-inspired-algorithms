//
// Created by andrew on 21/09/2025.
// Genetic Algorithm Implementation
//

#pragma once
#include "Solution.h"
#include <random>
#include <algorithm>
class GeneticAlgorithmTSP {
public:
    struct Point {
        double x,y;
    };

    GeneticAlgorithmTSP(int numCities,const std::vector<Point>& cities ,int populationSize = 50)
        : numCities(numCities), NP(populationSize), cities(cities),f_best(INFINITY)
    {
        rng.seed(std::random_device{}());
    }

    void run(int noIterations);
    void visualize() const;
private:
    int NP;
    int numCities;
    std::vector<std::vector<int>> population;
    std::vector<double> fitness;
    std::mt19937 rng;
    std::vector<Point> cities;
    std::uniform_real_distribution<double> uniformDist{0.0, 1.0};
    std::uniform_int_distribution<int> distInt{0, 0};
    std::tuple<Eigen::VectorXd,double> firstBestRoute;
    std::vector<Eigen::VectorXd> historyRoutes;

    double f_best;
    std::tuple<Eigen::VectorXd,double> bestResult;
  void initializePopulation() {
        this->population.clear();
        this->fitness.clear();

        std::vector<int> base(numCities);
        std::iota(base.begin(), base.end(), 0);

        this->distInt = std::uniform_int_distribution<int>(0, numCities - 1);

        for (int i = 0; i < NP; ++i) {
            auto indiv = base;
            std::shuffle(indiv.begin(), indiv.end(), rng);
            this->population.push_back(indiv);
            double fval = evaluateRoute(indiv);
            this->fitness.push_back(fval);

            if (fval < f_best) {
                f_best = fval;
                Eigen::VectorXd routeVec(indiv.size());
                for (int k = 0; k < numCities; ++k)
                    routeVec[k] = static_cast<double>(indiv[k]);
                bestResult = {routeVec, fval};
            }
        }
      this->firstBestRoute = this->bestResult;
    }

    static double distance(const Point& a, const Point& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return std::sqrt(dx*dx + dy*dy);
    }

    double evaluateRoute(const std::vector<int>& route) {
        double total = 0.0;
        for (int i = 0; i < numCities - 1; ++i)
            total += distance(cities[route[i]], cities[route[i+1]]);
        total += distance(cities[route.back()], cities[route.front()]);
        return total;
    }

    std::vector<int> orderCrossover(const std::vector<int>& A, const std::vector<int>& B) {
      int start = distInt(rng);
      int end = distInt(rng);
      if (start > end) std::swap(start, end);

      std::vector<int> child(numCities, -1);

      // zkopíruj podsekci z A
      for (int i = start; i <= end; ++i)
          child[i] = A[i];

      // doplň z B
      int bIdx = 0;
      for (int i = 0; i < numCities; ++i) {
          if (child[i] == -1) {
              // najdi první prvek z B, který ještě není v child
              while (std::find(child.begin(), child.end(), B[bIdx]) != child.end())
                  bIdx++;
              child[i] = B[bIdx];
          }
      }

      return child;
  }


    std::vector<int> mutate(std::vector<int> route) {
        int i = distInt(rng);
        int j = distInt(rng);
        std::swap(route[i], route[j]);
        return route;
    }
};
