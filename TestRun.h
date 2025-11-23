//
// Created by andrew on 30/09/2025.
//
#pragma once
#include <memory>
#include <matplot/matplot.h>

#include "Ackley.h"
#include "Griewank.h"
#include "HillClimbing.h"
#include "Rastrigin.h"
#include "Schwefel.h"
#include "Sphere.h"
#include "BlindSearch.h"
#include "DifferentialEvolution.h"
#include "Levy.h"
#include "Michalewicz.h"
#include "Rosenbrock.h"
#include "SimulatedAnnealing.h"
#include "Zakharov.h"
#include "GeneticAlgorithm.h"
#include "DifferentialEvolution.h"
#include "ParticleSwarmOptimization.h"
#include "SomaAllToOne.h"
#include "AntColonyOptimization.h"
#include <iostream>

#include "FireFlyAlgorithm.h"
#include "TeachingLearningBasedOptimization.h"
#include "XlsWriter.h"

class TestRun {
public:

    static void runXlsTest() {
        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Levy>("Levy", 0.1),
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10),
            std::make_shared<Ackley>("Ackley", 0.25f),
        };
        int dimension = 30;
        std::tuple<int,int> bounds = {-5, 5};
        int NP = 30;
        int F = 8;
        int CR = 9;
        int c1 = 2;                             // Koeficient osobní složky
        int c2 = 2;                             // Koeficient globální složky
        int minVel = -10;                       // Min. rychlost
        int maxVel = 10;
        FireFlyAlgorithm::Parameters params = {0.4, 1.2, 0.7};
        std::vector<std::shared_ptr<Solution>> solutions;
        solutions.push_back(std::make_shared<DifferentialEvolution>(dimension, bounds, NP, F, CR));
        solutions.push_back(std::make_shared<ParticleSwarmOptimization>(dimension, bounds, NP, c1, c2, minVel, maxVel));
        solutions.push_back(std::make_shared<SomaAllToOne>(dimension, bounds, NP, 0.15, 0.3, 3.0));
        solutions.push_back(std::make_shared<FireFlyAlgorithm>(dimension, bounds, NP, params));
        solutions.push_back(std::make_shared<TeachingLearningBasedOptimization>(dimension, bounds, NP));


        XlsWriter xlsWriter(solutions,functions);
        xlsWriter.write(10,30);
    }


    static void runTextTLBO() {
        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Levy>("Levy", 0.1),
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10),
            std::make_shared<Ackley>("Ackley", 0.25f),
        };
        int dimension = 2;
        std::tuple<int,int> bounds = {-5, 5};
        int popSize = 30;
        for (const auto& f : functions) {
            TeachingLearningBasedOptimization tlbo(dimension, bounds, popSize);

            tlbo.run(f,15);
            tlbo.visualize();
        }
    }


    static void runTestFireFly() {
        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Levy>("Levy", 0.1),
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10),
            std::make_shared<Ackley>("Ackley", 0.25f),
        };

        FireFlyAlgorithm fireFlyAlgorithm(
             2,
             { -5, 5 },
             50,
             { 0.4, 1.2, 0.7 }
         );

        for (const auto& f : functions) {
            fireFlyAlgorithm.run(f,15);
            fireFlyAlgorithm.visualize();
        }

    }
    static void runTestAntColonyOptimization() {
            std::vector<AntColonyOptimization::Point> cities = {
                {0.0, 0.0},
                {1.0, 5.0},
                {5.0, 2.0},
                {6.0, 6.0},
                {8.0, 3.0},
                {2.0, 8.0},
                {9.0, 9.0},
                {3.0, 1.0},
                {7.0, 5.0},
                {4.0, 7.0}
            };

            AntColonyOptimization::AntColonyParams params = {0};
            params.alpha = 1.0;   // vliv feromonu
            params.beta  = 5.0;   // vliv heuristiky (vzdálenosti)
            params.rho   = 0.3;   // míra odpařování feromonu
            params.Q     = 100.0; // konstanta feromonu

            int noAnts = 15;        // počet mravenců
            int noIterations = 50;  // počet iterací

            AntColonyOptimization aco(cities, noAnts, params);
            aco.run(noIterations);
            aco.visualize();
    }


    static void runTestSomaAllToOne() {
        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Levy>("Levy", 0.1),
            std::make_shared<Ackley>("Ackley", 0.25f),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10)
        };

        for (auto& f : functions) {
            SomaAllToOne soma(2, {-5, 5}, 30, 0.15, 0.3, 3.0);
            soma.run(f, 15);
            soma.visualize();
        }
    }

    static void runTestParticleSwarmOptimization() {
        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Levy>("Levy", 0.1),
            std::make_shared<Ackley>("Ackley", 0.25f),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10)
        };
        int dimension = 2;                     // Počet dimenzí (např. 10)
        std::tuple<int, int> bounds = {-100, 100}; // Rozsah proměnných
        int popSize = 30;                       // Velikost roje (typicky 20–50)
        int c1 = 2;                             // Koeficient osobní složky
        int c2 = 2;                             // Koeficient globální složky
        int minVel = -10;                       // Min. rychlost
        int maxVel = 10;                        // Max. rychlost
        int noIterations = 15;                 // Počet iterací

        for (const auto& func : functions) {
            std::cout << "=== Running PSO on " << func->getName() << " ===" << std::endl;
            ParticleSwarmOptimization pso(
                dimension,
                bounds,
                popSize,
                c1,
                c2,
                minVel,
                maxVel
            );
            pso.run(func, noIterations);
            pso.visualize();
            std::cout << "Finished " << func->getName() << "\n" << std::endl;
        }

    }

    static void runTestDifferentialAlgorithm() {
        int dimension = 3;
        std::tuple<int,int> bounds = {-5, 5};
        int NP = 20;
        int F = 8;
        int CR = 9;
        int iterations = 20;


        std::vector<std::shared_ptr<Function>> functions = {
            std::make_shared<Ackley>("Ackley", 0.25f),
            std::make_shared<Rastrigin>("Rastrigin", 0.1),
            std::make_shared<Griewank>("Griewank", 0.05),
            std::make_shared<Rosenbrock>("Rosenbrock", 0.1),
            std::make_shared<Michalewicz>("Michalewicz", 0.05, 10),
            std::make_shared<Schwefel>("Schwefel"),
            std::make_shared<Sphere>("Sphere", 0.1),
            std::make_shared<Zakharov>("Zakharov", 0.1),
            std::make_shared<Levy>("Levy", 0.1)
        };

        for (auto& f : functions) {
            DifferentialEvolution diffEvolution(dimension, bounds, NP, F, CR);
            diffEvolution.run(f, iterations);
            diffEvolution.visualize();
        }

    }


};
