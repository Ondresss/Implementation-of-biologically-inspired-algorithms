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


class TestRun {
public:


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

   static void runTestSimulatedAnnealing() {
    std::shared_ptr<Function> ackley      = std::make_shared<Ackley>("Ackley", 0.25f);
    std::shared_ptr<Function> rastrigin   = std::make_shared<Rastrigin>("Rastrigin", 0.1);
    std::shared_ptr<Function> griewank    = std::make_shared<Griewank>("Griewank", 0.05);
    std::shared_ptr<Function> rosenbrock  = std::make_shared<Rosenbrock>("Rosenbrock", 0.1);
    std::shared_ptr<Function> michalewicz = std::make_shared<Michalewicz>("Michalewicz", 0.05, 10);
    std::shared_ptr<Function> schwefel    = std::make_shared<Schwefel>("Schwefel");
    std::shared_ptr<Function> sphere      = std::make_shared<Sphere>("Sphere", 0.1);
    std::shared_ptr<Function> zakharov    = std::make_shared<Zakharov>("Zakharov", 0.1);
    std::shared_ptr<Function> levy        = std::make_shared<Levy>("Levy", 0.1);

    SimulatedAnnealing simulatedAnnealingAckley(3, std::make_tuple(-32.768, 32.768), 60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingRastrigin(3, std::make_tuple(-5.12, 5.12), 60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingGriewank(3, std::make_tuple(-10, 10),       20, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingRosenbrock(3, std::make_tuple(-5,10), 60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingMichalewicz(3, std::make_tuple(0, EIGEN_PI),   60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingSchwefel(3, std::make_tuple(-500, 500),        60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingSphere(3, std::make_tuple(-5.12, 5.12),        60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingZakharov(3, std::make_tuple(-10, 10),        60, 0.4, 1e-4, 10.0, 0.90);
    SimulatedAnnealing simulatedAnnealingLevy(3, std::make_tuple(-10, 10),              60, 0.4, 1e-4, 10.0, 0.90);
    //
    // simulatedAnnealingAckley.run(ackley, 1000);
    // simulatedAnnealingAckley.visualize();

    // simulatedAnnealingRastrigin.run(rastrigin, 1000);
    // simulatedAnnealingRastrigin.visualize();

    // simulatedAnnealingGriewank.run(griewank, 10000);
    // simulatedAnnealingGriewank.visualize();
    //
    // simulatedAnnealingRosenbrock.run(rosenbrock, 1000);
    // simulatedAnnealingRosenbrock.visualize();
    //
    // simulatedAnnealingMichalewicz.run(michalewicz, 1000);
    // simulatedAnnealingMichalewicz.visualize();
    //
    // simulatedAnnealingSchwefel.run(schwefel, 1000);
    // simulatedAnnealingSchwefel.visualize();
    //
    // simulatedAnnealingSphere.run(sphere, 1000);
    // simulatedAnnealingSphere.visualize();
    // //
    // simulatedAnnealingZakharov.run(zakharov, 1000);
    // simulatedAnnealingZakharov.visualize();
    //
    simulatedAnnealingLevy.run(levy, 1000);
    simulatedAnnealingLevy.visualize();
}


    static void runGeneticAlgorithmTSP() {
       std::random_device rd;
       std::mt19937 rng(rd());
       std::uniform_int_distribution<int> cityCountDist(20, 40);
       int numCities = cityCountDist(rng);

       std::uniform_real_distribution<double> coordDist(0.0, 200.0);
       std::vector<GeneticAlgorithmTSP::Point> cities(numCities);
       for (int i = 0; i < numCities; ++i) {
           cities[i].x = coordDist(rng);
           cities[i].y = coordDist(rng);
       }

       int populationSize = 50;
       int generations = 200;
       GeneticAlgorithmTSP ga(numCities, cities, populationSize);

       ga.run(generations);

       ga.visualize();

       matplot::show();
   }

    static void runTest1HillClimbing() {
        BlindSearch blind_search(3,std::make_tuple(-5.12, 5.13));
        HillClimbing hillClimbing(3,std::make_tuple(-5.12, 5.13),50,1.0);
        BlindSearch blindSearchAckley(3,std::make_tuple(-32.768, 32.768));
        HillClimbing hillClimbingAckley(3,std::make_tuple(-32.768, 32.768),70,0.5);
        BlindSearch blindSearchZakharov(3,std::make_tuple(-5,10));
        HillClimbing hillClimbingZakharov(3,std::make_tuple(-5,10),30,0.5);
        BlindSearch blindSearchRosenbrock(3,std::make_tuple(-5,10));
        HillClimbing hillClimbingRosenbrock(3,std::make_tuple(-5, 10),60,0.5);
        BlindSearch blindSearchMichalewicz(3,std::make_tuple(0,EIGEN_PI));
        HillClimbing hillClimbingMichalewicz(3,std::make_tuple(0, EIGEN_PI),60,0.5);
        BlindSearch blindSearchRastrigin(3,std::make_tuple(-5.12, 5.12));
        HillClimbing hillClimbingRastrigin(3,std::make_tuple(-5.12, 5.12),60,0.5);
        HillClimbing hillClimbingGriewank(3,std::make_tuple(-50,50),60,0.5);
        BlindSearch blindSearchGriewank(3,std::make_tuple(-600,600));
        HillClimbing hillClimbingSchwefel(3,std::make_tuple(-500,500),60,0.5);
        BlindSearch blindSearchSchwefel(3,std::make_tuple(-500, 500));
        HillClimbing hillClimbingLevy(3,std::make_tuple(-10,10),60,0.5);
        BlindSearch blindSearchLevy(3,std::make_tuple(-10, 10));


        std::shared_ptr<Function> ackley = std::make_shared<Ackley>("Ackley",0.25f);
        std::shared_ptr<Function> rastrigin = std::make_shared<Rastrigin>("Rastrigin",0.1);
        std::shared_ptr<Function> griewank = std::make_shared<Griewank>("Griewank",0.05);
        std::shared_ptr<Function> rosenbrock = std::make_shared<Rosenbrock>("Rosenbrock",0.1);
        std::shared_ptr<Function> michalewicz = std::make_shared<Michalewicz>("Michalewicz",0.05,10);
        std::shared_ptr<Function> schwefel = std::make_shared<Schwefel>("Schwefel");
        std::shared_ptr<Function> sphere = std::make_shared<Sphere>("Sphere",0.1);
        std::shared_ptr<Function> zakharov = std::make_shared<Zakharov>("Zakharov",0.1);
        std::shared_ptr<Function> levy = std::make_shared<Levy>("Levy",0.1);


        hillClimbingGriewank.run(griewank,1000);
        hillClimbingGriewank.visualize();

        hillClimbingAckley.run(ackley,100);
        hillClimbingAckley.visualize();

        hillClimbing.run(sphere,10000);
        hillClimbing.visualize();
        // blind_search.run(sphere,10000);
        // blind_search.visualize();


        hillClimbingZakharov.run(zakharov,10000);
        hillClimbingZakharov.visualize();
        // blindSearchZakharov.run(zakharov,10000);
        // blindSearchZakharov.visualize();

        hillClimbingRosenbrock.run(rosenbrock,10000);
        hillClimbingRosenbrock.visualize();
        // blindSearchRosenbrock.run(rosenbrock,10000);
        // blindSearchRosenbrock.visualize();

        hillClimbingMichalewicz.run(michalewicz,500);
        hillClimbingMichalewicz.visualize();
        // blindSearchMichalewicz.run(michalewicz,10000);
        // blindSearchMichalewicz.visualize();
        //
        hillClimbingRastrigin.run(rastrigin,10000);
        hillClimbingRastrigin.visualize();
        // blindSearchRastrigin.run(rastrigin,10000);
        // blindSearchRastrigin.visualize();

        hillClimbingSchwefel.run(schwefel,10000);
        hillClimbingSchwefel.visualize();
        // blindSearchSchwefel.run(schwefel,10000);
        // blindSearchSchwefel.visualize();

        hillClimbingLevy.run(levy,10000);
        hillClimbingLevy.visualize();
        // blindSearchLevy.run(levy,10000);
        // blindSearchLevy.visualize();

        // blindSearchGriewank.run(griewank,10000);
        // blindSearchGriewank.visualize();

    }

};
