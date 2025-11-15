cmake_minimum_required(VERSION 3.10)
project(BIA)

set(CMAKE_CXX_STANDARD 23)

find_package(Matplot++ REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(BIA main.cpp
        Solution.h
        Function.h
        Ackley.cpp
        Ackley.h
        BlindSearch.cpp
        BlindSearch.h
        Solution.cpp
        Rastrigin.cpp
        Rastrigin.h
        Rosenbrock.cpp
        Rosenbrock.h
        Griewank.cpp
        Griewank.h
        Schwefel.cpp
        Schwefel.h
        Levy.cpp
        Levy.h
        Michalewicz.cpp
        Michalewicz.h
        Zakharov.cpp
        Zakharov.h
        HillClimbing.cpp
        HillClimbing.h
        Sphere.cpp
        Sphere.h
        SimulatedAnnealing.cpp
        SimulatedAnnealing.h
        TestRun.h
        GeneticAlgorithm.cpp
        GeneticAlgorithm.h
        DifferentialEvolution.cpp
        DifferentialEvolution.h
        ParticleSwarmOptimization.cpp
        ParticleSwarmOptimization.h
        SomaAllToOne.cpp
        SomaAllToOne.h
        AntColonyOptimization.cpp
        AntColonyOptimization.h
        FireFlyAlgorithm.cpp
        FireFlyAlgorithm.h)


target_link_libraries(BIA PRIVATE Matplot++::matplot)
target_link_options(BIA PRIVATE -fsanitize=address,undefined)
target_include_directories(BIA PRIVATE ${EIGEN3_INCLUDE_DIR})