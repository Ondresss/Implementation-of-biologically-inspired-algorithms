    //
    // Created by andrew on 18/10/2025.
    //

    #include "ParticleSwarmOptimization.h"

    #include <iostream>
    #include <ranges>
    #include <matplot/matplot.h>


    void ParticleSwarmOptimization::run (std::shared_ptr<Function> f, int noIterations) {
        std::cout << "PSO running" << std::endl;
        this->f = f;
        this->historyPopulation.clear();
        this->generateInitialPopulation();
        this->currentBestParticle = this->selectBestIndividual();
        if (!this->currentBestParticle) {
            std::cerr << "No individuals found!" << std::endl;
            std::exit(-1);
        }
        int i = 0;
        while (i < noIterations) {
            std::vector<std::shared_ptr<Particle>> snapshot;
            snapshot.reserve(this->initPopulation.size());
            for (const auto& p : this->initPopulation) {
                auto copy = std::make_shared<Particle>();
                copy->parameters = p->parameters;
                copy->velocity = p->velocity;
                copy->pBest = p->pBest;
                snapshot.push_back(copy);
            }
            this->historyPopulation.push_back(snapshot);
            for (auto pair: std::views::enumerate(this->initPopulation)) {
                auto& x = std::get<1>(pair);
                Eigen::VectorXd newVelocity = this->calculateVelocity(x,i,noIterations,this->currentBestParticle->parameters);
                Eigen::VectorXd newParameters = x->parameters + newVelocity;
                this->clampParameters(newParameters);
                double newFitness = this->f->evaluate(newParameters);
                double bestPersonalFitness = this->f->evaluate(x->pBest);
                x->velocity = newVelocity;
                x->parameters = newParameters;
                if (newFitness < bestPersonalFitness) {
                    x->pBest = newParameters;
                }
            }
            this->currentBestParticle = this->selectBestIndividual();
            if (!this->currentBestParticle) {
                std::cerr << "No individuals found!" << std::endl;
                std::exit(-1);
            }
            std::cout << "PSO iteration: " << i << std::endl;
            i++;
        }
        std::cout << "PSO ending" << std::endl;
    }


double ParticleSwarmOptimization::getBestSolution() {
        auto bestIndividual = std::min_element(this->initPopulation.begin(), this->initPopulation.end(),[this](const auto& lhs, const auto& rhs) {
              return this->f->evaluate(lhs->parameters) < this->f->evaluate(rhs->parameters);
        });
        double bestFitness = this->f->evaluate((*bestIndividual)->parameters);
        return bestFitness;
}


void ParticleSwarmOptimization::clampParameters(Eigen::VectorXd& params) const {
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


    void ParticleSwarmOptimization::visualize() {
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
        this->historyPopulation.erase(this->historyPopulation.begin());
        for (size_t i = 0; i < this->historyPopulation.size(); ++i) {
            px.clear(); py.clear(); pz.clear();
            for (auto& particle : historyPopulation[i]) {
                px.push_back(particle->parameters(0));
                py.push_back(particle->parameters(1));
                pz.push_back(this->f->evaluate(particle->parameters));
            }
            scatter_p->x_data(px);
            scatter_p->y_data(py);
            scatter_p->z_data(pz);

            title("Iteration " + std::to_string(i + 1));
            f->draw();
            std::this_thread::sleep_for(std::chrono::duration<double>(0.9));
        }
        std::vector<double> vx, vy, vz;
        auto bestParameter = this->currentBestParticle->parameters;
        vx.push_back(bestParameter(0));
        vy.push_back(bestParameter(1));
        vz.push_back(this->f->evaluate(bestParameter));

        auto scatter_pop = ax->scatter3(vx, vy, vz);
        scatter_pop->marker_style("o");
        scatter_pop->marker_size(20);
        scatter_pop->marker_face_color("red");

        ax->xlabel("X");
        ax->ylabel("Y");
        ax->zlabel("Z");
        ax->grid(true);


    }

    Eigen::VectorXd ParticleSwarmOptimization::calculateVelocity(std::shared_ptr<Particle> particle,int i,int mMax, const Eigen::VectorXd& gBest) {
        const double ws = 0.9;
        const double we = 0.4;
        double w = ws - ((ws - we) * i) / static_cast<double>(mMax);
        double r1 = this->r1Random(this->rng);
        double r2 = this->r1Random(this->rng);
        const auto [c1,c2] = this->learningConstants;
        Eigen::VectorXd velocity = particle->velocity * w
                                  + r1 * c1 * (particle->pBest - particle->parameters)
                                  + r2 * c2 * (gBest - particle->parameters);
        this->clampParametersVelocity(velocity);

        double maxVelNorm = std::get<1>(this->minMaxVelocity); // nebo samostatný limit
        double velNorm = velocity.norm();
        if (velNorm > maxVelNorm && velNorm > 0.0) {
            velocity = velocity * (maxVelNorm / velNorm);
        }
        return velocity;
    }


    void ParticleSwarmOptimization::generateInitialPopulation() {
        this->initPopulation.clear();
        this->initPopulation.reserve(this->popSize);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int i = 0; i < this->popSize; ++i) {
            std::shared_ptr<Particle> particle = std::make_shared<Particle>();
            auto poz = this->generateRandomSolution();
            particle->parameters = poz;
            particle->pBest = poz;
            double minVel = std::get<0>(this->minMaxVelocity);
            double maxVel = std::get<1>(this->minMaxVelocity);

            particle->velocity = Eigen::VectorXd::NullaryExpr(
                this->dimension, [minVel, maxVel, this, &uni]() {
                    double r = uni(this->rng);
                    return minVel + r * (maxVel - minVel);
            });
            this->initPopulation.push_back(particle);
        }
    }


    std::shared_ptr<ParticleSwarmOptimization::Particle> ParticleSwarmOptimization::selectBestIndividual() {
        if (this->initPopulation.empty()) {
            return nullptr;
        }
        auto bestIndividual = std::min_element(this->initPopulation.begin(), this->initPopulation.end(),[this](const auto& lhs, const auto& rhs) {
                return this->f->evaluate(lhs->parameters) < this->f->evaluate(rhs->parameters);
        });

        if (bestIndividual != this->initPopulation.end()) {
            return *bestIndividual;
        }

        return nullptr;
    }

    void ParticleSwarmOptimization::clampParametersVelocity(Eigen::VectorXd &params) {
        double minVel = std::get<0>(this->minMaxVelocity);
        double maxVel = std::get<1>(this->minMaxVelocity);
        for (int j = 0; j < params.size(); ++j)
            params(j) = std::clamp(params(j), minVel, maxVel);
    }
