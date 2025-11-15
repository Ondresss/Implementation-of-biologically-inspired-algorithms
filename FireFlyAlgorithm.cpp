//
// Created by andrew on 09/11/2025.
//

#include "FireFlyAlgorithm.h"

void FireFlyAlgorithm::run(std::shared_ptr<Function> f, int noIterations) {
    this->f = f;
    this->initialiseFireFlies();
    this->fireFliesHistory.clear();
    int iter = 0;
    while (iter < noIterations) {
        for (FireFly& fireFly : this->fireFlies) {
            for (FireFly& nextFireFly : this->fireFlies) {
                if (fireFly.fitness > nextFireFly.fitness) {
                    double distance = this->calculateDistance(fireFly, nextFireFly);
                    double beta = this->params.beta0 * std::exp(-this->params.gamma * std::pow(distance,2));
                    Eigen::VectorXd direction = nextFireFly.parameters - fireFly.parameters;
                    this->updateFireFlyPosition(fireFly,direction,beta);
                }
            }
        }
        for (FireFly& fireFly : this->fireFlies) {
            fireFly.fitness = this->f->evaluate(fireFly.parameters);
        }
        this->fireFliesHistory.push_back(this->fireFlies);
        iter++;
    }
}

void FireFlyAlgorithm::visualize() {
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
    for (size_t i = 0; i < this->fireFliesHistory.size(); ++i) {
        px.clear(); py.clear(); pz.clear();
        for (auto& fireFly : this->fireFliesHistory[i]) {
            px.push_back(fireFly.parameters(0));
            py.push_back(fireFly.parameters(1));
            pz.push_back(fireFly.fitness);
        }
        scatter_p->x_data(px);
        scatter_p->y_data(py);
        scatter_p->z_data(pz);

        title("Iteration " + std::to_string(i + 1));
        f->draw();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.9));
    }

    std::vector<double> bx, by, bz;
    const FireFly& bestFireFly = this->findBestFireFly();
    bx.push_back(bestFireFly.parameters(0));
    by.push_back(bestFireFly.parameters(1));
    bz.push_back(bestFireFly.fitness);
    auto scatter_b = ax->scatter3(bx, by, bz);
    scatter_b->marker_style("o");
    scatter_b->marker_size(15);
    scatter_b->marker_face_color("red");


}

void FireFlyAlgorithm::initialiseFireFlies() {
    this->fireFlies.clear();
    this->fireFlies.reserve(this->noFireFlies);
    Eigen::VectorXd randomvec = Eigen::VectorXd::Zero(this->dimension);

    for (int i = 0; i < this->noFireFlies; i++) {
        Eigen::VectorX parameters = this->generateRandomSolution();
        double fitness =  this->f->evaluate(parameters);
        FireFly baseFireFly = {
            .fitness = fitness,
            .parameters = parameters,
            .randomVector = randomvec
        };
        this->fireFlies.push_back(baseFireFly);
    }
}

const FireFlyAlgorithm::FireFly & FireFlyAlgorithm::findBestFireFly() {
    auto it = std::min_element(fireFlies.begin(), fireFlies.end(),
                           [](const FireFly& a, const FireFly& b){ return a.fitness < b.fitness; });
    return *it;
}

double FireFlyAlgorithm::calculateDistance(const FireFly &f, const FireFly &s) const {
    double sum = 0.0;
    for (int i = 0; i < this->dimension; ++i) {
        sum+= std::pow(s.parameters[i] - f.parameters[i], 2);
    }
    return std::sqrt(sum);
}

void FireFlyAlgorithm::updateFireFlyPosition(FireFly &f, const Eigen::VectorXd& direction,double beta) {
    std::normal_distribution<double> dist(0.0, 0.25);
    for (int i = 0; i < this->dimension; ++i) {
        f.randomVector(i) = dist(rng);
    }
    f.parameters = f.parameters + beta * direction + this->params.alpha * f.randomVector;
    this->clampParameters(f.parameters);  
}

void FireFlyAlgorithm::clampParameters(Eigen::VectorXd& params_) const {
    auto [boundsX, boundsY] = this->f->getOfficialBounds();
    const auto& [xmin, xmax] = boundsX;
    const auto& [ymin, ymax] = boundsY;

    for (int j = 0; j < this->dimension; ++j) {
        if (j == 0)
            params_(j) = std::clamp(params_(j), xmin, xmax);
        else
            params_(j) = std::clamp(params_(j), ymin, ymax);
    }

}
