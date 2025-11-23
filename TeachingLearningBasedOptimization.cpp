//
// Created by andrew on 15/11/2025.
//

#include "TeachingLearningBasedOptimization.h"

#include <ranges>

void TeachingLearningBasedOptimization::run(std::shared_ptr<Function> f, int noIterations) {
    std::cout << "TLBO running" << std::endl;
    this->f = f;
    this->history.clear();
    this->initializePopulation();
    int iter = 0;
    while (iter < noIterations) {
        this->runTeacherPhase();
        this->runLearnerPhase();
        this->history.push_back(this->population);
        std::cout << "TLBO iter: " << iter << std::endl;
        iter++;
    }
    std::cout << "TLBO ended " << std::endl;
}

void TeachingLearningBasedOptimization::visualize() {
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
    for (size_t i = 0; i < this->history.size(); ++i) {
        px.clear(); py.clear(); pz.clear();
        for (auto& ind : this->history[i]) {
            px.push_back(ind.parameters(0));
            py.push_back(ind.parameters(1));
            pz.push_back(ind.fitness);
        }
        scatter_p->x_data(px);
        scatter_p->y_data(py);
        scatter_p->z_data(pz);

        title("Iteration " + std::to_string(i + 1));
        f->draw();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.9));
    }

    std::vector<double> bx, by, bz;
    const Individual& bestInd = this->findBestIndividual();
    bx.push_back(bestInd.parameters(0));
    by.push_back(bestInd.parameters(1));
    bz.push_back(bestInd.fitness);
    auto scatter_b = ax->scatter3(bx, by, bz);
    scatter_b->marker_style("o");
    scatter_b->marker_size(15);
    scatter_b->marker_face_color("red");


}

const TeachingLearningBasedOptimization::Individual& TeachingLearningBasedOptimization::findBestIndividual() const {
    if (this->population.empty()) throw std::runtime_error("No population found");
    auto best = std::min_element(this->population.begin(), this->population.end(),[](const auto& x, const auto& y) {
        return x.fitness < y.fitness;
    });
    if (best == this->population.end()) throw std::runtime_error("No best individual found");
    return *best;
}


double TeachingLearningBasedOptimization::getBestSolution() {
    auto best = std::min_element(this->population.begin(), this->population.end(),[](const auto& x, const auto& y) {
        return x.fitness < y.fitness;
    });
    double bestFitness = this->f->evaluate(best->parameters);
    return bestFitness;
}

void TeachingLearningBasedOptimization::clampParameters(Eigen::VectorXd &params_) const {
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

Eigen::VectorXd TeachingLearningBasedOptimization::getMeanVector() const {
    if (population.empty()) {
        throw std::runtime_error("Population is empty");
    }
    Eigen::VectorXd M = Eigen::VectorXd::Zero(this->dimension);
    for (const auto& indiv : population) {
        M += indiv.parameters;
    }
    M /= static_cast<double>(population.size());
    return M;
}

void TeachingLearningBasedOptimization::runTeacherPhase() {

    auto teacher = this->findBestIndividual();
    Eigen::VectorXd teacherParams = teacher.parameters;
    auto M = this->getMeanVector();
    std::uniform_int_distribution<int> TFP(1,2);
    std::uniform_real_distribution<double> RP(0,1);
    double TF = TFP(this->rng);

    for (Individual& student : this->population) {
        double r = RP(this->rng);
        Eigen::VectorXd xNew = student.parameters + r * (teacherParams - TF * M);
        this->clampParameters(xNew);
        double xFitness = this->f->evaluate(xNew);
        if (xFitness < student.fitness) {
            student.fitness = xFitness;
            student.parameters = xNew;
        }
    }
}

int TeachingLearningBasedOptimization::findNextStudent(int exclude) {
    std::uniform_int_distribution<int> dist(0,static_cast<int>(this->population.size()) - 1);
    int index = dist(this->rng);
    while (index == exclude) {
        index = dist(this->rng);
    }
    return index;
}


void TeachingLearningBasedOptimization::runLearnerPhase() {
    for (const auto& p : std::views::enumerate(this->population)) {
        auto& student = std::get<1>(p);
        int i = std::get<0>(p);
        double nextStudent = this->findNextStudent(i);
        Eigen::VectorXd direction = Eigen::VectorXd::Zero(this->dimension);
        if (student.fitness < this->population.at(nextStudent).fitness) {
            direction = student.parameters - this->population.at(nextStudent).parameters;
        } else {
            direction = this->population.at(nextStudent).parameters - student.parameters;
        }
        std::uniform_real_distribution<double> RP(0,1);
        double r = RP(this->rng);
        Eigen::VectorXd xNew = student.parameters + r * direction;
        this->clampParameters(xNew);
        double xFitness = this->f->evaluate(xNew);
        if (xFitness < student.fitness) {
            student.fitness = xFitness;
            student.parameters = xNew;
        }
    }
}

void TeachingLearningBasedOptimization::initializePopulation() {

    this->population.reserve(this->populationSize);
    for (int i = 0; i < this->populationSize; ++i) {
        auto vec = this->generateRandomSolution();
        double fitness = this->f->evaluate(vec);
        this->population.push_back({vec,fitness});
    }

}
