#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>


using namespace std;

class CoDE {
public:
    CoDE(int popSize, int dim, int up, int down)
        : POP_SIZE(popSize), DIM(dim), MAX_FES(10000 * dim), F({1.0, 1.0, 0.8}), CR({0.1, 0.9, 0.2}), UP(up),
          DOWN(down), FES(0) {
    }

    void run() {
        vector<individual> population = initialization();
        individual GBEST = population[0];
        for (const auto &ind: population) {
            if (ind.fitness < GBEST.fitness) {
                GBEST = ind;
            }
        }

        evolution(population, GBEST);
        cout << "Final Best: " << GBEST.fitness << endl;
        cout << "Solution: ";
        for (auto val: GBEST.solution) {
            cout << val << " ";
        }
        cout << endl;
        cout << "Total Function Evaluations: " << FES << endl;
    }

private:
    const int POP_SIZE;
    const int DIM;
    const int MAX_FES;
    const vector<double> F;
    const vector<double> CR;
    const int UP;
    const int DOWN;
    int FES = 0; // 函数评估次数计数器

    // 使用 std::vector 替代 std::array
    static double Rosenbrock(const vector<double> &x) {
        double sum = 0.0;
        for (int i = 0; i < x.size() - 1; ++i) {
            sum += 100.0 * pow(x[i + 1] - pow(x[i], 2), 2) + pow(x[i] - 1.0, 2);
        }
        return sum;
    }

    struct individual {
        double fitness;
        vector<double> solution; // 使用 std::vector 替代 std::array
    };

    static mt19937 gen;

    static double ran(double a, double b) {
        uniform_real_distribution<double> dis(a, b);
        return dis(gen);
    }


    vector<int> select_indices(int current, int numIndices) {
        std::vector<int> indices;
        indices.reserve(numIndices);
        std::uniform_int_distribution<> distrib(0, POP_SIZE - 1);

        while (indices.size() < numIndices) {
            int candidate = distrib(gen);
            if (candidate != current && std::find(indices.begin(), indices.end(), candidate) == indices.end()) {
                indices.push_back(candidate);
            }
        }

        return indices;
    }

    // Function to perform the boundary handling operation
    static double boundaryHandling(double u_i_j_G, double L_j, double U_j) {
        if (u_i_j_G < L_j) {
            return min(U_j, 2 * L_j - u_i_j_G);
        } else if (u_i_j_G > U_j) {
            return max(L_j, 2 * U_j - u_i_j_G);
        } else {
            return u_i_j_G;
        }
    }


    vector<individual> initialization() {
        vector<individual> population(POP_SIZE);
        for (auto &ind: population) {
            ind.solution.resize(DIM); // 动态分配大小
            for (auto &val: ind.solution) {
                val = ran(DOWN, UP);
            }
            ind.fitness = Rosenbrock(ind.solution);
        }
        FES += POP_SIZE; // 增加函数评估次数
        return population;
    }

    individual generator(const int current, const vector<individual> &population) {
        uniform_int_distribution<> jrand_dist(0, DIM - 1);
        uniform_int_distribution<> distrib(0, POP_SIZE - 1);
        uniform_int_distribution<> fcr_dist(0, 2); // 随机选择 F 和 CR 的索引
        int a, b, c, d, e;
        vector<individual> threeU(3);
        for (int i = 0; i < 3; ++i) {
            threeU[i].solution.resize(DIM);
        }

        int fcr_index1 = fcr_dist(gen); // 为每个策略随机选择 F和 CR 的索引
        int fcr_index2 = fcr_dist(gen);
        int fcr_index3 = fcr_dist(gen);

        // "rand/1/bin" strategy
        vector<int> indices = select_indices(current, 3);

        a = indices[0], b = indices[1], c = indices[2];
        vector<double> mutant1(DIM); // 使用 std::vector 替代 std::array

        for (int j = 0; j < DIM; ++j) {
            mutant1[j] = population[a].solution[j] + F[fcr_index1] * (
                             population[b].solution[j] - population[c].solution[j]);
            mutant1[j] = boundaryHandling(mutant1[j], DOWN, UP);
        }


        int jrand = jrand_dist(gen);
        for (int k = 0; k < DIM; ++k) {
            if (ran(0.0, 1.0) < CR[fcr_index1] || k == jrand) {
                threeU[0].solution[k] = mutant1[k];
            } else {
                threeU[0].solution[k] = population[current].solution[k];
            }
        }

        threeU[0].fitness = Rosenbrock(threeU[0].solution);


        // "current to rand/1" strategy
        // The mechanism to choose the indices for mutation is slightly different from that of the classic
        // "current to rand/1", we found that using the following mechanism to choose the indices for
        // mutation can improve the performance to certain degree.
        a = distrib(gen);
        b = distrib(gen);
        c = distrib(gen);

        double r = ran(0.0, 1.0);
        for (int k = 0; k < DIM; ++k) {
            threeU[1].solution[k] = population[current].solution[k] +
                                    r * (population[a].solution[k] - population[current].solution[k]) + F[fcr_index2] *
                                    (
                                        population[b].solution[k] - population[c].solution[k]);
            threeU[1].solution[k] = boundaryHandling(threeU[1].solution[k], DOWN, UP);
        }
        threeU[1].fitness = Rosenbrock(threeU[1].solution);

        // "rand/2/bin" strategy
        indices = select_indices(current, 5);
        a = indices[0], b = indices[1], c = indices[2], d = indices[3], e = indices[4];
        vector<double> mutant3(DIM); // 使用 std::vector 替代 std::array
        // The first scaling factor is randomly chosen from 0 to 1
        for (int j = 0; j < DIM; ++j) {
            mutant3[j] = population[a].solution[j] + ran(0.0, 1.0) * (
                             population[b].solution[j] - population[c].solution[j]) + F[fcr_index3] * (
                             population[d].solution[j] - population[e].solution[j]);
            mutant3[j] = boundaryHandling(mutant3[j], DOWN, UP);
        }

        jrand = jrand_dist(gen);
        for (int k = 0; k < DIM; ++k) {
            if (ran(0.0, 1.0) < CR[fcr_index3] || k == jrand) {
                threeU[2].solution[k] = mutant3[k];
            } else {
                threeU[2].solution[k] = population[current].solution[k];
            }
        }
        threeU[2].fitness = Rosenbrock(threeU[2].solution);
        FES += 3;
        individual best3U = threeU[0];
        for (int g = 1; g < 3; ++g) {
            if (threeU[g].fitness < best3U.fitness) {
                best3U = threeU[g];
            }
        }
        return best3U;
    }


    void evolution(vector<individual> &population, individual &gb) {
        while (FES < MAX_FES) {
            bool improved = false;

            for (size_t i = 0; i < POP_SIZE; ++i) {
                if (FES >= MAX_FES) {
                    break;
                }

                individual trial = generator(i, population);

                if (trial.fitness < population[i].fitness) {
                    population[i].fitness = trial.fitness;
                    population[i].solution = trial.solution;

                    if (trial.fitness < gb.fitness) {
                        gb.fitness = trial.fitness;
                        gb.solution = trial.solution;
                        improved = true;
                    }
                }
            }

            if (FES % 100 == 0 || improved) {
                cout << "Function Evaluations " << FES << ": Best = " << gb.fitness << endl;
            }
        }
    }
};

// 修改: 初始化随机数生成器，确保跨平台兼容性
mt19937 CoDE::gen(static_cast<unsigned int>(time(nullptr)));

int main() {
    CoDE code(30, 30, 100, -100); // 移除 F 和 CR 参数
    code.run();
    return 0;
}
