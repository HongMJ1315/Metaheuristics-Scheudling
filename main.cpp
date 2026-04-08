#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <iomanip>
#include <string>
#include <omp.h> // 引入 OpenMP 標頭檔
#include <unordered_map>
#include <cstdint>

namespace fs = std::filesystem;

struct TestCase{
    int num_jobs;
    int num_machines;
    std::string instance_name;
    std::vector<std::vector<int>> job_mtx;
};

bool read_taillard_file(const std::string &filepath, TestCase &tc){
    std::ifstream file(filepath);
    if(!file.is_open()){
        std::cerr << "Error: 無法開啟檔案 " << filepath << "\n";
        return false;
    }

    file >> tc.num_jobs >> tc.num_machines >> tc.instance_name;
    tc.job_mtx.assign(tc.num_machines, std::vector<int>(tc.num_jobs));

    for(int m = 0; m < tc.num_machines; ++m){
        for(int j = 0; j < tc.num_jobs; ++j){
            file >> tc.job_mtx[m][j];
        }
    }

    file.close();
    return true;
}

class Scheduling{
    struct Machine{
        int job_id;
        int time;
    };
public:
    std::vector<std::vector<int>> job_mtx;

    int run_scheduling(std::vector<int> order){
        if(job_mtx.empty() || order.empty()){
            return 0;
        }

        int num_machines = job_mtx.size();
        std::vector<int> machine_end_time(num_machines, 0);

        for(int job : order){
            int current_job_ready_time = 0;
            for(int m = 0; m < num_machines; ++m){
                int start_time = std::max(current_job_ready_time, machine_end_time[m]);
                int finish_time = start_time + job_mtx[m][job];

                machine_end_time[m] = finish_time;
                current_job_ready_time = finish_time;
            }
        }
        return machine_end_time.back();
    }
};

class MetaheuristicSolver{
public:

    enum CrossoverType{ OX, LOX, PMX, CX };


    // --- 新增：定義「個體」的資料結構 ---
    struct Individual{
        std::vector<int> chromosome; // 排列方式 (染色體)
        int makespan;                // 目標函數值 (適應度)

        // 方便排序：Makespan 越小越好
        bool operator<(const Individual &other) const{
            return makespan < other.makespan;
        }
    };

    Individual generate_random_individual(){
        Individual ind;
        ind.chromosome.resize(num_jobs);
        for(int i = 0; i < num_jobs; ++i){
            ind.chromosome[i] = i;
        }
        std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), rng);
        ind.makespan = evaluator.run_scheduling(ind.chromosome);
        return ind;
    }

private:

    Scheduling &evaluator;
    int num_jobs;
    std::mt19937 rng;
    std::vector<int> makespan_history, best_makespan_history;
    // 產生單一隨機個體


    // --- 新增：GA 選擇機制 (Tournament Selection) ---
    Individual tournament_selection(const std::vector<Individual> &population, int k = 3){
        std::uniform_int_distribution<int> dist(0, population.size() - 1);
        Individual best_participant = population[dist(rng)];

        for(int i = 1; i < k; ++i){
            Individual participant = population[dist(rng)];
            if(participant.makespan < best_participant.makespan){
                best_participant = participant;
            }
        }
        return best_participant;
    }

    // --- 新增：Order Crossover (OX) 交配算子 ---
    std::pair<Individual, Individual> order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)，兩個子代共用這組切點
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經被放入子代中
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. 複製切點範圍內的基因到對應的子代
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. 交叉填補剩餘的位置
        int p2_idx = (cut2 + 1) % num_jobs;
        int off1_idx = (cut2 + 1) % num_jobs;

        int p1_idx = (cut2 + 1) % num_jobs;
        int off2_idx = (cut2 + 1) % num_jobs;

        for(int i = 0; i < num_jobs; ++i){
            // 處理 offspring1: 從 Parent 2 依序填補
            int gene_from_p2 = p2.chromosome[p2_idx];
            if(!in_offspring1[gene_from_p2]){
                offspring1.chromosome[off1_idx] = gene_from_p2;
                off1_idx = (off1_idx + 1) % num_jobs;
            }
            p2_idx = (p2_idx + 1) % num_jobs;

            // 處理 offspring2: 從 Parent 1 依序填補
            int gene_from_p1 = p1.chromosome[p1_idx];
            if(!in_offspring2[gene_from_p1]){
                offspring2.chromosome[off2_idx] = gene_from_p1;
                off2_idx = (off2_idx + 1) % num_jobs;
            }
            p1_idx = (p1_idx + 1) % num_jobs;
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Linear Order Crossover (LOX) 交配算子 ---
    std::pair<Individual, Individual> linear_order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經被放入子代中
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. 複製切點範圍內的基因到對應的子代 (這部分與 OX 相同)
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. 收集另一個父代中「尚未被使用」的基因 (保持由左至右的順序)
        std::vector<int> remaining_for_off1;
        std::vector<int> remaining_for_off2;

        for(int i = 0; i < num_jobs; ++i){
            if(!in_offspring1[p2.chromosome[i]]){
                remaining_for_off1.push_back(p2.chromosome[i]);
            }
            if(!in_offspring2[p1.chromosome[i]]){
                remaining_for_off2.push_back(p1.chromosome[i]);
            }
        }

        // 4. 由左至右 (Linear) 依序填補子代的空位 (跳過 cut1 到 cut2 的區段)
        int idx1 = 0, idx2 = 0;
        for(int i = 0; i < num_jobs; ++i){
            // 如果是在切點範圍內，直接跳過 (因為已經在步驟 2 填好了)
            if(i >= cut1 && i <= cut2) continue;

            offspring1.chromosome[i] = remaining_for_off1[idx1++];
            offspring2.chromosome[i] = remaining_for_off2[idx2++];
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Partially-Mapped Crossover (PMX) 交配算子 ---
    std::pair<Individual, Individual> partially_mapped_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. 隨機選擇兩個切點 (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // 記錄哪些基因已經「存在於切點範圍內」，用以判斷是否衝突
        std::vector<bool> in_off1_middle(num_jobs, false);
        std::vector<bool> in_off2_middle(num_jobs, false);

        // 建立雙向映射表 (Mapping dictionaries)
        std::vector<int> map_p1_to_p2(num_jobs, -1);
        std::vector<int> map_p2_to_p1(num_jobs, -1);

        // 2. 複製切點範圍內的基因，並建立映射關係
        for(int i = cut1; i <= cut2; ++i){
            int gene1 = p1.chromosome[i];
            int gene2 = p2.chromosome[i];

            // 填入子代 1 與 子代 2 的中間段落
            offspring1.chromosome[i] = gene1;
            in_off1_middle[gene1] = true;

            offspring2.chromosome[i] = gene2;
            in_off2_middle[gene2] = true;

            // 紀錄互相對應的映射關係
            map_p1_to_p2[gene1] = gene2;
            map_p2_to_p1[gene2] = gene1;
        }

        // 3. 填補切點以外的剩餘位置
        for(int i = 0; i < num_jobs; ++i){
            // 跳過中間已經填好的部分
            if(i >= cut1 && i <= cut2) continue;

            // --- 處理 Offspring 1 (繼承 Parent 2 的外圍基因) ---
            int candidate1 = p2.chromosome[i];
            // 如果發生衝突 (該基因已經在 Offspring 1 的中間段出現過)
            // 就透過映射表找出替代基因，這組 while 迴圈能完美解決連鎖映射的問題
            while(in_off1_middle[candidate1]){
                candidate1 = map_p1_to_p2[candidate1];
            }
            offspring1.chromosome[i] = candidate1;

            // --- 處理 Offspring 2 (繼承 Parent 1 的外圍基因) ---
            int candidate2 = p1.chromosome[i];
            // 同樣的邏輯，反向查表
            while(in_off2_middle[candidate2]){
                candidate2 = map_p2_to_p1[candidate2];
            }
            offspring2.chromosome[i] = candidate2;
        }

        return { offspring1, offspring2 };
    }

    // --- 新增：Cycle Crossover (CX) 交配算子 ---
    std::pair<Individual, Individual> cycle_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 建立索引查表：能以 O(1) 時間找到「某個基因在 Parent 1 中的位置」
        std::vector<int> p1_index_of(num_jobs, 0);
        for(int i = 0; i < num_jobs; ++i){
            p1_index_of[p1.chromosome[i]] = i;
        }

        // 記錄哪些位置已經被 Cycle 走訪過
        std::vector<bool> visited(num_jobs, false);

        // 1. 找出第一個 Cycle (這裡固定從 index 0 開始)
        int current_idx = 0;

        // 當還沒繞回起點形成迴圈時，繼續追蹤
        while(!visited[current_idx]){
            visited[current_idx] = true;

            // 在 Cycle 內：
            // 子代 1 繼承 Parent 1 該位置的基因
            offspring1.chromosome[current_idx] = p1.chromosome[current_idx];
            // 子代 2 繼承 Parent 2 該位置的基因
            offspring2.chromosome[current_idx] = p2.chromosome[current_idx];

            // 尋找下一個 index：
            // 找出 Parent 2 在當前位置的基因，並查詢該基因在 Parent 1 的位置
            int val_in_p2 = p2.chromosome[current_idx];
            current_idx = p1_index_of[val_in_p2];
        }

        // 2. 將非 Cycle 範圍的剩餘位置，用「另一個 Parent」直接填補
        for(int i = 0; i < num_jobs; ++i){
            if(!visited[i]){
                // 不在 Cycle 內：
                // 子代 1 繼承 Parent 2 該位置的基因
                offspring1.chromosome[i] = p2.chromosome[i];
                // 子代 2 繼承 Parent 1 該位置的基因
                offspring2.chromosome[i] = p1.chromosome[i];
            }
        }

        return { offspring1, offspring2 };
    }

    
    // --- 新增：突變算子 (Swap Mutation) ---
    void mutate(Individual &ind, double mutation_rate){
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        if(prob(rng) < mutation_rate){
            std::uniform_int_distribution<int> dist(0, num_jobs - 1);
            int idx1 = dist(rng);
            int idx2 = dist(rng);
            while(idx1 == idx2) idx2 = dist(rng);
            std::swap(ind.chromosome[idx1], ind.chromosome[idx2]);
        }
    }

    void save_plot(const std::string &save_dir, const std::string &algo_prefix, const std::string &instance_name, int best_makespan){
        if(best_makespan_history.empty()) return;

        std::string filename = save_dir + "/" + algo_prefix + "_" + instance_name + "_" + std::to_string(best_makespan) + ".jpg";

        FILE *pipe = popen("gnuplot", "w");
        if(pipe){
            fprintf(pipe, "set terminal jpeg size 800,600\n");
            fprintf(pipe, "set output '%s'\n", filename.c_str());
            fprintf(pipe, "set title '%s Trajectory - %s'\n", algo_prefix.c_str(), instance_name.c_str());
            fprintf(pipe, "set xlabel 'Iteration'\n");
            fprintf(pipe, "set ylabel 'Makespan'\n");

            fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', '-' with lines title 'Current Makespan' lc rgb 'red'\n");

            for(size_t i = 0; i < best_makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, best_makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            for(size_t i = 0; i < makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            fflush(pipe);
            pclose(pipe);
        }
    }

public:
    MetaheuristicSolver(Scheduling &sched, int jobs) : evaluator(sched), num_jobs(jobs){
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    // ==========================================
    // 基因演算法 (Genetic Algorithm) 主程式
    // ==========================================
    std::vector<int> genetic_algorithm(int pop_size = 50, int max_gen = 500, double crossover_rate = 0.8, double mutation_rate = 0.1,
        std::string instance_name = "Instance", std::string save_dir = "./img"){
        std::cout << "Run GA with Order Crossover (OX)" << std::endl;

        best_makespan_history.clear();
        makespan_history.clear();

        // 1. 初始化族群 (Initialization)
        std::vector<Individual> population;
        for(int i = 0; i < pop_size; ++i){
            population.push_back(generate_random_individual());
        }

        // 尋找初始最佳解
        Individual global_best = *std::min_element(population.begin(), population.end());

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        // 世代演化迴圈
        for(int gen = 0; gen < max_gen; ++gen){
            std::vector<Individual> next_generation;

            // 菁英保留策略 (Elitism)：將上一代最好的個體直接複製到下一代
            Individual current_best = *std::min_element(population.begin(), population.end());
            next_generation.push_back(current_best);

            // 產生下一代 (直到數量等於 pop_size)
            while(next_generation.size() < pop_size){
                // 選擇父母 (Tournament Selection)
                Individual p1 = tournament_selection(population);
                Individual p2 = tournament_selection(population);
                Individual offspring1, offspring2;

                // 交配 (Crossover)
                if(prob(rng) < crossover_rate){
                    auto offsprings = order_crossover(p1, p2);
                    offspring1 = offsprings.first;
                    offspring2 = offsprings.second;
                }
                else{
                    // 如果不交配，隨機複製其中一個父母
                    offspring1 = (prob(rng) < 0.5) ? p1 : p2;
                    offspring2 = (prob(rng) < 0.5) ? p1 : p2;
                }

                // 突變 (Mutation)
                mutate(offspring1, mutation_rate);
                mutate(offspring2, mutation_rate);

                // 重新評估適應度 (Makespan)
                offspring1.makespan = evaluator.run_scheduling(offspring1.chromosome);
                offspring2.makespan = evaluator.run_scheduling(offspring2.chromosome);

                next_generation.push_back(offspring1);
                next_generation.push_back(offspring2);
            }

            // 更新族群
            population = next_generation;

            // 更新歷史最佳解
            current_best = *std::min_element(population.begin(), population.end());
            if(current_best.makespan < global_best.makespan){
                global_best = current_best;
            }

            // 紀錄歷史軌跡 (用來畫圖)
            // 群體演算法通常紀錄 "該代最佳解" 與 "歷史最佳解"
            makespan_history.push_back(current_best.makespan);
            best_makespan_history.push_back(global_best.makespan);
        }

        std::cout << "GA Result: " << global_best.makespan << std::endl;
        save_plot(save_dir, "GA", instance_name, global_best.makespan);
        return global_best.chromosome;
    }

    void cross_test(){
        std::cout << "Testing Order Crossover (OX)..." << std::endl;
        Individual ind1 = generate_random_individual();
        Individual ind2 = generate_random_individual();
        auto offsprings = order_crossover(ind1, ind2);
        std::cout << "Parent 1: ";
        for(int gene : ind1.chromosome) std::cout << gene << " ";
        std::cout << "\nParent 2: ";
        for(int gene : ind2.chromosome) std::cout << gene << " ";
        std::cout << "\nOffspring: ";
        for(int gene : offsprings.first.chromosome) std::cout << gene << " ";
        std::cout << "\nOffspring: ";
        for(int gene : offsprings.second.chromosome) std::cout << gene << " ";
        std::cout << std::endl;
    }

    // ... (保留您原本的 II, SA, TS 等演算法) ...
};

void calculate_and_write_stats(std::ofstream &csv, const std::string &name, const std::vector<int> &data){
    int min_val = *std::min_element(data.begin(), data.end());
    int max_val = *std::max_element(data.begin(), data.end());
    double avg_val = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    csv << min_val << "," << std::fixed << std::setprecision(2) << avg_val << "," << max_val << ",";
}

int main(){

    MetaheuristicSolver solver(*(new Scheduling()), 10);
    solver.cross_test();

    /*
    std::string test_case_dir = "./Test_case";
    if(!fs::exists(test_case_dir) || !fs::is_directory(test_case_dir)){
        std::cerr << "找不到資料夾: " << test_case_dir << "\n";
        return 1;
    }

    std::ofstream csv_file("results.csv");
    csv_file << "Instance,II_Min,II_Avg,II_Max,SA_Min,SA_Avg,SA_Max,TS_Min,TS_Avg,TS_Max\n";

    int NUM_RUNS = 20;

    for(const auto &entry : fs::directory_iterator(test_case_dir)){
        if(entry.path().extension() == ".txt"){
            std::string filepath = entry.path().string();
            std::cout << "========================================\n";
            std::cout << "開始測試: " << filepath << " (共 " << NUM_RUNS << " 次平行執行)\n";

            TestCase tc;
            if(!read_taillard_file(filepath, tc)) continue;

            Scheduling scheduler;
            scheduler.job_mtx = tc.job_mtx;

            std::vector<int> ga_results(NUM_RUNS);

            for(int run = 1; run <= NUM_RUNS; ++run){
                fs::create_directories("./img/Test_" + std::to_string(run));
            }

#pragma omp parallel for schedule(dynamic)
            for(int run = 1; run <= NUM_RUNS; ++run){
                std::string save_dir = "./img/Test_" + std::to_string(run);

                MetaheuristicSolver solver(scheduler, tc.num_jobs);
                solver.genetic_algorithm(50, 500, 0.8, 0.1, tc.instance_name + "_GA", save_dir);

#pragma omp critical
                {
                    std::cout << "  - 第 " << run << " 次執行完成，圖片已儲存至 " << save_dir << "\n";
                }
            }

            csv_file << tc.instance_name << ",";
            calculate_and_write_stats(csv_file, "GA", ga_results);
            csv_file << "\n";

            std::cout << ">> " << tc.instance_name << " 統計資料已寫入 results.csv\n";
        }
    }

    csv_file.close();
    std::cout << "所有測資執行完畢！結果已匯出至 results.csv\n";
// */
    return 0;
}