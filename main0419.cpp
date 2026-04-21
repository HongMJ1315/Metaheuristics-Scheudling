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
#include <omp.h> // �ޤJ OpenMP ���Y��
#include <unordered_map>
#include <cstdint>
#include <unordered_set>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

std::unordered_map<std::string, int> bks_map = {
    {"tai20_5_1", 1278},
    {"tai20_10_1", 1582},
    {"tai20_20_1", 2297},
    {"tai50_5_1", 2724},
    {"tai50_10_1", 2991},
    {"tai50_20_1", 3850},
    {"tai100_5_1", 5493},
    {"tai100_10_1", 5770},
    {"tai100_20_1", 6202}
};

struct TestCase{
    int num_jobs;
    int num_machines;
    std::string instance_name;
    std::vector<std::vector<int>> job_mtx;
};

bool read_taillard_file(const std::string &filepath, TestCase &tc){
    std::ifstream file(filepath);
    if(!file.is_open()){
        std::cerr << "Error: �L�k�}���ɮ� " << filepath << "\n";
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

    enum MutationType{ Insertion, Swap };

    enum LSSelectionType{ Elite, Top5Random };

    // --- �s�W�G�w�q�u����v����Ƶ��c ---
    struct Individual{
        std::vector<int> chromosome; // �ƦC�覡 (�V����)
        int makespan;                // �ؼШ�ƭ� (�A����)

        // ��K�ƧǡGMakespan �V�p�V�n
        bool operator<(const Individual &other) const{
            return makespan < other.makespan;
        }
    };
    struct MAResult{
        std::vector<int> chromosome;
        int makespan;
        int total_gen;
        double runtime_sec;
    };

    Individual generate_random_individual(int &ffe_counter){
        Individual ind;
        ind.chromosome.resize(num_jobs);
        for(int i = 0; i < num_jobs; ++i){
            ind.chromosome[i] = i;
        }
        std::shuffle(ind.chromosome.begin(), ind.chromosome.end(), rng);
        ind.makespan = evaluate(ind.chromosome, ffe_counter);
        return ind;
    }

private:

    Scheduling &evaluator;
    int num_jobs;
    std::mt19937 rng;
    std::vector<int> makespan_history, best_makespan_history, median_makespan_history;
    // ���ͳ�@�H������


    // --- �s�W�GMA ��ܾ��� (Tournament Selection) ---
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

    // --- �s�W�GFFE�����w��W�� ---
    int evaluate(const std::vector<int> &chromosome, int &ffe_counter){
        ffe_counter++; // �C�I�s�@���A�o�̴N�۰� +1
        return evaluator.run_scheduling(chromosome);
    }

    // --- �s�W�GNEH �غc���ҵo��k ---
    std::vector<int> generate_neh_solution(int &ffe_count){
        // �B�J 1 & 2�G�p��C�Ӥu�@���`�B�z�ɶ��A�åѤj��p�Ƨ�
        std::vector<std::pair<int, int>> job_totals(num_jobs);
        for(int i = 0; i < num_jobs; ++i){
            // �Q�� evaluator ������@�u�@ i ���ɶ� (���P��Ӥu�@�b�Ҧ��������ɶ��`�M)
            int total_time = evaluate({ i }, ffe_count);
            job_totals[i] = { total_time, i };
        }

        // �̷��`�ɶ������Ƨ� (�Ѥj��p)
        std::sort(job_totals.begin(), job_totals.end(), [](const auto &a, const auto &b){
            return a.first > b.first;
        });

        // �B�J 3�G�̧Ƕi��̨δ��J (Best Insertion)
        std::vector<int> current_seq;
        current_seq.push_back(job_totals[0].second); // ����J�`�ɶ��̪����u�@

        for(int i = 1; i < num_jobs; ++i){
            int job_to_insert = job_totals[i].second;
            int best_makespan = 1e9;
            int best_pos = -1;

            // ���մ��J�ثe�ǦC���Ҧ��i���m (�q 0 �� i)
            for(int pos = 0; pos <= i; ++pos){
                std::vector<int> temp_seq = current_seq;
                temp_seq.insert(temp_seq.begin() + pos, job_to_insert);

                // �����o�ӡu�����Ƶ{�v�� Makespan
                int current_makespan = evaluate(temp_seq, ffe_count);

                if(current_makespan < best_makespan){
                    best_makespan = current_makespan;
                    best_pos = pos;
                }
            }
            // �T�w�̨Φ�m��A�������J
            current_seq.insert(current_seq.begin() + best_pos, job_to_insert);
        }

        return current_seq;
    }

    // --- �s�W�GOrder Crossover (OX) ��t��l ---
    std::pair<Individual, Individual> order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. �H����ܨ�Ӥ��I (Cut points)�A��Ӥl�N�@�γo�դ��I
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // �O�����ǰ�]�w�g�Q��J�l�N��
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. �ƻs���I�d�򤺪���]��������l�N
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. ��e��ɳѾl����m
        int p2_idx = (cut2 + 1) % num_jobs;
        int off1_idx = (cut2 + 1) % num_jobs;

        int p1_idx = (cut2 + 1) % num_jobs;
        int off2_idx = (cut2 + 1) % num_jobs;

        for(int i = 0; i < num_jobs; ++i){
            // �B�z offspring1: �q Parent 2 �̧Ƕ��
            int gene_from_p2 = p2.chromosome[p2_idx];
            if(!in_offspring1[gene_from_p2]){
                offspring1.chromosome[off1_idx] = gene_from_p2;
                off1_idx = (off1_idx + 1) % num_jobs;
            }
            p2_idx = (p2_idx + 1) % num_jobs;

            // �B�z offspring2: �q Parent 1 �̧Ƕ��
            int gene_from_p1 = p1.chromosome[p1_idx];
            if(!in_offspring2[gene_from_p1]){
                offspring2.chromosome[off2_idx] = gene_from_p1;
                off2_idx = (off2_idx + 1) % num_jobs;
            }
            p1_idx = (p1_idx + 1) % num_jobs;
        }

        return { offspring1, offspring2 };
    }

    // --- �s�W�GLinear Order Crossover (LOX) ��t��l ---
    std::pair<Individual, Individual> linear_order_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. �H����ܨ�Ӥ��I (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // �O�����ǰ�]�w�g�Q��J�l�N��
        std::vector<bool> in_offspring1(num_jobs, false);
        std::vector<bool> in_offspring2(num_jobs, false);

        // 2. �ƻs���I�d�򤺪���]��������l�N (�o�����P OX �ۦP)
        for(int i = cut1; i <= cut2; ++i){
            offspring1.chromosome[i] = p1.chromosome[i];
            in_offspring1[p1.chromosome[i]] = true;

            offspring2.chromosome[i] = p2.chromosome[i];
            in_offspring2[p2.chromosome[i]] = true;
        }

        // 3. �����t�@�Ӥ��N���u�|���Q�ϥΡv����] (�O���ѥ��ܥk������)
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

        // 4. �ѥ��ܥk (Linear) �̧Ƕ�ɤl�N���Ŧ� (���L cut1 �� cut2 ���Ϭq)
        int idx1 = 0, idx2 = 0;
        for(int i = 0; i < num_jobs; ++i){
            // �p�G�O�b���I�d�򤺡A�������L (�]���w�g�b�B�J 2 ��n�F)
            if(i >= cut1 && i <= cut2) continue;

            offspring1.chromosome[i] = remaining_for_off1[idx1++];
            offspring2.chromosome[i] = remaining_for_off2[idx2++];
        }

        return { offspring1, offspring2 };
    }

    // --- �s�W�GPartially-Mapped Crossover (PMX) ��t��l ---
    std::pair<Individual, Individual> partially_mapped_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // 1. �H����ܨ�Ӥ��I (Cut points)
        std::uniform_int_distribution<int> dist(0, num_jobs - 1);
        int cut1 = dist(rng);
        int cut2 = dist(rng);
        if(cut1 > cut2) std::swap(cut1, cut2);

        // �O�����ǰ�]�w�g�u�s�b����I�d�򤺡v�A�ΥH�P�_�O�_�Ĭ�
        std::vector<bool> in_off1_middle(num_jobs, false);
        std::vector<bool> in_off2_middle(num_jobs, false);

        // �إ����V�M�g�� (Mapping dictionaries)
        std::vector<int> map_p1_to_p2(num_jobs, -1);
        std::vector<int> map_p2_to_p1(num_jobs, -1);

        // 2. �ƻs���I�d�򤺪���]�A�ëإ߬M�g���Y
        for(int i = cut1; i <= cut2; ++i){
            int gene1 = p1.chromosome[i];
            int gene2 = p2.chromosome[i];

            // ��J�l�N 1 �P �l�N 2 �������q��
            offspring1.chromosome[i] = gene1;
            in_off1_middle[gene1] = true;

            offspring2.chromosome[i] = gene2;
            in_off2_middle[gene2] = true;

            // �������۹������M�g���Y
            map_p1_to_p2[gene1] = gene2;
            map_p2_to_p1[gene2] = gene1;
        }

        // 3. ��ɤ��I�H�~���Ѿl��m
        for(int i = 0; i < num_jobs; ++i){
            // ���L�����w�g��n������
            if(i >= cut1 && i <= cut2) continue;

            // --- �B�z Offspring 1 (�~�� Parent 2 ���~���]) ---
            int candidate1 = p2.chromosome[i];
            // �p�G�o�ͽĬ� (�Ӱ�]�w�g�b Offspring 1 �������q�X�{�L)
            // �N�z�L�M�g����X���N��]�A�o�� while �j��৹���ѨM�s��M�g�����D
            while(in_off1_middle[candidate1]){
                candidate1 = map_p1_to_p2[candidate1];
            }
            offspring1.chromosome[i] = candidate1;

            // --- �B�z Offspring 2 (�~�� Parent 1 ���~���]) ---
            int candidate2 = p1.chromosome[i];
            // �P�˪��޿�A�ϦV�d��
            while(in_off2_middle[candidate2]){
                candidate2 = map_p2_to_p1[candidate2];
            }
            offspring2.chromosome[i] = candidate2;
        }

        return { offspring1, offspring2 };
    }

    // --- �s�W�GCycle Crossover (CX) ��t��l ---
    std::pair<Individual, Individual> cycle_crossover(const Individual &p1, const Individual &p2){
        Individual offspring1, offspring2;
        offspring1.chromosome.assign(num_jobs, -1);
        offspring2.chromosome.assign(num_jobs, -1);

        // �إ߯��ެd���G��H O(1) �ɶ����u�Y�Ӱ�]�b Parent 1 ������m�v
        std::vector<int> p1_index_of(num_jobs, 0);
        for(int i = 0; i < num_jobs; ++i){
            p1_index_of[p1.chromosome[i]] = i;
        }

        // �O�����Ǧ�m�w�g�Q Cycle ���X�L
        std::vector<bool> visited(num_jobs, false);

        // 1. ��X�Ĥ@�� Cycle (�o�̩T�w�q index 0 �}�l)
        int current_idx = 0;

        // ���٨S¶�^�_�I�Φ��j��ɡA�~��l��
        while(!visited[current_idx]){
            visited[current_idx] = true;

            // �b Cycle ���G
            // �l�N 1 �~�� Parent 1 �Ӧ�m����]
            offspring1.chromosome[current_idx] = p1.chromosome[current_idx];
            // �l�N 2 �~�� Parent 2 �Ӧ�m����]
            offspring2.chromosome[current_idx] = p2.chromosome[current_idx];

            // �M��U�@�� index�G
            // ��X Parent 2 �b���e��m����]�A�ìd�߸Ӱ�]�b Parent 1 ����m
            int val_in_p2 = p2.chromosome[current_idx];
            current_idx = p1_index_of[val_in_p2];
        }

        // 2. �N�D Cycle �d�򪺳Ѿl��m�A�Ρu�t�@�� Parent�v�������
        for(int i = 0; i < num_jobs; ++i){
            if(!visited[i]){
                // ���b Cycle ���G
                // �l�N 1 �~�� Parent 2 �Ӧ�m����]
                offspring1.chromosome[i] = p2.chromosome[i];
                // �l�N 2 �~�� Parent 1 �Ӧ�m����]
                offspring2.chromosome[i] = p1.chromosome[i];
            }
        }

        return { offspring1, offspring2 };
    }


    // --- �s�W�G���ܺ�l (Swap Mutation) ---
    void mutate(Individual &ind, double mutation_rate, MutationType mutation_type = Insertion){

        if(mutation_type == Insertion){
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            if(prob(rng) < mutation_rate){
                std::uniform_int_distribution<int> dist(0, num_jobs - 1);
                int idx1 = dist(rng);
                int idx2 = dist(rng);
                while(idx1 == idx2) idx2 = dist(rng);

                // ��X idx1 ���u�@�A�ô��J�� idx2 ����m
                int job = ind.chromosome[idx1];
                ind.chromosome.erase(ind.chromosome.begin() + idx1);
                ind.chromosome.insert(ind.chromosome.begin() + idx2, job);
            }
        }
        else if(mutation_type == Swap){
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            if(prob(rng) < mutation_rate){
                std::uniform_int_distribution<int> dist(0, num_jobs - 1);
                int idx1 = dist(rng);
                int idx2 = dist(rng);
                while(idx1 == idx2) idx2 = dist(rng);
                std::swap(ind.chromosome[idx1], ind.chromosome[idx2]);
            }
        }
    }

    // --- �s�W�GTabu Search (swap) �@������ local search ---
    void sa_insertion_local_search(
        Individual &ind, int &ffe_counter, int max_ffe, int max_ls_iter = 20, double init_temp = 50, double cooling_rate = 0.90){

        if(ind.chromosome.empty() || ffe_counter >= max_ffe) return;

        std::vector<int> current = ind.chromosome;
        int current_makespan = ind.makespan;
        if(current_makespan == 0){
            current_makespan = evaluate(current, ffe_counter);
        }

        std::vector<int> best = current;
        int best_makespan = current_makespan;

        double temperature = init_temp;
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        for(int iter = 0; iter < max_ls_iter && ffe_counter < max_ffe; ++iter){
            bool moved = false;

            for(int idx_from = 0; idx_from < num_jobs && !moved && ffe_counter < max_ffe; ++idx_from){
                for(int idx_to = 0; idx_to < num_jobs && !moved && ffe_counter < max_ffe; ++idx_to){
                    if(idx_from == idx_to) continue;

                    std::vector<int> neighbor = current;

                    int job = neighbor[idx_from];
                    neighbor.erase(neighbor.begin() + idx_from);
                    neighbor.insert(neighbor.begin() + idx_to, job);

                    int neighbor_makespan = evaluate(neighbor, ffe_counter);
                    int delta = neighbor_makespan - current_makespan;

                    bool accept = false;

                    if(delta < 0){
                        accept = true;
                    }
                    else if(temperature > 1e-9){
                        double accept_prob = std::exp(-static_cast<double>(delta) / temperature);
                        if(prob(rng) < accept_prob){
                            accept = true;
                        }
                    }

                    if(accept){
                        current = neighbor;
                        current_makespan = neighbor_makespan;
                        moved = true;

                        if(current_makespan < best_makespan){
                            best = current;
                            best_makespan = current_makespan;
                        }
                    }
                }
            }

            if(!moved) break;

            temperature *= cooling_rate;
        }

        ind.chromosome = best;
        ind.makespan = best_makespan;
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

            // �P�_�O�_������Ƹ�� (MA �M��)
            bool has_median = !median_makespan_history.empty();

            if(has_median){
                fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', "
                    "'-' with lines title 'Current/Gen Best Makespan' lc rgb 'red', "
                    "'-' with lines title 'Population Median' lc rgb 'green'\n");
            }
            else{
                fprintf(pipe, "plot '-' with lines title 'Best Makespan' lc rgb 'blue', "
                    "'-' with lines title 'Current Makespan' lc rgb 'red'\n");
            }

            // �ǰe�̨θѸ�� (�Žu)
            for(size_t i = 0; i < best_makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, best_makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            // �ǰe���N/�����Ѹ�� (���u)
            for(size_t i = 0; i < makespan_history.size(); ++i){
                fprintf(pipe, "%zu %d\n", i, makespan_history[i]);
            }
            fprintf(pipe, "e\n");

            // �p�G������ơA�ǰe����Ƹ�� (��u)
            if(has_median){
                for(size_t i = 0; i < median_makespan_history.size(); ++i){
                    fprintf(pipe, "%zu %d\n", i, median_makespan_history[i]);
                }
                fprintf(pipe, "e\n");
            }

            fflush(pipe);
            pclose(pipe);
        }
    }

    std::string cross_str(MetaheuristicSolver::CrossoverType type){
        switch(type){
            case OX: return "OX";
            case LOX: return "LOX";
            case PMX: return "PMX";
            case CX: return "CX";
            default: return "Unknown";
        }
    }

public:
    MetaheuristicSolver(Scheduling &sched, int jobs) : evaluator(sched), num_jobs(jobs){
        std::random_device rd;
        rng = std::mt19937(rd());
    }



    // ==========================================
    // �g�]�t��k (Memetic Algorithm) �D�{��
    // ==========================================
    MAResult memetic_algorithm(
        int pop_size = 50,
        double crossover_rate = 0.8, double mutation_rate = 0.1,
        CrossoverType crossover_type = OX, bool with_tabu = false,
        bool dynamic_tabu = false, bool do_neh = false, bool use_sa_local_search = false, int ls_gen_num = 20, int ls_iters = 20, double init_temp = 20, double cooling_rate = 0.9,
        LSSelectionType ls_selection_type = Elite, MutationType mutation_type = Insertion, std::string instance_name = "Instance", std::string save_dir = "./img"){

        std::cout << "Run MA with " << cross_str(crossover_type) << (with_tabu ? " (Tabu Enabled)" : "") << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now(); //�p�ɶ}�l

        best_makespan_history.clear();
        makespan_history.clear();
        median_makespan_history.clear();

        // 1. ��l�Ʊڸs (Initialization)
        int current_ffe = 0;
        int max_ffe = pop_size * 1000;
        std::vector<Individual> population;
        for(int i = 0; i < pop_size; ++i){
            // �[�J NEH �u�����
            if(do_neh && i == 0){
                Individual neh_seed;
                neh_seed.chromosome = generate_neh_solution(current_ffe);
                neh_seed.makespan = evaluate(neh_seed.chromosome, current_ffe);
                population.push_back(neh_seed);
                std::cout << "  [Info] Injected NEH Seed with Makespan: " << neh_seed.makespan << std::endl;
            }
            // �ڸs���[�J Random ����
            else{
                population.push_back(generate_random_individual(current_ffe));
            }
        }

        // �M���l�̨θ�
        Individual global_best = *std::min_element(population.begin(), population.end());

        std::uniform_real_distribution<double> prob(0.0, 1.0);

        // Tabu Hash �����]�w (�Ȧb with_tabu == true �ɵo���@��)
        std::unordered_map<uint64_t, int> tabu_list;
        int gen_tabu_tenure = 10;

        auto compute_hash = [](const std::vector<int> &arr) -> uint64_t{
            uint64_t hash_val = 0;
            uint64_t p_pow = 1;
            const uint64_t p = 313;
            for(size_t i = 0; i < arr.size(); ++i){
                hash_val += (arr[i] + 1) * p_pow;
                p_pow *= p;
            }
            return hash_val;
        };

        // �@�N�t�ưj��
        int gen = 0; // �O�d gen �ȥΩ�έp�P�e�ϡA���A�@�������ܼ�
        while(current_ffe < max_ffe){
            gen++;

            std::vector<Individual> next_generation;
            std::unordered_set<uint64_t> current_pop_hashes;

            // �׭^�O�d���� (Elitism)
            Individual current_best = *std::min_element(population.begin(), population.end());
            next_generation.push_back(current_best);

            if(with_tabu){
                // 1. �p��i�פ�� (0.0 �� 1.0)
                double progress = static_cast<double>(current_ffe) / max_ffe;

                
                if(dynamic_tabu){
                    // 2. �u�ʰI����G��l tenure �� 10�A�H�� FFE �W�[�I��� 0
                    gen_tabu_tenure = 10 - static_cast<int>(progress * 10);

                    
                    // 3. �T�O���p�� 0
                    gen_tabu_tenure = std::max(1, gen_tabu_tenure);
                    // std::cout << "  [Info] Generation: " << gen << ", FFE: " << current_ffe << ", Progress: " << (progress * 100) << "%, Tabu Tenure: " << gen_tabu_tenure << std::endl;
                    if(gen_tabu_tenure == 1){
                        std::cout << "  [Info] Dynamic Tabu Tenure reached 0 at FFE: " << current_ffe << std::endl;
                        with_tabu = false; // Tabu List ���A
                    }
                }
                else{
                    gen_tabu_tenure = 10; // ���� static tabu tenure
                }
                // 4. ��s Tabu List
                // �`�N�G�o�̪����ϥη��e�� tabu_tenure�A���ݭn�B�~�P�_
                uint64_t best_hash = compute_hash(current_best.chromosome);
                current_pop_hashes.insert(best_hash);

                // �� tabu_tenure �� 0 �ɡA�g�J gen + 0�A�T���ˬd�|�۰ʥ���
                tabu_list[best_hash] = gen + gen_tabu_tenure;
            }

            // ���ͤU�@�N (����ƶq���� pop_size)
            while(next_generation.size() < pop_size){
                Individual p1 = tournament_selection(population);
                Individual p2 = tournament_selection(population);
                Individual off1, off2;
                // ====================================================
                // ���| A�G�쥻���º� MA (�����歫�Ʊư��P�ƧǸT��)
                // ====================================================
                if(!with_tabu){
                    if(prob(rng) < crossover_rate){
                        if(crossover_type == OX){
                            auto offsprings = order_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == LOX){
                            auto offsprings = linear_order_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == PMX){
                            auto offsprings = partially_mapped_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                        else if(crossover_type == CX){
                            auto offsprings = cycle_crossover(p1, p2);
                            off1 = offsprings.first; off2 = offsprings.second;
                        }
                    }
                    else{
                        off1 = (prob(rng) < 0.5) ? p1 : p2;
                        off2 = (prob(rng) < 0.5) ? p1 : p2;
                    }

                    // ���� (Mutation)
                    mutate(off1, mutation_rate, mutation_type);
                    mutate(off2, mutation_rate, mutation_type);

                    off1.makespan = evaluate(off1.chromosome, current_ffe);
                    off2.makespan = evaluate(off2.chromosome, current_ffe);

                    next_generation.push_back(off1);
                    if(next_generation.size() < pop_size){
                        next_generation.push_back(off2);
                    }
                }
                // ====================================================
                // ���| B�G�[�W Tabu �P�P�N���Ʊư���� MA
                // ====================================================
                else{
                    bool off1_valid = false;
                    bool off2_valid = false;
                    int retries = 0;

                    while(retries < 10 && (!off1_valid || !off2_valid)){
                        Individual temp1 = p1, temp2 = p2;

                        // ��t (Crossover)
                        if(prob(rng) < crossover_rate){
                            if(crossover_type == OX){
                                auto offsprings = order_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == LOX){
                                auto offsprings = linear_order_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == PMX){
                                auto offsprings = partially_mapped_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                            else if(crossover_type == CX){
                                auto offsprings = cycle_crossover(p1, p2);
                                temp1 = offsprings.first; temp2 = offsprings.second;
                            }
                        }

                        // ���� (Mutation)
                        mutate(temp1, mutation_rate, mutation_type);
                        mutate(temp2, mutation_rate, mutation_type);

                        uint64_t h1 = compute_hash(temp1.chromosome);
                        uint64_t h2 = compute_hash(temp2.chromosome);

                        if(!off1_valid){
                            bool is_tabu = ((tabu_list.count(h1) && tabu_list[h1] > gen) || current_pop_hashes.count(h1));
                            if(!is_tabu){
                                temp1.makespan = evaluate(temp1.chromosome, current_ffe);
                                off1 = temp1;
                                off1_valid = true;
                                current_pop_hashes.insert(h1);
                                tabu_list[h1] = gen + gen_tabu_tenure;
                            }
                        }

                        if(!off2_valid && (next_generation.size() + (off1_valid ? 1 : 0) < pop_size)){
                            bool is_tabu = ((tabu_list.count(h2) && tabu_list[h2] > gen) || current_pop_hashes.count(h2));
                            if(!is_tabu){
                                temp2.makespan = evaluate(temp2.chromosome, current_ffe);
                                off2 = temp2;
                                off2_valid = true;
                                current_pop_hashes.insert(h2);
                                tabu_list[h2] = gen + gen_tabu_tenure;
                            }
                        }
                        else if(next_generation.size() + (off1_valid ? 1 : 0) >= pop_size){
                            off2_valid = true;
                        }

                        retries++;
                    }

                    if(!off1_valid){
                        off1 = generate_random_individual(current_ffe);
                        uint64_t h1 = compute_hash(off1.chromosome);
                        current_pop_hashes.insert(h1);
                        tabu_list[h1] = gen + gen_tabu_tenure;
                    }
                    next_generation.push_back(off1);

                    if(next_generation.size() < pop_size){
                        if(!off2_valid){
                            off2 = generate_random_individual(current_ffe);
                            uint64_t h2 = compute_hash(off2.chromosome);
                            current_pop_hashes.insert(h2);
                            tabu_list[h2] = gen + gen_tabu_tenure;
                        }
                        next_generation.push_back(off2);
                    }
                } // End of else (���| B)
            }

            // ��s�ڸs
            population = next_generation;

            // �����N�̨έ���
            auto best_it = std::min_element(population.begin(), population.end());

            // Local search(SA), �Y�X�N�q�e5�W�H����1�Ӱ� local search
            if(gen % ls_gen_num == 0 && use_sa_local_search && current_ffe < max_ffe){
                if(ls_selection_type == Elite){
                    auto best_it = std::min_element(population.begin(), population.end());
                    if(best_it != population.end()){
                        sa_insertion_local_search(*best_it, current_ffe, max_ffe, ls_iters, init_temp, cooling_rate);
                    }
                }
                else if(ls_selection_type == Top5Random){
                    std::sort(population.begin(), population.end());

                    int top_k = std::min(5, static_cast<int>(population.size()));
                    std::uniform_int_distribution<int> top_dist(0, top_k - 1);
                    int selected_idx = top_dist(rng);

                    sa_insertion_local_search(population[selected_idx], current_ffe, max_ffe, ls_iters, init_temp, cooling_rate);
                }
            }


            // ��s���v�̨θ�
            current_best = *std::min_element(population.begin(), population.end());
            if(current_best.makespan < global_best.makespan){
                global_best = current_best;
            }

            // �p�⤤���
            std::vector<int> current_makespans(pop_size);
            for(int i = 0; i < pop_size; ++i){
                current_makespans[i] = population[i].makespan;
            }
            std::nth_element(current_makespans.begin(), current_makespans.begin() + pop_size / 2, current_makespans.end());
            int current_median = current_makespans[pop_size / 2];

            makespan_history.push_back(current_best.makespan);
            best_makespan_history.push_back(global_best.makespan);
            median_makespan_history.push_back(current_median);

        }

        std::cout << "MA Result: " << global_best.makespan << std::endl;
        std::string prefix = "MA_" + cross_str(crossover_type) ;
        save_plot(save_dir, prefix, instance_name, global_best.makespan);
        auto end_time = std::chrono::high_resolution_clock::now();
        double runtime_sec = std::chrono::duration<double>(end_time - start_time).count();
        return { global_best.chromosome, global_best.makespan, gen, runtime_sec };
    }

    void cross_test(int &current_ffe){
        std::cout << "Testing Order Crossover (OX)..." << std::endl;
        Individual ind1 = generate_random_individual(current_ffe);
        Individual ind2 = generate_random_individual(current_ffe);
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

};

void calculate_and_write_stats(std::ofstream &csv, const std::string &name, const std::vector<int> &data, const std::vector<int> &gens, const std::vector<double> &time){
    int min_val = *std::min_element(data.begin(), data.end());
    int max_val = *std::max_element(data.begin(), data.end());
    double avg_val = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum_sq_diff = 0.0;
    for(int val : data){
        sum_sq_diff += (val - avg_val) * (val - avg_val);
    }
    double std_dev = std::sqrt(sum_sq_diff / (data.size() > 1 ? data.size() - 1 : 1));
    double avg_gen = std::accumulate(gens.begin(), gens.end(), 0.0) / gens.size();
    double avg_runtime = std::accumulate(time.begin(), time.end(), 0.0) / time.size();

    csv << min_val << ","
        << std::fixed << std::setprecision(2) << avg_val << ","
        << max_val << ","
        << std::fixed << std::setprecision(2) << std_dev << ","
        << std::fixed << std::setprecision(2) << avg_gen << ","
        << std::fixed << std::setprecision(2) << avg_runtime << ",";
}

int main(){
    std::cout << "HI";
#define TABU_TEST false
    // MetaheuristicSolver solver(*(new Scheduling()), 10);
    // solver.cross_test();
    // /*
    std::string test_case_dir = "./Test_case";
    if(!fs::exists(test_case_dir) || !fs::is_directory(test_case_dir)){
        std::cerr << "�䤣���Ƨ�: " << test_case_dir << "\n";
        return 1;
    }

    std::ofstream csv_file("results.csv");
    csv_file << "Instance,LOX_iter20_Min,LOX_iter20_Avg,LOX_iter20_Max,LOX_iter20_Std,LOX_iter20_AvgGen,LOX_iter20_AvgTime,LOX_gen50_Min,LOX_gen50_Avg,LOX_gen50_Max,LOX_gen50_Std,LOX_gen50_AvgGen,LOX_gen50_AvgTime,LOX_gen100_Min,LOX_gen100_Avg,LOX_gen100_Max,LOX_gen100_Std,LOX_gen100_AvgGen,LOX_gen100_AvgTime\n";
    //csv_file << "Instance, OX_Min,OX_Avg,OX_Max,OX_Std,LOX_Min,LOX_Avg,LOX_Max,LOX_Std,PMX_Min,PMX_Avg,PMX_Max,PMX_Std,CX_Min,CX_Avg,CX_Max,CX_Std\n";
    //csv_file << "Instance,MA_LOX_Min,MA_LOX_Avg,MA_LOX_dev,MA_OX_neh_Min,MA_OX_neh_Avg,MA_OX_neh_dev\n";
    //csv_file << "Instance,MA_OX_Min,MA_OX_Avg,MA_OX_Max,MA_LOX_Min,MA_LOX_Avg,MA_LOX_Max,MA_PMX_Min,MA_PMX_Avg,MA_PMX_Max,MA_CX_Min,MA_CX_Avg,MA_CX_Max\n";

    int NUM_RUNS = 20;

    for(const auto &entry : fs::directory_iterator(test_case_dir)){
        if(entry.path().extension() == ".txt"){
            std::string filepath = entry.path().string();
            std::cout << "========================================\n";
            std::cout << "�}�l����: " << filepath << " (�@ " << NUM_RUNS << " ���������)\n";

            TestCase tc;
            if(!read_taillard_file(filepath, tc)) continue;

            Scheduling scheduler;
            scheduler.job_mtx = tc.job_mtx;

            std::vector<int> lox_results(NUM_RUNS);
            std::vector<int> lox_results2(NUM_RUNS);
            std::vector<int> lox_results3(NUM_RUNS);
            std::vector<int> lox_gen(NUM_RUNS);
            std::vector<int> lox_gen2(NUM_RUNS);
            std::vector<int> lox_gen3(NUM_RUNS);
            std::vector<double> lox_time(NUM_RUNS);
            std::vector<double> lox_time2(NUM_RUNS);
            std::vector<double> lox_time3(NUM_RUNS);



            for(int run = 1; run <= NUM_RUNS; ++run){
                fs::create_directories("./img/Test_" + std::to_string(run));
            }

#pragma omp parallel for schedule(dynamic)
            for(int run = 1; run <= NUM_RUNS; ++run){
                std::string save_dir = "./img/Test_" + std::to_string(run);

                MetaheuristicSolver solver(scheduler, tc.num_jobs);

                auto lox_res = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, false, false, true, false, 20, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "LOX_Without_Tabu", save_dir);
                lox_results[run - 1] = lox_res.makespan;
                lox_gen[run - 1] = lox_res.total_gen;
                lox_time[run - 1] = lox_res.runtime_sec;
                auto lox_res2 = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, true, false, true, false, 50, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "LOX_Static_Tabu", save_dir);
                lox_results2[run - 1] = lox_res2.makespan;
                lox_gen2[run - 1] = lox_res2.total_gen;
                lox_time2[run - 1] = lox_res2.runtime_sec;
                auto lox_res3 = solver.memetic_algorithm(50, 0.8, 0.5,
                    MetaheuristicSolver::LOX, true, true, true, false, 100, 50, 200, 0.95, MetaheuristicSolver::LSSelectionType::Top5Random, MetaheuristicSolver::MutationType::Insertion,
                    tc.instance_name + "LOX_Dynamic_Tabu", save_dir);
                lox_results3[run - 1] = lox_res3.makespan;
                lox_gen3[run - 1] = lox_res3.total_gen;
                lox_time3[run - 1] = lox_res3.runtime_sec;


#pragma omp critical
                {
                    std::cout << "  - �� " << run << " �����槹���A�Ϥ��w�x�s�� " << save_dir << "\n";
                }
            }
            csv_file << tc.instance_name << ",";
            calculate_and_write_stats(csv_file, "LOX_Without_Tabu", lox_results, lox_gen, lox_time);
            calculate_and_write_stats(csv_file, "LOX_Static_Tabu", lox_results2, lox_gen2, lox_time2);
            calculate_and_write_stats(csv_file, "LOX_Dynamic_Tabu", lox_results3, lox_gen3, lox_time3);
            csv_file << "\n";

            std::cout << ">> " << tc.instance_name << " �έp��Ƥw�g�J results.csv\n";
        }
    }

    csv_file.close();
    std::cout << "�Ҧ�������槹���I���G�w�ץX�� results.csv\n";
    // */
    return 0;
}