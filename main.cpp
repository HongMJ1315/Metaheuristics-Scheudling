#include<iostream>
#include<vector>


class Scheduling{
    struct Machine{
        int job_id;
        int time;
    };
public:
    std::vector<std::vector<int>> job_mtx; //[machine][job]

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

int main(){
    Scheduling scheduler;
    scheduler.job_mtx = {
        {5, 2, 1},
        {3, 5, 3},
        {4, 2, 2}
    };

    std::vector<int> job_order = {1, 2, 0}; // Job indices
    int total_time = scheduler.run_scheduling(job_order);
    std::cout << "Total time to complete all jobs: " << total_time << std::endl;

    return 0;
}