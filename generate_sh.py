num_gpus = 8
num_processes_each_gpu = 5
num_data_each_process = 7500
start_index = 0
data_root = '/data/nobn'

with open('run.sh', 'w') as f:
    for gpu in range(num_gpus):
        for process in range(num_processes_each_gpu):
            f.write('python construct_train.py --gpu_id ' + str(gpu) + 
                    ' --num_data ' + str(num_data_each_process) + 
                    ' --start_index ' + str(start_index) + 
                    ' --data_root ' + data_root + ' &\n')
            start_index += num_data_each_process