import os
import IOModule.Debug_log as Class_Debug_log
import IOModule.Output_log as Class_Output_log
from time import time

import CqSim.Job_trace as Class_Job_trace
import CqSim.Backfill as Class_Backfill
import CqSim.Start_window as Class_Start_window
import CqSim.Basic_algorithm as Class_Basic_algorithm
import CqSim.Info_collect as Class_Info_collect

import Extend.SWF.Filter_job_SWF as filter_job_ext
import Extend.SWF.Filter_node_SWF as filter_node_ext
import Extend.SWF.Node_struc_SWF as node_struc_ext

import Trainer.PG_Trainer as pg_trainer
import Trainer.A2C_Trainer as a2c_trainer
import Trainer.PPO_Trainer as ppo_trainer
import Trainer.FCFS as FCFS


def cqsim_main(para_list):
    print("....................")
    for item in para_list:
        print(str(item) + ": " + str(para_list[item]))
    print("....................")

    trace_name = para_list['path_in'] + para_list['job_trace']
    save_name_j = para_list['path_fmt'] + \
        para_list['job_save'] + para_list['ext_fmt_j']
    config_name_j = para_list['path_fmt'] + \
        para_list['job_save'] + para_list['ext_fmt_j_c']
    struc_name = para_list['path_in'] + para_list['node_struc']
    save_name_n = para_list['path_fmt'] + \
        para_list['node_save'] + para_list['ext_fmt_n']
    config_name_n = para_list['path_fmt'] + \
        para_list['node_save'] + para_list['ext_fmt_n_c']

    output_sys = para_list['path_out'] + \
        para_list['output'] + para_list['ext_si']
    output_adapt = para_list['path_out'] + \
        para_list['output'] + para_list['ext_ai']
    output_result = para_list['path_out'] + \
        para_list['output'] + para_list['ext_jr']
    output_fn = {'sys': output_sys,
                 'adapt': output_adapt, 'result': output_result}
    log_freq_int = para_list['log_freq']
    read_input_freq = para_list['read_input_freq']

    if not os.path.exists(para_list['path_fmt']):
        os.makedirs(para_list['path_fmt'])

    if not os.path.exists(para_list['path_out']):
        os.makedirs(para_list['path_out'])

    if not os.path.exists(para_list['path_debug']):
        os.makedirs(para_list['path_debug'])

    # Debug
    print(".................... Debug")
    debug_path = para_list['path_debug'] + \
        para_list['debug'] + para_list['ext_debug']
    module_debug = Class_Debug_log.Debug_log(
        lvl=para_list['debug_lvl'], show=2, path=debug_path, log_freq=log_freq_int)
    # module_debug.start_debug()

    # Job Filter
    print(".................... Job Filter")
    module_filter_job = filter_job_ext.Filter_job_SWF(
        trace=trace_name, save=save_name_j, config=config_name_j, debug=module_debug)
    module_filter_job.feed_job_trace()
    # module_filter_job.read_job_trace()
    # module_filter_job.output_job_data()
    module_filter_job.output_job_config()

    # Node Filter
    print(".................... Node Filter")
    module_filter_node = filter_node_ext.Filter_node_SWF(
        struc=struc_name, save=save_name_n, config=config_name_n, debug=module_debug)
    module_filter_node.read_node_struc()
    module_filter_node.output_node_data()
    module_filter_node.output_node_config()

    # Job Trace
    print(".................... Job Trace")
    module_job_trace = Class_Job_trace.Job_trace(start=para_list['start'], num=para_list['read_num'], anchor=para_list['anchor'],
                                                 density=para_list['cluster_fraction'], read_input_freq=para_list['read_input_freq'], debug=module_debug)
    module_job_trace.initial_import_job_file(save_name_j)
    # module_job_trace.import_job_file(save_name_j)
    module_job_trace.import_job_config(config_name_j)

    # Node Structure
    print(".................... Node Structure")
    module_node_struc = node_struc_ext.Node_struc_SWF(debug=module_debug)
    module_node_struc.import_node_file(save_name_n)
    module_node_struc.import_node_config(config_name_n)

    # Backfill
    print(".................... Backfill")
    module_backfill = Class_Backfill.Backfill(
        mode=para_list['backfill'], node_module=module_node_struc, debug=module_debug, para_list=para_list['bf_para'])

    # Start Window
    print(".................... Start Window")
    module_win = Class_Start_window.Start_window(
        mode=para_list['win'], node_module=module_node_struc, debug=module_debug, para_list=para_list['win_para'], para_list_ad=para_list['ad_win_para'])

    # Basic Algorithm
    print(".................... Basic Algorithm")
    module_alg = Class_Basic_algorithm.Basic_algorithm(
        element=[para_list['alg'], para_list['alg_sign']], debug=module_debug, para_list=para_list['ad_alg_para'])

    # Information Collect
    print(".................... Information Collect")
    module_info_collect = Class_Info_collect.Info_collect(
        alg_module=module_alg, debug=module_debug)

    # Output Log
    print(".................... Output Log")
    module_output_log = Class_Output_log.Output_log(
        output=output_fn, log_freq=log_freq_int)

    # CqSim Simulator with RL
    print(".................... Cqsim Simulator using RL")
    module_list = {'job': module_job_trace, 'node': module_node_struc, 'backfill': module_backfill,
                   'win': module_win, 'alg': module_alg, 'info': module_info_collect, 'output': module_output_log}
    job_cols = int(para_list['job_info_size']) // int(para_list['input_dim'])
    batch_size = int(para_list['batch_size'])
    window_size = int(para_list['window_size'])
    learning_rate = float(para_list['learning_rate'])
    reward_discount = float(para_list['reward_discount'])
    is_training = True if (
        para_list['is_training'] == '1' or para_list['is_training'] == 1) else False
    input_weight_file = para_list['input_weight_file']
    output_weight_file = para_list['output_weight_file']
    do_render = True if para_list['do_render'] == '1' else False
    layer_size = para_list['layer_size']

    # Invoking the CqGym and PG model. This function manages the parameters required for initialization the
    # Gym Environment along with CqSim simulator and also for loading RL model - PG.
    if para_list['rl_alg'] == 'PPO':
        ppo_trainer.model_engine(module_list, module_debug, job_cols, window_size, module_node_struc.tot,
                                 is_training, input_weight_file, output_weight_file, do_render, learning_rate, reward_discount, batch_size, layer_size)
    elif para_list['rl_alg'] == 'A2C':
        a2c_trainer.model_engine(module_list, module_debug, job_cols, window_size, module_node_struc.tot,
                                 is_training, input_weight_file, output_weight_file, do_render, learning_rate, reward_discount, batch_size, layer_size)
    elif para_list['rl_alg'] == 'PG':
        pg_trainer.model_engine(module_list, module_debug, job_cols, window_size, module_node_struc.tot,
                                is_training, input_weight_file, output_weight_file, do_render, learning_rate, reward_discount, batch_size, layer_size)
    else:  # FCFS
        print(".................... FCFS")
        FCFS.model_engine(module_list, module_debug,
                          job_cols, window_size, do_render)
