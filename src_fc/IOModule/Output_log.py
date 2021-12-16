import IOModule.Log_print as Log_print

__metaclass__ = type

class Output_log:
    def __init__(self, output = None, log_freq = 1):
        self.myInfo = "Output_log"
        self.output_path = output
        self.sys_info_buf = []
        self.job_buf = []
        self.log_freq = log_freq
        self.reset_output()
    
    def reset(self, output = None, log_freq = 1):
        if output:
            self.output_path = output
            self.sys_info_buf = []
            self.job_buf = []
            self.log_freq = log_freq
            self.reset_output()

    def reset_output(self):   
        self.sys_info = Log_print.Log_print(self.output_path['sys'],0)
        self.sys_info.reset(self.output_path['sys'],0)
        self.sys_info.file_open()
        self.sys_info.file_close()
        self.sys_info.reset(self.output_path['sys'],1)   
        
        self.adapt_info = Log_print.Log_print(self.output_path['adapt'],0)
        self.adapt_info.reset(self.output_path['adapt'],0)
        self.adapt_info.file_open()
        self.adapt_info.file_close()
        self.adapt_info.reset(self.output_path['adapt'],1)
        
        self.job_result = Log_print.Log_print(self.output_path['result'],0)
        self.job_result.reset(self.output_path['result'],0)
        self.job_result.file_open()
        self.job_result.file_close()
        self.job_result.reset(self.output_path['result'],1)              

        self.reward_result = Log_print.Log_print(self.output_path['reward'],0)
        self.reward_result.reset(self.output_path['reward'],0)
        self.reward_result.file_open()
        self.reward_result.file_close()
        self.reward_result.reset(self.output_path['reward'],1)              
            

    def print_sys_info(self, sys_info = None):
        if sys_info != None:
            self.sys_info_buf.append(sys_info)
        if (len(self.sys_info_buf) >= self.log_freq) or (sys_info == None):
            sep_sign=";"
            #pre_context = "Printing..............................\n"
            self.sys_info.file_open()
            for sys_info in self.sys_info_buf:
                context = ""
                context += str(int(sys_info['date']))
                context += sep_sign
                context += (str(sys_info['uti']))
                context += sep_sign
                self.sys_info.log_print(context,1)
            self.sys_info.file_close()
            self.sys_info_buf = []
        
    
    def print_result(self, job_module, job_index = None):
        if job_index != None:
            self.job_buf.append(job_module.job_info(job_index))
        if (len(self.job_buf) >= self.log_freq) or (job_index == None):
            self.job_result.file_open()
            sep_sign=";"
            for temp_job in self.job_buf:
                #temp_job = job_module.job_info(job_index)
                context = ""
                context += str(temp_job['id'])
                context += sep_sign
                context += str(temp_job['reqProc'])
                context += sep_sign
                context += str(temp_job['reqTime'])
                context += sep_sign
                context += str(temp_job['run'])
                context += sep_sign
                context += str(temp_job['wait'])
                context += sep_sign
                context += str(temp_job['submit'])
                context += sep_sign
                context += str(temp_job['start'])
                context += sep_sign
                context += str(temp_job['end'])
                self.job_result.log_print(context,1)
            self.job_result.file_close()
            self.job_buf = []
    
    def print_reward(self, reward_seq):
        if reward_seq is not None:
            self.reward_result.file_open()
            for reward in reward_seq:
                self.reward_result.log_print(reward, 1)
            self.reward_result.file_close()