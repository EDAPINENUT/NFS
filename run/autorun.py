import os
from multiprocessing import Process,Manager
import signal
import warnings
warnings.filterwarnings('always')

cmd =  [
        'CUDA_VISIBLE_DEVICES={} ' + 'python GST_AGCRN_NS_10_20.py',
        'CUDA_VISIBLE_DEVICES={} ' + 'python GST_AGCRN_NS_10_40.py']



def run(command,gpuid,gpustate):
    os.system(command.format(gpuid))
    gpustate[str(gpuid)]=True

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
    except Exception as e:
        print(str(e))

GPU_ID = [0]

if __name__ =='__main__':
    signal.signal(signal.SIGTERM, term)
    gpustate=Manager().dict({str(i):True for i in GPU_ID})
    processes=[]
    idx=0
    while idx<len(cmd):
        for gpuid in GPU_ID:
            if gpustate[str(gpuid)]==True:
                print(idx)
                gpustate[str(gpuid)]=False
                p=Process(target=run,args=(cmd[idx],gpuid,gpustate),name=str(gpuid))
                p.start()
                print(gpustate)
                processes.append(p)
                idx+=1
                break

    for p in processes:
        p.join()
