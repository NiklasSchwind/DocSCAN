import multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
import time
from datetime import datetime
import os




#Experiments = ['PYTHONPATH=src python src/NLPScan.py --path 20newsgroup --embeddings_', 'PYTHONPATH=src python src/test.py', 'PYTHONPATH=src python src/test.py']
Experiments = [	{'--embedding_model': 'SBert', '--path': '20newsgroup'},
				{'--embedding_model': 'SBert', '--path': 'IMDB'},
				{'--embedding_model': 'SBert', '--path': 'ag_news'},
				{'--embedding_model': 'SBert', '--path': 'TREC-6'},
				{'--embedding_model': 'SBert', '--path': 'TREC-50'},
				{'--embedding_model': 'SBert', '--path': 'DBPedia'},
				{'--embedding_model': 'SimCSEsupervised', '--path': '20newsgroup'},
				{'--embedding_model': 'SimCSEsupervised', '--path': 'IMDB'},
				{'--embedding_model': 'SimCSEsupervised', '--path': 'ag_news'},
				{'--embedding_model': 'SimCSEsupervised', '--path': 'TREC-6'},
				{'--embedding_model': 'SimCSEsupervised', '--path': 'TREC-50'},
				{'--embedding_model': 'SimCSEsupervised', '--path': 'DBPedia'},
				{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup'},
				{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB'},
				{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news'},
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6'},
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50'},
				{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia'},
				{'--embedding_model': 'TSDEA', '--path': '20newsgroup'},
				{'--embedding_model': 'TSDEA', '--path': 'IMDB'},
				{'--embedding_model': 'TSDEA', '--path': 'ag_news'},
				{'--embedding_model': 'TSDEA', '--path': 'TREC-6'},
				{'--embedding_model': 'TSDEA', '--path': 'TREC-50'},
				{'--embedding_model': 'TSDEA', '--path': 'DBPedia'}]

def start_experiment(experiment, device):
	outfile = f'NeighborLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_no3thstep_withNeighbors.txt'
	with open(outfile, 'w') as f:
		f.write('Start')
	experiment_prompt = 'PYTHONPATH=src python src/NLPScan.py'
	for key in experiment.keys():
		experiment_prompt = f'{experiment_prompt} {key} {experiment[key]}'
	print(f'Started {experiment_prompt} --device {device} --outfile {outfile}')
	os.system(f'{experiment_prompt} --device {device} --outfile {outfile}')
	print("Comment finished")


#processes = [mp.Process(target=start_experiment()) for experiment in Experiments]

nvmlInit()
deviceCount = nvmlDeviceGetCount()

if deviceCount > 0:
	CUDA = {i : nvmlDeviceGetHandleByIndex(i) for i in range(deviceCount)}
	device = 'CUDA'
	freeCUDA = {i:False for i in range(deviceCount)}
	count = 0
else:
	print('NO CUDA AVAILABLE')
	device = 'cpu'
	count = len(Experiments) + 1

processes = {}
possible_devices = list(range(deviceCount))
used_devices = []
process_device = {}

while count < len(Experiments):

	#See if any CUDA is free
	for i in possible_devices:
		info = nvmlDeviceGetMemoryInfo(CUDA[i])
		util_rate = nvmlDeviceGetUtilizationRates(CUDA[i]).memory
		gpu_cores = nvmlDeviceGetUtilizationRates(CUDA[i]).gpu
		if util_rate <= 30:# and gpu_cores == 0:
			freeCUDA[i] = True

	#If yes, relocate an experiment to CUDA and block cuda
	for i in possible_devices:
		if freeCUDA[i]:
			device = f'cuda:{i}'
			now = datetime.now()
			current_time = now.strftime("%H:%M:%S")
			print(count)
			#outfile = f"LogsCorrect/{Experiments[count].replace(' ','_').replace('.','_').replace('/','_')}_started_{current_time}.txt"
			processes[count] = mp.Process(target=start_experiment,args=(Experiments[count], device))
			processes[count].start()
			freeCUDA[i] = False
			possible_devices.remove(i)
			process_device[i] =  count
			used_devices.append(i)
			count += 1

	#check if experiments are done and deblock cuda if yes
	for i in used_devices:
		proc = processes[process_device[i]]
		proc.join(timeout=0)

		if proc.is_alive():
			pass
		else:
			possible_devices.append(i)
			used_devices.remove(i)
	print("Looking for free devices")
	time.sleep(10)

print("Finished all Experiments")











