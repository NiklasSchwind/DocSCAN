import multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
import time
from datetime import datetime
import os




#Experiments = ['PYTHONPATH=src python src/NLPScan.py --path 20newsgroup --embeddings_', 'PYTHONPATH=src python src/test.py', 'PYTHONPATH=src python src/test.py']
'''
Experiments = [	{'--embedding_model': 'SBert', '--path': '20newsgroup','--clustering_method': 'SCANLoss', '--model_method': 'PrototypeBert', '--threshold': 0.99,'--num_epochs': 2  },
				{'--embedding_model': 'SBert', '--path': 'TREC-6','--clustering_method': 'SCANLoss', '--model_method': 'PrototypeBert','--threshold': 0.99,'--num_epochs': 2 },
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': '20newsgroup','--clustering_method': 'SCANLoss', '--model_method': 'PrototypeBert', '--threshold': 0.95,'--num_epochs': 2  },
				{'--embedding_model': 'SBert', '--path': 'TREC-6','--clustering_method': 'SCANLoss', '--model_method': 'PrototypeBert','--threshold': 0.95,'--num_epochs': 2 },
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'PrototypeBert', '--threshold': 0.95, '--num_epochs': 5},
			{'--embedding_model': 'SBert', '--path': '20newsgroup','--clustering_method': 'SCANLoss', '--model_method': 'DocBert', '--threshold': 0.99,'--num_epochs': 2  },
				{'--embedding_model': 'SBert', '--path': 'TREC-6','--clustering_method': 'SCANLoss', '--model_method': 'DocBert','--threshold': 0.99,'--num_epochs': 2 },
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 2},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 5},
				   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
					'--model_method': 'DocBert', '--threshold': 0.99, '--num_epochs': 5},

				   ]
'''

Experiments = [
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5},
	{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5}
]

for experiment in Experiments:
	experiment['--model_method'] = 'DocSCAN_finetuning'
	experiment2 = experiment
	experiment['--augmentation_method'] = 'Backtranslation_fr_en'
	experiment2['--augmentation_method'] = 'Cropping'
	Experiments.append(experiment2)

def start_experiment(experiment, device):
	if experiment["--augmentation_method"] == 'Backtranslation_fr_en':
		outfile = f'LogsSelflabeling/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}.txt'
	else:
		outfile = f'LogsSelflabeling/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}_ratio_0.2.txt'

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











