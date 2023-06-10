import multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
import time
from datetime import datetime
import os
import copy


Experiments = [
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 7.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 7.5},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 8.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 8.5},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 9.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 9.5},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 10.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 11.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 12.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 15.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first', '--entropy_weight': 20.0},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.01},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.02},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.03},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.05},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.075},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.1},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.15},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.2},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.01},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.02},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.03},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.05},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.075},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.1},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.15},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion', '--ratio_for_deletion': 0.2}





]




def start_experiment(experiment, device):
	#if (experiment["--augmentation_method"] == 'Backtranslate_en_fr' or experiment["--augmentation_method"] == 'Summarization' or experiment["--augmentation_method"] == 'Backtranslate_en_de' or experiment["--augmentation_method"] == 'Backtranslate_en_de_fr') and experiment['--embedding_model'] == 'IndicativeSentence':
	#	outfile = f'IndicativeSentencesExperimentLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_t5_model_{experiment["--t5_model"]}.txt'#_threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	#if ((experiment["--augmentation_method"] == 'Backtranslate_en_fr' or experiment["--augmentation_method"] == 'Summarization' or experiment["--augmentation_method"] == 'Backtranslate_en_de' or experiment["--augmentation_method"] == 'Backtranslate_en_de_fr')): #and experiment['--embedding_model'] != 'IndicativeSentence':
	#	outfile = f'RealSelflabelingExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}_t5_model_{experiment["--t5_model"]}.txt'
	#elif (experiment["--augmentation_method"] != 'Backtranslate_en_fr' and experiment["--augmentation_method"] != 'Summarization' and experiment["--augmentation_method"] != 'Backtranslate_en_de' and experiment["--augmentation_method"] != 'Backtranslate_en_de_fr') and experiment['--embedding_model'] == 'IndicativeSentence':
	#	outfile = f'IndicativeSentencesExperimentLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}.txt'  # _threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	#else:
	#	outfile = f'RealSelflabelingExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}.txt'
	if experiment['--model_method'] == 'PrototypeAccuracy' and experiment['--embedding_model'] != 'IndicativeSentence':
		outfile = f'PrototypeAccuracy/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}.txt'
	elif experiment['--model_method'] == 'PrototypeAccuracy' and experiment['--embedding_model'] == 'IndicativeSentence':
		outfile = f'PrototypeAccuracy/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_entropy_weight_{experiment["--entropy_weight"]}_threshold_{experiment["--threshold"]}.txt'
	elif experiment['--model_method'] == 'DocSCAN_finetuning_multi':
		outfile = f'DeletionRatioExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	else:
		outfile = f'EntropyWeightExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_entropy_weight_{experiment["--entropy_weight"]}.txt'

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











