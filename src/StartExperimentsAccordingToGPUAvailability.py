import multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
import time
from datetime import datetime
import os
import copy

'''
Experiments_proto = [#{'--path': 'ag_news', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'DBPedia', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
#{'--path': '20newsgroup', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
#{'--path': 'TREC-6', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
#{'--path': 'TREC-50', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
#{'--path': 'IMDB', '--model_method': 'PrototypeAccuracy', '--threshold': 0.99, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'ag_news', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'DBPedia', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,
{'--path': '20newsgroup', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'TREC-6', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'TREC-50', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,
{'--path': 'IMDB', '--model_method': 'PrototypeAccuracy', '--threshold': 0.95, '--num_epochs': 5, '--repetitions':10} ,

]
'''
Experiments_proto = [{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 10, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 10, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Summarization', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 10, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 10, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 10, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs':10, '--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 10, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
{'--path': 'ag_news_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'DBPedia_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': '20newsgroup', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-6', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'TREC-50', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 10, '--augmentation_method':'Backtranslate_en_de_fr', '--t5_model': 'base'},
			   {'--path': 'IMDB_smaller', '--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95,
				'--num_epochs': 5, '--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
]



optimal_indicative_sentence = {'DBPedia': 'Category: <mask>. ', 'DBPedia_smaller': 'Category: <mask>. ', 'ag_news': 'Category: <mask>. ', 'ag_news_smaller': 'Category: <mask>. ', '20newsgroup': 'Enjoy the following article about <mask>: ', 'TREC-6': ' <mask>.', 'TREC-50':' <mask>.', 'IMDB': ' All in all, it was <mask>.', 'IMDB_smaller': ' All in all, it was <mask>.'}
realistic_indicative_sentence = {'DBPedia': 'Category: <mask>. ', 'DBPedia_smaller': 'Category: <mask>. ', 'ag_news': 'Category: <mask>. ', 'ag_news_smaller': 'Category: <mask>. ', '20newsgroup': 'Category: <mask>. ', 'TREC-6': ' <mask>.', 'TREC-50':' <mask>.', 'IMDB': ' All in all, it was <mask>.', 'IMDB_smaller': ' All in all, it was <mask>.'}

optimal_indicative_sentence_position = {'DBPedia': 'first', 'DBPedia_smaller': 'first', 'ag_news': 'first', 'ag_news_smaller': 'first', '20newsgroup': 'first', 'TREC-6': 'last', 'TREC-50':'last', 'IMDB': 'last', 'IMDB_smaller': 'last'}
realistic_indicative_sentence_position = {'DBPedia': 'first', 'DBPedia_smaller': 'first', 'ag_news': 'first', 'ag_news_smaller': 'first', '20newsgroup': 'first', 'TREC-6': 'last', 'TREC-50':'last', 'IMDB': 'last', 'IMDB_smaller': 'last'}

optimal_entropy_weight = {'DBPedia': 7.0, 'DBPedia_smaller': 7.0, 'ag_news': 1.4, 'ag_news_smaller': 1.4, '20newsgroup': 4.0, 'TREC-6': 2.4, 'TREC-50': 1.6, 'IMDB': 1.4, 'IMDB_smaller': 1.4}
realistic_entropy_weight = {'DBPedia': 3.0, 'DBPedia_smaller': 3.0, 'ag_news': 3.0, 'ag_news_smaller': 3.0, '20newsgroup': 3.0, 'TREC-6': 3.0, 'TREC-50': 3.0, 'IMDB': 3.0, 'IMDB_smaller': 3.0}

Experiments = []

includeSBert = False

for experiment in Experiments_proto:
	experiment_IS_optimal = copy.deepcopy(experiment)
	experiment_IS_realistic = copy.deepcopy(experiment)
	experiment_SBert = copy.deepcopy(experiment)

	experiment_IS_optimal["--embedding_model"] = 'IndicativeSentence'
	experiment_IS_optimal['--clustering_method'] = 'EntropyLoss'
	experiment_IS_optimal['--indicative_sentence'] = optimal_indicative_sentence[experiment['--path']].replace('<', '^').replace('>','?').replace(' ', '_').replace('!', '5')
	experiment_IS_optimal['--indicative_sentence_position'] = optimal_indicative_sentence_position[experiment['--path']]
	experiment_IS_optimal['--entropy_weight'] =  optimal_entropy_weight[experiment['--path']]

	experiment_IS_realistic["--embedding_model"] = 'IndicativeSentence'
	experiment_IS_realistic['--clustering_method'] = 'EntropyLoss'
	experiment_IS_realistic['--indicative_sentence'] = realistic_indicative_sentence[experiment['--path']].replace('<','^').replace('>', '?').replace(' ', '_').replace('!', '5')
	experiment_IS_realistic['--indicative_sentence_position'] = realistic_indicative_sentence_position[experiment['--path']]
	experiment_IS_realistic['--entropy_weight'] = realistic_entropy_weight[experiment['--path']]

	experiment_SBert['--embedding_model'] = 'SBert'
	experiment_SBert['--clustering_method'] = 'SCANLoss'

	if includeSBert:
		Experiments.append(experiment_SBert)

	Experiments.append(experiment_IS_realistic)
	Experiments.append(experiment_IS_optimal)


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
	elif experiment['--model_method'] == 'DocSCAN_finetuning_multi' and experiment['--augmentation_method'] == 'Deletion':
		outfile = f'DeletionRatioExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	elif experiment['--model_method'] == 'DocSCAN_finetuning_multi' and experiment['--augmentation_method'] != 'Deletion':
		outfile = f'RealSelflabelingExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_entropy_weight_{experiment["--entropy_weight"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}_t5_model_{experiment["--t5_model"]}.txt'
	#else:
	#	outfile = f'EntropyWeightExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_entropy_weight_{experiment["--entropy_weight"]}.txt'

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











