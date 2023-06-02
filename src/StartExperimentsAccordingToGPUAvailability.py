import multiprocessing as mp
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlDeviceGetCount
import time
from datetime import datetime
import os
import copy



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
'''
Experiments = [
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Dropout'},
#{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
#			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Dropout'},
#{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
#			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Deletion'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Deletion'},
#{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
#			   {'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},

]

'''
'''
Experiments = [
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Dropout'},
#{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
#			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Dropout'},
#{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
#			   {'--embedding_model': 'SBert', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Cropping'},
			   {'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Dropout'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Dropout'},
#{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
#			   {'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
#				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion'},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'SBert', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5, '--data_augmentation': 'Backtranslation_de_en'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 10,'--data_augmentation': 'Backtranslation_de_en'},

]
'''
'''
Experiments = [{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
{'--embedding_model': 'SBert', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05},
			   {'--embedding_model': 'SBert', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--data_augmentation': 'Deletion', "--ratio_for_deletion": 0.05}

]

'''
'''
Experiments = [{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.01, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.02, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.03, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.05, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.07, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.1, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.15, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.2, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.25, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.3, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.35, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.4, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.45, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,
				'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.5, '--repetitions': 10},
{'--embedding_model': 'SBert', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--data_augmentation': 'Deletion','--ratio_for_deletion': 0.55, '--repetitions': 10}]
'''
'''
Experiments = [{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' It was <mask>!', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' I <mask> it!', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' It was <mask>!', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' I <mask> it!', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' Answer: <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' I think <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' Answer: <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' I think <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' Answer: <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' I think <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' Answer: <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' I think <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' },
			   '''
'''
FutureExperiment = [
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Topic: <mask>. ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Read <mask> for more on this topic.', '--indicative_sentence_position': 'last' }]
'''
'''
Experiments = [
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first' },

]
'''

'''
Experiments = [
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' Related: <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 4.0 },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' Related: <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.0 },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' All in all, it was <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 4.0 },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' All in all, it was <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.0 },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'last', '--entropy_weight': 4.0 },
{'--embedding_model': 'IndicativeSentence', '--path': 'DBPedia', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': 'Category: <mask>. ', '--indicative_sentence_position': 'last', '--entropy_weight': 3.0 },
]

'''

'''
Experiments = [
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.5 },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 4.0 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 4.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 5.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 6.0},
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 6.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 7.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 8.0 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 8.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 9.0 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 9.5 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 11 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 12 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 13 },
				{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
				'--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 15 },
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 3.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 4.0},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 4.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 5.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 6.0},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 6.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 7.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 8.0},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 8.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 9.0},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 9.5},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 11},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 12},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 13},
	{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': 'Enjoy the following article about <mask>: ', '--indicative_sentence_position': 'first',
	 '--entropy_weight': 15},
]
'''
'''
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 0.6},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 1.0},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 1.2},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 1.4},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 1.6},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 1.8},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 2.0},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 2.2},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 2.4},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 2.6},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 2.8},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.0},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.2},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 3.4},
	{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 4.0},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'False',
	 '--indicative_sentence': ' <mask>.', '--indicative_sentence_position': 'last', '--entropy_weight': 5.0},
]
'''

'''
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' (<mask>)', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '[<mask>: ] ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '<mask> News: ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': '20newsgroup', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' Related: <mask>.', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '[Category: <mask>] ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' (<mask>)', '--indicative_sentence_position': 'last' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '[<mask>: ] ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '<mask> News: ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'ag_news', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': ' Related: <mask>.', '--indicative_sentence_position': 'last' },
	{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'SCANLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
	 '--indicative_sentence': 'Sentiment: <mask>.', '--indicative_sentence_position': 'first'},
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Sentiment: <mask>.', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
	 '--indicative_sentence': '[Question: <mask>] ', '--indicative_sentence_position': 'first'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-6', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '<mask> question: ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
	 '--indicative_sentence': '[Question: <mask>] ', '--indicative_sentence_position': 'first'},
{'--embedding_model': 'IndicativeSentence', '--path': 'TREC-50', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': '<mask> question: ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
				'--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
				'--indicative_sentence': 'Just <mask>! ', '--indicative_sentence_position': 'first' },
{'--embedding_model': 'IndicativeSentence', '--path': 'IMDB', '--clustering_method': 'EntropyLoss',
	 '--model_method': 'DocSCAN', '--num_epochs': 5, '--repetitions': 10, '--new_embeddings': 'True',
	 '--indicative_sentence': ' All in all, it was <mask>.', '--indicative_sentence_position': 'last'}
]

'''

Experiments_proto = [
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Dropout'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'base'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'small'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Deletion'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Deletion'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Summarization', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_fr', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.99, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'large'},
{'--embedding_model': 'SBert', '--path': 'ag_news_smaller', '--clustering_method': 'SCANLoss',
				'--model_method': 'DocSCAN_finetuning_multi', '--threshold': 0.95, '--num_epochs': 5,'--augmentation_method': 'Backtranslate_en_de_fr', '--t5_model': 'large'},



]

Experiments = []
for experiment in Experiments_proto:
	#experiment["--indicative_sentence"] = experiment["--indicative_sentence"].replace('<', '^').replace('>',
	#																									'?').replace(
	#	' ', '_').replace('!', '5')
	#if experiment['--t5_model'] != 'large':
	#	experiment['--repetitions'] = 10
	experiment1 = copy.deepcopy(experiment)
	experiment2 = copy.deepcopy(experiment)
	experiment3 = copy.deepcopy(experiment)
	experiment4 = copy.deepcopy(experiment)

	#experiment1['--path'] = 'DBPedia_smaller'
	#experiment2['--path'] = 'TREC-6'
	#experiment3['--path'] = 'TREC-50'
	#experiment3['--num_epochs'] = 10
	#experiment4['--path'] = '20newsgroup'

	Experiments.append(experiment)
	#Experiments.append(experiment1)
	#Experiments.append(experiment2)
	#Experiments.append(experiment3)
	#Experiments.append(experiment4)










def start_experiment(experiment, device):
	#if (experiment["--augmentation_method"] == 'Backtranslate_en_fr' or experiment["--augmentation_method"] == 'Summarization' or experiment["--augmentation_method"] == 'Backtranslate_en_de' or experiment["--augmentation_method"] == 'Backtranslate_en_de_fr') and experiment['--embedding_model'] == 'IndicativeSentence':
	#	outfile = f'IndicativeSentencesExperimentLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}_t5_model_{experiment["--t5_model"]}.txt'#_threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	if ((experiment["--augmentation_method"] == 'Backtranslate_en_fr' or experiment["--augmentation_method"] == 'Summarization' or experiment["--augmentation_method"] == 'Backtranslate_en_de' or experiment["--augmentation_method"] == 'Backtranslate_en_de_fr')): #and experiment['--embedding_model'] != 'IndicativeSentence':
		outfile = f'RealSelflabelingExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}_t5_model_{experiment["--t5_model"]}.txt'
	#elif (experiment["--augmentation_method"] != 'Backtranslate_en_fr' and experiment["--augmentation_method"] != 'Summarization' and experiment["--augmentation_method"] != 'Backtranslate_en_de' and experiment["--augmentation_method"] != 'Backtranslate_en_de_fr') and experiment['--embedding_model'] == 'IndicativeSentence':
	#	outfile = f'IndicativeSentencesExperimentLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_indicativesentence_{experiment["--indicative_sentence"]}.txt'  # _threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}_ratio_{experiment["--ratio_for_deletion"]}.txt'
	else:
		outfile = f'RealSelflabelingExperiments/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_augmentation_method_{experiment["--augmentation_method"]}.txt'

	#	outfile = f'DeletionRatioLogs/Dataset_{experiment["--path"]}_Embedding_{experiment["--embedding_model"]}_clustering_method_{experiment["--clustering_method"]}_model_method_{experiment["--model_method"]}_epochs_{experiment["--num_epochs"]}_threshold_{experiment["--threshold"]}_{experiment["--augmentation_method"]}.txt'

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











