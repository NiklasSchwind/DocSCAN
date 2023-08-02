# Learning to Classify Texts without Labels. A NLP Adaption of SCAN

This is the code base for the experiments conducted in my master thesis "Learning to Classify Texts without Labels. A NLP Adaption of SCAN". It is based on the code for the paper [DocSCAN: Unsupervised Text Classification via Learning from Neighbors](https://aclanthology.org/2022.konvens-1.4/), a previous approach to translating the [SCAN method](https://link.springer.com/chapter/10.1007/978-3-030-58607-2_16) to NLP.  There can be code fragments from the original DocSCAN code still in this code. 

## Installation

Assuming [Anaconda](https://docs.anaconda.com/anaconda/install/) and linux, the environment can be installed with the following command:
```shell
conda create -n scan python=3.6
conda activate scan

pip install -U sentence-transformers
conda install faiss-cpu -c pytorch
pip install -r requirements.txt
```

## Replicate Experiments

Run with 

```shell
PYTHONPATH=src python src/NLPScan.py --path dataset_path --embedding_model embedding_model --max_seq_length max_seq_length --topk num_nearest_neighbors --batch_size batch_size --dropout dropout --num_epochs num_epochs --device device --outfile outfile --clustering_method clustering_method --model_method model_method --threshold threshold --new_embeddings new_embeddings --augmentation_method augmentation_method --entropy_weight entropy_weight --ratio_for_deletion ratio_for_deletion --repetitions repetitions --indicative_sentence indicative_sentence --t5_model t5_model --max_prototypes max_prototypes
```

path is the path where the dataset is saved, it needs to contain 2 files, train.jsonl and test.jsonl where each line is a json dictionary containing the keys "text" and "label". 

embedding_model is the name of the embedding model that should be used, one can choose TSDEA, IndicativeSentence, SimCSESupervised and SimCSEUnsupervised. However, SimCSEUnsupervised and TSDEA first train the embedding model on the dataset. The default is SBert.

max_seq_length is the maximum sequence length that is considered by SBert, the default value is 128.

topk is the number of neighbors retrieved to build SCAN training set, the default value is 5.

batch_size is the batch size for training, the default is 64.

dropout is the dropout value for training, default is 0.1.

num_epoch is the number of epochs for training, default is 5.

device is the device used to train NLP-SCAN. Default is cpu but cuda and cuda:N is possible.

outfile is the path and name of a file where the output prints can be saved or NO. If NO is selected, the output will be shown directly on the shell. NO is also the default value. 

clustering_method is the clustering loss function that should be used. The values can be EntropyLoss and SCANLoss. SCANLoss is the default value. 

model_method is the training procedure that is used. Values can be DocSCAN, PrototypeBert, DocBERT, DocSCAN_finetuning_multi (for the Fine-Tuning through Self-Labeling step), NLPScan_fast (Includes the Fine-Tuning through Self-Labeling without augmentation method, that reuses the embeddings), k-means_mini_batch, SVM

threshold is the threshold for a confidence value so that a datapoint is considered a prototype. Default value is 0.95 

new_embeddings is a bool value which descides if embeddings should be reused. The default False. 

augmentation_method descides the augmentation method used in the Fine Tuning through Self Labeling step. It only works in combination with the model_method DocSCAN_finetuning_multi and one can choose from Dropout, Deletion, Cropping, Summarization, Backtranslation and Random. The default value is Random. 

entropy_weight is a float that decides the entropy weight. The default is 2. 

ratio_for_deletion is the deletion ratio for the augmentation methods cropping and deletion. The default is 0.02. 

repetitions is the number of repetitions of an experiment. The default is 3. 

indicative_sentence is the indicative sentence used in the indicative sentence embedding method. The default is "nothing" and to make it work one has to switch the characters '<', '>', ' ' and '!' with the characters '^', '?', '_' and '5'

indicative_sentence_position is the position of the indicative sentence in the text. Can be first and last, and the default value is first. 

t5_model sets the t5-model used in the summarization augmentation method. Can be small, base and large. The default is large. 

max_prototypes sets the maximum number of prototypes 
