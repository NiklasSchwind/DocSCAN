from typing import List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from sentence_transformers import SentenceTransformer,models, losses, datasets, InputExample
from transformers import RobertaTokenizer, RobertaModel,AutoTokenizer, AutoModel
import torch
import numpy as np
import os



class Embedder:

    def __init__(self,
                 texts: List[str],
                 path: str,
                 embedding_method: Literal['SBert', 'TSDEA', 'IndicativeSentence','SimCSEsupervised'],
                 device: str,
                 mode: Literal['test', 'train'] = 'train',
                 indicative_sentence: str = 'I <mask> it!', #I <mask> it! for sentiment
                 indicative_sentence_position: Literal['first', 'last'] = 'last',
                 batch_size: int = 128,
                 embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 max_sequence_length: int = 128,

                 ):
        self.texts = [str(text) for text in texts]
        self.embeddings = []
        self.max_sequence_length = max_sequence_length
        self.embedding_method = embedding_method
        self.path = path
        self.mode = mode
        self.device = device
        self.embedding_methods = {'SBert': self._embed_SBert, 'TSDEA': self._embed_TSDEA, 'IndicativeSentence': self._embed_IndicativeWordPrediction, 'SimCSEsupervised': self._embed_SimCSE_supervised}
        if self.embedding_method == 'IndicativeSentence':
            self.indicative_sentence = indicative_sentence
            self.indicative_sentence_position = indicative_sentence_position
            self.batch_size = batch_size
        elif self.embedding_method == 'SBert':
            self.embedding_model_name = embedding_model_name
            self.max_sequence_length = max_sequence_length



    def safe_embeddings(self):

        np.save(os.path.join(self.path, f"{self.mode}-{self.embedding_method}-embeddings.npy"), self.embeddings.cpu())

    def load_embeddings(self):

        return torch.from_numpy(np.load(os.path.join(self.path, f"{self.mode}-{self.embedding_method}-embeddings.npy")))

    def embed(self,
              createNewEmbeddings: bool = False,
              safeEmbeddings: bool = True
              ):

        if os.path.exists(os.path.join(self.path, f"{self.mode}-{self.embedding_method}-embeddings.npy")) and not createNewEmbeddings:
            embeddings = self.load_embeddings()
            self.embeddings = embeddings
        else:
            embeddings = self.embedding_methods[self.embedding_method]()
            self.embeddings = embeddings
            if safeEmbeddings:
                self.safe_embeddings()

        return embeddings

    def _embed_SBert(self):

        embedder = SentenceTransformer(self.embedding_model_name, device = self.device)
        embedder.max_seq_length = self.max_sequence_length
        corpus_embeddings = embedder.encode(self.texts, batch_size=32, show_progress_bar=True)

        return torch.from_numpy(corpus_embeddings)

    def _embed_IndicativeWordPrediction(self):

        embedding_text = []
        model_name = 'roberta-base'
        for text in self.texts:
            if self.indicative_sentence_position == 'first':
                embedding_text.append(self.indicative_sentence + text)
            elif self.indicative_sentence_position == 'last':
                embedding_text.append(text + self.indicative_sentence)

        # Load the RoBERTa model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to(self.device)

        num_sentences = len(self.texts)
        num_batches = (num_sentences + self.batch_size - 1) // self.batch_size

        # Initialize a list to store the mask token encodings for all batches
        mask_token_encodings = []

        # Process each batch of input sentences
        for i in range(num_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, num_sentences)
            # Extract the input tensors for the current batch
            input_ids = []
            attention_mask = []
            for text in embedding_text[start:end]:
                encoded_inputs = tokenizer.encode_plus(text, padding='max_length', return_tensors='pt').to(
                self.device)
                if self.indicative_sentence_position == 'first':
                    input_ids.append(encoded_inputs['input_ids'][0,:512])
                    attention_mask.append(encoded_inputs['attention_mask'][0,:512])
                elif self.indicative_sentence_position == 'last':
                    input_ids.append(encoded_inputs['input_ids'][0, -512:])
                    attention_mask.append(encoded_inputs['attention_mask'][0, -512:])
            input_ids = torch.cat(input_ids, dim=0 ).reshape((end-start, 512))
            attention_mask = torch.cat(attention_mask,dim=0).reshape((end-start, 512))
            # Feed the input tensors to the RoBERTa model
            with torch.no_grad():
                batch_output = model(input_ids, attention_mask = attention_mask)#, attention_mask=attention_mask)
            # Retrieve the encodings of the mask tokens from the output tensor
            mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)
            batch_mask_token_encodings = batch_output[0][mask_token_indices[0], mask_token_indices[1], :]
            # Add the mask token encodings for the current batch to the list
            mask_token_encodings.append(batch_mask_token_encodings)

        return torch.cat(mask_token_encodings,dim=0)

    def _embed_TSDEA(self):
        # Define your sentence transformer model using CLS pooling
        model_name = 'bert-base-uncased'
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
        TSDAEModel = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = self.device)

        # Transform dataset to right format
        train_dataset = datasets.DenoisingAutoEncoderDataset(self.texts)

        # DataLoader to batch data, use recommended batch size
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, device = self.device)

        # Define recommanded loss function
        train_loss = losses.DenoisingAutoEncoderLoss(TSDAEModel, decoder_name_or_path=model_name,
                                                     tie_encoder_decoder=True).to(self.device)

        # Call the fit method
        TSDAEModel.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            weight_decay=0,
            scheduler='constantlr',
            optimizer_params={'lr': 3e-5},
            show_progress_bar=True
        )

        #TSDAEModel.save('output/tsdae-model')

        corpus_embeddings = TSDAEModel.encode(self.texts)

        return torch.cat(corpus_embeddings)


    def _embed_SimCSE_unsupervised(self):
        # Define sentence transformer model using CLS pooling
        model_name = 'distilroberta-base'  # 'sentence-transformers/all-mpnet-base-v2'#'distilroberta-base'
        word_embedding_model = models.Transformer(model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        SimCSEmodel = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(self.device)

        # Create sentence pairs for training
        TrainData_paired = [InputExample(texts=[s, s]) for s in self.texts]

        # DataLoader to batch the data using recommended batchsize
        TrainData_batched = torch.utils.data.DataLoader(TrainData_paired, batch_size=128, shuffle=True).to(self.device)

        # Define recommended loss function
        train_loss = losses.MultipleNegativesRankingLoss(SimCSEmodel).to(self.device)

        # Call the fit method
        SimCSEmodel.fit(
            train_objectives=[(TrainData_batched, train_loss)],
            epochs=5,
            show_progress_bar=True
        )
        corpus_embeddings = SimCSEmodel.encode(self.texts)

        return torch.cat(corpus_embeddings)

    def _embed_SimCSE_supervised(self):
        # Define sentence transformer model using CLS pooling
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        texts = [i for i in self.texts if i is not None]
        tokenized_texts = tokenizer.encode_plus(texts, padding=True, return_tensors='pt').to(self.device)
        corpus_embeddings = model.encode(tokenized_texts)

        return corpus_embeddings









