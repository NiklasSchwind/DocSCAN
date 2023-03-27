from typing import List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from sentence_transformers import SentenceTransformer,models, losses, datasets
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os



class Embedder:

    def __init__(self,
                 texts: List[str],
                 path: str,
                 embedding_method: Literal['SBert', 'TSDEA', 'IndicativeSentence'],
                 device: str,
                 mode: Literal['test', 'train'] = 'train',
                 indicative_sentence: str = 'Topic:',
                 indicative_sentence_position: Literal['first', 'last'] = 'first',
                 batch_size: int = 128,
                 embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 max_sequence_length: int = 128,

                 ):
        self.texts = texts
        self.embeddings = []
        self.max_sequence_length = max_sequence_length
        self.embedding_method = embedding_method
        self.path = path
        self.mode = mode
        self.device = device
        self.embedding_methods = {'SBert': self._embed_SBert, 'TSDEA': self._embed_TSDEA, 'IndicativeSentence': self._embed_IndicativeWordPrediction}
        if self.embedding_method == 'IndicativeSentence':
            self.indicative_sentence = indicative_sentence
            self.indicative_sentence_position = indicative_sentence_position
            self.batch_size = batch_size
        elif self.embedding_method == 'SBert':
            self.embedding_model_name = embedding_model_name
            self.max_sequence_length = max_sequence_length



    def safe_embeddings(self):

        np.save(os.path.join(self.path, f"{self.mode}-{self.embedding_method}-embeddings.npy"), self.embeddings)

    def load_embeddings(self):

        return np.load(os.path.join(self.path, f"{self.mode}-{self.embedding_method}-embeddings.npy"))

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

        return corpus_embeddings

    def _embed_IndicativeWordPrediction(self):

        embedding_text = []
        model_name = 'roberta-base'
        for text in self.texts:
            if self.indicative_sentence_position == 'first':
                embedding_text.append(self.indicative_sentence + ' <mask>.' + text)
            elif self.indicative_sentence_position == 'last':
                embedding_text.append(text + self.indicative_sentence + ' <mask>.')

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
                input_ids.append(encoded_inputs['input_ids'][:512])
                attention_mask.append(encoded_inputs['attention_mask'][:512])

            input_ids = torch.cat(input_ids, dim=0 )
            attention_mask = torch.cat(attention_mask,dim=0)
            print(input_ids)
            '''
            print(encoded_inputs)
            # Split the input tensors into batches
            input_ids = encoded_inputs['input_ids']
            print(input_ids.shape)
            attention_mask = encoded_inputs['attention_mask']
            '''
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

        return corpus_embeddings









