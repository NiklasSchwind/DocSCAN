from typing import List
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelWithLMHead, BartTokenizer, BartForConditionalGeneration, RobertaTokenizer, RobertaModel
import random
import torch
from sentence_transformers import SentenceTransformer
from utils.EncodeDropout import encode_with_dropout
from tqdm import tqdm




class DataAugmentation:
    def __init__(self, device: str, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size

    def sentencelevel_backtranslation(self, data: List[str]):

        data = [text.split('.') for text in data]
        data = [[sentence.split('\n') for sentence in text] for text in data]

        data_in = [[[i,sentence] for i, sentence in enumerate(text) if len(sentence.split(' ')) >= 6 or len(sentence.split(' ')) == self.max_length(text)] for text in data]

        data_in = [random.choice(text) for text in data_in]

        indexes = [text[0] for text in data_in]

        data_in =  [text[1] for text in data_in]

        data_out = self.backtranslation(data_in)

        for i, index in enumerate(indexes):
            data[i][index] = data_out[i].replace('.','')

        return ['.'.join(text) for text in data]

    def sentencelevel_paraphrasing(self, data: List[str]):

        data = [text.replace('\n', '.').split('.') for text in data]

        data_in = [[[i,sentence] for i, sentence in enumerate(text) if len(sentence.split(' ')) >= 6 or len(sentence.split(' ')) == self.max_length(text)] for text in data]

        data_in = [random.choice(text) for text in data_in]

        indexes = [text[0] for text in data_in]

        data_in =  [text[1] for text in data_in]

        data_out = self.paraphrase_texts(list(data_in), 32, max([len(sentence)+1 for sentence in list(data_in)]))

        for i, index in enumerate(indexes):
            data[i][index] = ' '+ data_out[i].replace('.','')

        return ['.'.join(text) for text in data]

    def max_length(self, sentence_list):

        return max([len(sentence.split(' ')) for sentence in sentence_list])

    def backtranslation(self, data: List[str], language_order: str = None, original_language : str = 'en'):

        if language_order is None:
            language_order = ['fr', 'en']
        language_before = original_language
        for language in language_order:
            augmented_data = []
            tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}')
            model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            for i in tqdm(range(0,len(data),self.batch_size)):
                batch = data[i:min(i+(self.batch_size),len(data))]
                augmented_data += self._translate_texts(tokenizer,model,language, batch)
            data = augmented_data
            language_before = language

        return data

    def _divide_chunks(self, list, number):
        # looping till length l
        for i in range(0, len(list), number):
            if i+number <=len(list):
                yield list[i:i + number]
            else:
                yield list[i:len(list)]

    def _format_batch_texts(self, language:str, batch_texts:List[str]):

        formated_bach = [">>{}<< {}".format(language, text) for text in batch_texts]

        return formated_bach


    def _translate_texts(self,tokenizer,model,language: str,texts: List[str]):

        formated_batch_texts = self._format_batch_texts(language, texts)

        tokenized_texts = tokenizer.prepare_seq2seq_batch(formated_batch_texts, return_tensors="pt",  padding=True).to(self.device)


        tokenized_texts['input_ids'] = tokenized_texts['input_ids'][:,:512]
        tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'][:, :512]

        # Generate translation using model
        translated = model.generate(**tokenized_texts, do_sample = True)

        # Convert the generated tokens indices back into text
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return translated_texts

    def random_deletion(self, texts : List[str], ratio : float):

        return [' '.join([word for word in text.split(' ') if random.random() >= ratio]) for text in texts]

    def SBert_embed_with_dropout(self, texts: List[str],sbert_model, max_seq_length):
        texts = list(texts)
        with torch.no_grad():
            embedder = SentenceTransformer(sbert_model)
            embedder.max_seq_length = max_seq_length
            embedder.train()
            corpus_embeddings = encode_with_dropout(embedder, texts, batch_size=32, show_progress_bar=True,
                                                    eval=False, device=self.device)
            embedder.eval()

        return corpus_embeddings

    def random_cropping(self, texts : List[str], ratio: float):

        split_texts = [text.split(' ') for text in texts]
        lens = [len(split_text) for split_text in split_texts]

        crop_out = [sum([1 if random.random() <= ratio/2 else 0 for j in range(lens[i])]) for i in range(len(split_texts))]
        crop_out_high = [sum([0 if random.random() <= ratio/2 else 1 for j in range(lens[i])]) for i in range(len(split_texts))]
        bounds_crop_out = [(min(i),max(i)) for i in zip(crop_out, crop_out_high)]

        return [' '.join(split_text[bounds_crop_out[i][0]:bounds_crop_out[i][1]]) for i, split_text in enumerate(split_texts)]

    def summarize_batch(self, texts : List[str] , batch_size=16, max_length=150):


        print("Starting Summarization")

        tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
        model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size  # Calculate the number of batches

        preds = []  # List to store the generated summaries

        for i in tqdm(range(num_batches)):

            start_index = i * batch_size
            end_index = (i + 1) * batch_size if (i + 1) * batch_size <= num_texts else num_texts

            batch_texts = texts[start_index:end_index]  # Get the batch of texts
            encoding = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", add_special_tokens=True,
                                                   padding=True,
                                                   truncation=True)

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=1,
                                           max_length=max_length, repetition_penalty=2.5, length_penalty=1.0,
                                           early_stopping=True )

            batch_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                           generated_ids]
            preds.extend(batch_preds)  # Append the batch predictions to the overall predictions list


        return preds


    def backtranslate_batch_t5(self, texts, batch_size = 64, languages = ['English', 'French', 'English'], t5_model = 'large'):

        tokenizer = AutoTokenizer.from_pretrained(f't5-{t5_model}')
        tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors="pt", add_special_tokens=True,
                                    padding=True,
                                    truncation=True)
        min_length = int(max(3, min([len(text) for text in tokenized_texts])))
        max_length = int(tokenized_texts["input_ids"].size(dim = 1) + 7)
        '''
        min_length = int(max(3,min([len(text.split(' ')) for text in texts])-7))
        max_length = int(max([len(text.split(' ')) for text in texts])+7)
        '''
        for i in range(len(languages)-1):

            prefix = f"translate {languages[i]} to {languages[i+1]}: "
            print(prefix)
            texts = self._t5_generate_output(texts, prefix, batch_size, min_length, max_length, t5_model)
            print(texts)

        return texts

    def summarize_batch_t5(self, texts, batch_size=128,  t5_model = 'large'):

        tokenizer = AutoTokenizer.from_pretrained(f't5-{t5_model}')
        tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors="pt", add_special_tokens=True,
                                                      padding=True,
                                                      truncation=True)
        min_length = int(min([len(text.split(' ')) for text in texts])/3)
        max_length = int(tokenized_texts["input_ids"].size(dim=1)/4 + 3)

        #min_length = int(min([len(text.split(' ')) for text in texts])/3)
        #max_length = int(max([len(text.split(' ')) for text in texts])/4+3)

        prefix = "summarize: "
        texts = self._t5_generate_output(texts, prefix, batch_size, min_length, max_length, t5_model)

        return texts

    def _t5_generate_output(self,texts, prefix, batch_size,min_length, max_length,  t5_model = 'large'):

        texts = [prefix + text for text in texts]

        tokenizer = AutoTokenizer.from_pretrained(f't5-{t5_model}')
        model = AutoModelWithLMHead.from_pretrained(f't5-{t5_model}', return_dict=True).to(self.device)

        num_texts = len(texts)
        num_batches = (num_texts + batch_size - 1) // batch_size  # Calculate the number of batches

        preds = []  # List to store the generated summaries

        for i in tqdm(range(num_batches)):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            batch_texts = texts[start_index:end_index]  # Get the batch of texts
            encoding = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", add_special_tokens=True,
                                                   padding=True,
                                                   truncation=True)

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,min_length=min_length, max_length=max_length, do_sample=True)#, num_beams=2,

                                           # repetition_penalty=2.5, length_penalty=1.0,
                                           #early_stopping=True)

            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            preds.extend(batch_preds)  # Append the batch predictions to the overall predictions list

        return preds

    def paraphrase_texts(self, texts, batch_size, max_length, model_name: str = 'eugenesiow/bart-paraphrase'):
        # Load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Tokenize input texts
        tokenized_texts = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized_texts["input_ids"].to(self.device)
        attention_mask = tokenized_texts["attention_mask"].to(self.device)

        num_texts = len(texts)
        paraphrases = []

        for i in tqdm(range(0, num_texts, batch_size)):
            input_batch = input_ids[i:min(i + batch_size,num_texts)]
            mask_batch = attention_mask[i:min(i + batch_size,num_texts)]

            # Generate paraphrases
            with torch.no_grad():
                outputs = model.generate(input_batch, attention_mask=mask_batch, max_length=max_length, do_sample=True,
                                         num_return_sequences=1)

            # Decode paraphrases
            batch_paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            paraphrases.extend(batch_paraphrases)

        return paraphrases

    def random_sentence(self,texts, alldata):
        return random.choices(alldata,k=len(texts))

    def embed_IndicativeSentences_with_dropout(self,texts, indicative_sentence, indicative_sentence_position, dropout:float = 0.2):
        embedding_text = []
        model_name = 'roberta-base'
        for text in texts:
            if indicative_sentence_position == 'first':
                embedding_text.append(indicative_sentence + text)
            elif indicative_sentence_position == 'last':
                embedding_text.append(text + indicative_sentence)
        # print(embedding_text)
        # Load the RoBERTa model and tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to(self.device)
        for layer in model.encoder.layer:
            layer.output.dropout.p = dropout
        num_sentences = len(texts)
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
                if indicative_sentence_position == 'first':
                    input_ids.append(encoded_inputs['input_ids'][0, :512])
                    attention_mask.append(encoded_inputs['attention_mask'][0, :512])
                elif indicative_sentence_position == 'last':
                    input_ids.append(encoded_inputs['input_ids'][0, -512:])
                    attention_mask.append(encoded_inputs['attention_mask'][0, -512:])
            input_ids = torch.cat(input_ids, dim=0).reshape((end - start, 512))
            attention_mask = torch.cat(attention_mask, dim=0).reshape((end - start, 512))
            # Feed the input tensors to the RoBERTa model
            with torch.no_grad():
                batch_output = model(input_ids, attention_mask=attention_mask)  # , attention_mask=attention_mask)
            # Retrieve the encodings of the mask tokens from the output tensor
            mask_token_indices = torch.where(input_ids == tokenizer.mask_token_id)
            batch_mask_token_encodings = batch_output[0][mask_token_indices[0], mask_token_indices[1], :]
            # Add the mask token encodings for the current batch to the list
            mask_token_encodings.append(batch_mask_token_encodings)

        return torch.cat(mask_token_encodings, dim=0).to(self.device)


