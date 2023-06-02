from typing import List
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelWithLMHead
import random
import torch
from sentence_transformers import SentenceTransformer
from utils.EncodeDropout import encode_with_dropout
from tqdm import tqdm




class DataAugmentation:
    def __init__(self, device: str, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size


    def backtranslation(self, data: List[str], language_order: List[str], original_language : str = 'en'):

        language_before = original_language
        for language in language_order:
            augmented_data = []
            tokenizer = MarianTokenizer.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}')
            model = MarianMTModel.from_pretrained(f'Helsinki-NLP/opus-mt-{language_before}-{language}').to(self.device)
            for batch in self._divide_chunks(data, self.batch_size):

                augmented_data += self._translate_texts(tokenizer,model,language_before, batch)
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

        tokenized_texts = tokenizer(formated_batch_texts, return_tensors="pt",  padding=True ).to(self.device)

        #
        tokenized_texts['input_ids'] = tokenized_texts['input_ids'][:,:512]
        tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'][:, :512]

        # Generate translation using model
        translated = model.generate(**tokenized_texts)

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

    def random_cropping(self, texts : List[str]):

        split_texts = [text.split(' ') for text in texts]
        lens = [len(split_text) for split_text in split_texts]

        return [split_text[random.randint(0,lens[i]):random.randint(0,lens[i])] for i, split_text in enumerate(split_texts)]

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
        '''
        tokenizer = AutoTokenizer.from_pretrained(f't5-{t5_model}')
        tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors="pt", add_special_tokens=True,
                                    padding=True,
                                    truncation=True)
        min_length = int(max(3, min([len(text) for text in tokenized_texts])))
        max_length = int(tokenized_texts["input_ids"].size(dim = 1) + 7)

        min_length = int(max(3,min([len(text.split(' ')) for text in texts])-7))
        max_length = int(max([len(text.split(' ')) for text in texts])+7)
        '''
        for i in range(len(languages)-1):

            prefix = f"translate {languages[i]} to {languages[i+1]}: "
            texts = self._t5_generate_output(texts, prefix, batch_size,  t5_model)

        return texts

    def summarize_batch_t5(self, texts, batch_size=128,  t5_model = 'large'):
        '''
        tokenizer = AutoTokenizer.from_pretrained(f't5-{t5_model}')
        tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors="pt", add_special_tokens=True,
                                                      padding=True,
                                                      truncation=True)
        min_length = int(min([len(text.split(' ')) for text in texts])/3)
        max_length = int(tokenized_texts["input_ids"].size(dim=1)/4 + 3)

        #min_length = int(min([len(text.split(' ')) for text in texts])/3)
        #max_length = int(max([len(text.split(' ')) for text in texts])/4+3)
        '''
        prefix = "summarize: "
        texts = self._t5_generate_output(texts, prefix, batch_size,  t5_model)

        return texts

    def _t5_generate_output(self,texts, prefix, batch_size,  t5_model = 'large'):

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

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,  do_sample=True)#, num_beams=2,
                                           #min_length=min_length,
                                           #max_length=max_length, repetition_penalty=2.5, length_penalty=1.0,
                                           #early_stopping=True)

            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)
            preds.extend(batch_preds)  # Append the batch predictions to the overall predictions list

        return preds






'''
    def summarize_batch_t5_small(self, texts, batch_size = 16, max_length=150):

        texts = ['summarize: '+text for text in texts]

        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        model = AutoModelWithLMHead.from_pretrained('t5-small', return_dict=True)

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

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=2, min_length = int(max_length/2),
                                           max_length=max_length, repetition_penalty=2.5, length_penalty=1.0,
                                           early_stopping=True)

            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds.extend(batch_preds)  # Append the batch predictions to the overall predictions list

        return preds
'''



