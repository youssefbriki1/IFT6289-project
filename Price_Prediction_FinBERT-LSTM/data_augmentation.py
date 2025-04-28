import nlpaug.augmenter.word as naw
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch 
import warnings
import spacy
import re
import random
warnings.filterwarnings("ignore")

class FinancialTextAugmenter:

    class EntityMaskAugmenter:
        def __init__(self, mask_with="[ORG]", mask_types=None):
            """
            mask_with: string to replace entities with
            mask_types: list of entity labels to mask (e.g., ['ORG', 'PERSON', 'GPE'])
            """
            self.mask_with = mask_with
            self.mask_types = mask_types if mask_types else ["ORG", "PERSON", "GPE", "DATE", "MONEY"]
            self.nlp = spacy.load("en_core_web_sm")
        
        def augment(self, text):
            doc = self.nlp(text)
            augmented_tokens = []
            for token in doc:
                if token.ent_type_ in self.mask_types:
                    augmented_tokens.append(self.mask_with)
                else:
                    augmented_tokens.append(token.text)
            return " ".join(augmented_tokens)
        
    class SentenceReorderAugmenter:
        def __init__(self, shuffle_ratio=0.8):
            """
            shuffle_ratio: probability of shuffling sentences
            """
            self.shuffle_ratio = shuffle_ratio

        def split_into_sentences(self, text):
            # Simple regex split; could use spaCy if needed
            sentences = re.split(r'(?<=[.!?;]) +', text)
            return [s for s in sentences if s]

        def augment(self, text):
            sentences = self.split_into_sentences(text)
            if len(sentences) <= 1:
                return text  # No reordering possible
            num_to_shuffle = int(len(sentences) * self.shuffle_ratio)
            indices_to_shuffle = random.sample(range(len(sentences)), num_to_shuffle)
            sentences_to_shuffle = [sentences[i] for i in indices_to_shuffle]
            random.shuffle(sentences_to_shuffle)
            for idx, sent_idx in enumerate(indices_to_shuffle):
                sentences[sent_idx] = sentences_to_shuffle[idx]
            return " ".join(sentences)
    
    #########

    def __init__(self, use_entity_mask=False, use_reorder=False, use_paraphrase=False, use_synonym=False, use_gpu=True):
        self.use_paraphrase = use_paraphrase
        self.use_synonym = use_synonym
        self.use_entity_mask = use_entity_mask
        self.use_reorder = use_reorder

        if not torch.cuda.is_available():
            use_gpu = False

        if self.use_synonym:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        if self.use_paraphrase:
            # Load the model and tokenizer
            model_name = "Vamsi/T5_Paraphrase_Paws"  
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.paraphraser = model.to(self.device)
        
        if self.use_entity_mask:
            self.entity_mask_aug = self.EntityMaskAugmenter()
        
        if self.use_reorder:
            self.reorder_aug = self.SentenceReorderAugmenter()

    def __paraphrase_helper(sentence, 
                     model, 
                     tokenizer, 
                     device, 
                     max_length=128, 
                     num_return_sequences=5, 
                     num_beams=10):
        prompt = f"paraphrase: {sentence} </s>"

        encoding = tokenizer.encode_plus(prompt,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=max_length,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
        )

        paraphrased_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return paraphrased_sentences

    def augment(self, text):
        aug_text = text
        
        # Order of operations: Paraphrase -> Synonym
        if self.use_paraphrase:
            try:
                """
                aug_text = self.paraphraser.augment(
                            input_phrase=aug_text,
                            do_diverse=True,
                            max_return_phrases = 10, 
                            beam_width=10,             
                            max_length=80,
                            early_stopping=False)[0]
                """
                aug_text = self.__paraphrase_helper(
                    sentence=aug_text,
                    model=self.paraphraser,
                    tokenizer=self.tokenizer,
                    device=self.device
                )[0]
                
            except Exception as e:
                print(f"Paraphrasing failed: {e}")

        if self.use_synonym:
            try:
                aug_text = self.synonym_aug.augment(aug_text)
                aug_text = aug_text[0]
            except Exception as e:
                print(f"Synonym replacement failed: {e}")
        
        if self.use_entity_mask:
            try:
                aug_text = self.entity_mask_aug.augment(aug_text)
            except Exception as e:
                print(f"Entity Masking failed: {e}")
        
        if self.use_reorder:
            try:
                aug_text = self.reorder_aug.augment(aug_text)
            except Exception as e:
                print(f"Sentence reordering failed: {e}")

        return aug_text
