import nlpaug.augmenter.word as naw
from parrot import Parrot
import torch 
import warnings
warnings.filterwarnings("ignore")

class FinancialTextAugmenter:
    def __init__(self, use_paraphrase=True, use_synonym=True, use_gpu=True):
        self.use_paraphrase = use_paraphrase
        self.use_synonym = use_synonym

        if not torch.cuda.is_available():
            use_gpu = False

        if self.use_synonym:
            self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        
        if self.use_paraphrase:
            self.paraphraser = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=use_gpu)


    def augment(self, text):
        aug_text = text
        
        # Order of operations: Paraphrase -> Synonym
        if self.use_paraphrase:
            try:
                aug_text = self.paraphraser.augment(
                            input_phrase=aug_text,
                            diversity_ranker="levenshtein",
                            do_diverse=False, 
                            max_return_phrases = 1, 
                            max_length=256, 
                            adequacy_threshold = 0.99, 
                            fluency_threshold = 0.90)
            except Exception as e:
                print(f"Paraphrasing failed: {e}")

        if self.use_synonym:
            try:
                aug_text = self.synonym_aug.augment(aug_text)
            except Exception as e:
                print(f"Synonym replacement failed: {e}")

        return aug_text
