from modules.search import RNNSearch, TransSearch


class Translator:
    def __init__(self, config, model, tokenizer):
        self.tokenizer = tokenizer

        if config.model_name != 'transformer':
            self.search = RNNSearch(config, model, tokenizer)
        else:
            self.search = TransSearch(config, model, tokenizer)

    def translate(self):
        self.tokenizer
        
        return