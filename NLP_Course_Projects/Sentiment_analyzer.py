import numpy as np
import pandas as pd
import nltk

def build_dictionary(text):
    tokens = nltk.word_tokenize(text)
    print("Tokens:", tokens)
    set_tokens = set(tokens)
    print("set", set_tokens)
    # for token in tokens:
    #    pass

    pass


def main():
    build_dictionary("I love coffee and also coffee loved me")
    pass