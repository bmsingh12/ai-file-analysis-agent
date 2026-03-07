from langchain_classic.document_loaders import PyPDFLoader
import pandas as pd

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


def load_csv(path):
    df = pd.read_csv(path)
    return df.to_string()