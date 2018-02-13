import os

from get_folder import getFolder

root_dir = getFolder()
files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]  # Filter the items and only keep files (strip out directories)

# Install: pip install spacy && python -m spacy download en
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en',  disable=[''])



for ff in files:
    f = open(root_dir + ff, 'r')
    paragraph = f.read()
    docs = nlp.pipe([paragraph.decode('utf-8')], n_threads=1)
    for doc in docs:
        assert doc.is_parsed
    #sentences = list(doc.sents)
#    [tok for tok in doc]
#    [chunk  for chunk in doc.noun_chunks]
#    [tok.lemma_ for tok in doc]
#    [entity.text for entity in doc.ents]
#    [tok.pos_ for tok in doc]
    f.close()
