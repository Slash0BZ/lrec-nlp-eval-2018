import plac
import random
import spacy
import thinc.neural.gpu_ops
from pathlib import Path
from POSEvaluator import readData

TAG_MAP = {
	'AFX': {'pos': 'ADJ'},
	'CC': {'pos': 'CONJ'},
	'CD': {'pos': 'NUM'},
	'DT': {'pos': 'DET'},
	'EX': {'pos': 'PRON'},
	'FW': {'pos': 'X'},
	'HYPH': {'pos': 'PUNCT'},
	'IN': {'pos': 'ADP'},
	'JJ': {'pos': 'ADJ'},
	'JJR': {'pos': 'ADJ'},
	'JJS': {'pos': 'ADJ'},
	'LS': {'pos': 'X'},
	'MD': {'pos': 'VERB'},
	'NFP': {'pos': 'PUNCT'},
	'NN': {'pos': 'NOUN'},
	'NNP': {'pos': 'PROPN'},
	'NNPS': {'pos': 'PROPN'},
	'NNS': {'pos': 'NOUN'},
	'PDT': {'pos': 'DET'},
	'POS': {'pos': 'PART'},
	'PRP': {'pos': 'PRON'},
	'PRP$': {'pos': 'DET'},
	'RB': {'pos': 'ADV'},
	'RBR': {'pos': 'ADV'},
	'RBS': {'pos': 'ADV'},
	'RP': {'pos': 'ADP'},
	'SYM': {'pos': 'SYM'},
	'TO': {'pos': 'PART'},
	'UH': {'pos': 'INTJ'},
	'VB': {'pos': 'VERB'},
	'VBD': {'pos': 'VERB'},
	'VBG': {'pos': 'VERB'},
	'VBN': {'pos': 'VERB'},
	'VBP': {'pos': 'VERB'},
	'VBZ': {'pos': 'VERB'},
	'WDT': {'pos': 'DET'},
	'WP': {'pos': 'PRON'},
	'WP$': {'pos': 'DET'},
	'WRB': {'pos': 'ADV'},
	'``': {'pos': 'PUNCT'},
	'#': {'pos': 'SYM'},
	'$': {'pos': 'SYM'},
	'\'': {'pos': 'PUNCT'},
	',': {'pos': 'PUNCT'},
	'-LRB-': {'pos': 'PUNCT'},
	'-RRB-': {'pos': 'PUNCT'},
	'.': {'pos': 'PUNCT'},
	':': {'pos': 'PUNCT'}
}

def train():
	training_data = []
	data = readData("./00-21.br")
	for d in data:
		pTags = d[0]
		sent = [x.decode('utf-8') for x in d[1]]
		sentStr = ""
		for tok in sent:
			sentStr += tok + " " 	
		specs = {}
		specs["words"] = sent
		specs["tags"] = pTags
		training_data.append((sentStr, specs))
	nlp = spacy.blank('en')
	tagger = nlp.create_pipe('tagger')
	for tag, values in TAG_MAP.items():
		tagger.add_label(tag, values)
	nlp.add_pipe(tagger)
	
	optimizer = nlp.begin_training(device=1)
	for i in range(25):
		random.shuffle(training_data)
		losses = {}
		for text, annotations in training_data:
			nlp.update([text], [annotations], sgd=optimizer, losses=losses)
		print(losses)	

	test_text = "I like blue eggs"
	doc = nlp(test_text.decode('utf-8'))
	print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])
	
	output_dir = "SpacyModels"
	# save model to output directory
	if output_dir is not None:
		output_dir = Path(output_dir)
		if not output_dir.exists():
			output_dir.mkdir()
		nlp.to_disk(output_dir)
		print("Saved model to", output_dir)

		print("Loading from", output_dir)
		nlp2 = spacy.load(output_dir)
		doc = nlp2(test_text.decode('utf-8'))
		print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])

train()
	 
