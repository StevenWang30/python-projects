# bert-serving-start -model_dir /code/Models/uncased_L-12_H-768_A-12/ -num_worker=1 -cpu -pooling_strategy=NONE -show_tokens_to_client

from bert_serving.client import BertClient
import IPython


bc = BertClient()
out = bc.encode(['What a nice day', 'everyone gets an A'])

IPython.embed()