from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from fast_bert.prediction import BertClassificationPredictor
import torch

logger = logging.getLogger()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.device_count() > 1:
    multi_gpu = True
else:
    multi_gpu = False


metrics = [{'name': 'accuracy', 'function': accuracy}]

OUTPUT_DIR = "Model_Artifacts"
DATA_PATH = "Data"
LABEL_PATH = "labels" 

### Dataloading
databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='xlnet-base-cased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='questionText',
                          label_col='TherapyPersona',
                          batch_size_per_gpu=4,
                          max_seq_length=512,
                          multi_gpu=multi_gpu,
                          multi_label=False,
                          model_type='xlnet')

### Learner Creation
learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='pytorchdump',
						metrics=metrics,
						device=device,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=multi_gpu,
						is_fp16=False,
						multi_label=False,
						logging_steps=50)

### Train and Save
learner.fit(epochs=2,
			lr=6e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="adamw")

learner.save_model()


### Inference
texts = ['this is a test','I sure love tests']
predictions = learner.predict_batch(texts)
