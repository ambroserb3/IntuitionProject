### This module takes the input sentence and uses a language model to assign it to one of our chat therapy personas.
from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = 'Model_Artifact/model_out'
LABEL_PATH = "labels"

def AssignTherapyPersona(text):
    '''
	Returns the model prediction of whether or not the specified text is corresponds to the Emotion class or Experential Class
	Arguments:
	- text: Text input to be make prediction on.
    '''
    predictor = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=LABEL_PATH, # location for labels.csv file
                    multi_label=False,
                    model_type='xlnet',
                    do_lower_case=False)

    # Single prediction
    prediction = predictor.predict(text)
    return prediction

### Unit Test ####
print(AssignTherapyPersona("just get me result for this text"))