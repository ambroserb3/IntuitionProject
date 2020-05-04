### This module takes the input sentence and uses a language model to assign it to one of our chat therapy personas.
from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = 'Model_Artifacts/model_out'
LABEL_PATH = "labels"

def AssignTherapyPersona(text):
    predictor = BertClassificationPredictor(
                    model_path=MODEL_PATH,
                    label_path=LABEL_PATH, # location for labels.csv file
                    multi_label=False,
                    model_type='xlnet',
                    do_lower_case=False)

    # Single prediction
    prediction = predictor.predict(text)
    ###Example output####
#     [('Emotion', 0.6595598459243774), ('Experiential', 0.34044015407562256)]
    return prediction[0][0]

### Unit Test ###
# test = AssignTherapyPersona("hey I'm feeling a bit sad and depressed")
# print(test)