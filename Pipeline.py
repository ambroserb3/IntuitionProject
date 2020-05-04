### This is the main module of the inference pipeline. ###

### This is the main module of the inference pipeline. ###

from Extraction import extraction
from filterPersona import AssignTherapyPersona
from simpletransformers.conv_ai import ConvAIModel

def InferencePipeline(user_input):
    ''' 
    This pipeline uses a trained xlnet model for inferring the chat persona type.
    Then interacts with that chatbot persona, trained on counsel chats dataset
    Arguments
    -user_input: this is text that is raw_input from the user
    '''    
    #Load models
    emotionPersona = ConvAIModel("gpt2", "EmotionModel/checkpoint-3-epoch-1", use_cuda=False)
    experentialPersona = ConvAIModel("gpt2", "ExpModel/checkpoint-3-epoch-1", use_cuda=False)
    print("Finished loading models, now prepping")
 
    #Prep data
    extractedText = extraction(user_input)
    
    #Make chat therapy persona type prediction with other model
    PersonaType = AssignTherapyPersona(extractedText)

    #Feed into Therapy Persona Conversational Ai Model
    if PersonaType == "Emotion":
        emotionPersona.interact()
    else:
        experentialPersona.interact()


if __name__ == "__main__":
    user_input = raw_input("Hi, how can I help?") 
    InferencePipeline(user_input)