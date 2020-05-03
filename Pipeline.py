### This is the main module of the inference pipeline. ###

import Extraction
import filterPersona

def InferencePipeline(user_input):
    ''' 
    This pipeline uses a trained xlnet model for inferring the chat persona type.
    Then interacts with that chatbot persona, trained on counsel chats dataset
    Arguments
    -user_input: this is text that is raw_input from the user
    '''
    print user_input
    
    #Load models
    emotionPersona = ConvAIModel("gpt", "outputs")
    experentialPersona = ConvAIModel("gpt", "outputs")
    print("Finished loading models, now prepping")
 
    #Prep data
    extractedText = extraction(user_input)
    
    #Make chat therapy persona type prediction
    PersonaType = filterPersona(extractedText)

    #Feed into Therapy Persona Conversational Ai Model
    if PersonaType = "Emotion":
        emotionPersona.interact()
    else:
        experentialPersona.interact()

	# Evaluate

if __name__ == "__main__":
    user_input = raw_input("Hi, how can I help?") 
    InferencePipeline(user_input)