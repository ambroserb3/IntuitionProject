# # The Models
# 1. A model for EntityRecognition to output a topic
# 2. A model for each persona/topic to elicit responses within that topic trained on dialogue history

# # THE PLAN
# 1. Entity Recognition to choose a topic
# 2. Based on topic, feed input into a second model
# 3. Output response

import Extraction
import filterPersona

def InferencePipeline():
    ''' The pipeline uses 
    '''
    #Load models
    model = ConvAIModel("gpt", "outputs")
	print("Model is: ", model_filename)
	print("Finished loading models, now prepping")
    #Prep data

    #Make chat therapy persona type prediction
    
    #Feed into Therapy Persona Conversational Ai Model

	# Evaluate
    
	return accuracy    

if __name__ == "__main__":