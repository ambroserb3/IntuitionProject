# IntuitionProject
An ML project to respond to concerns with appropriate response from an intuitive AI.

# How it works
1. We have three deep learning models. One of them is a general text classifier trained on XLNET which takes in the user's message as input and outputs a prediction on whether or not their message is "Emotional" or "Experential." These two classes include a certain set of topics from the original data set which can be set by business logic. The other two models are the chatbot therapist personas based on each of those classes (Emotional, and Experiential) built on GPT2 using the hugging face transformers implementation. 

2. The idea, is that the emotional persona will respond to topics like depression or anxiety and be more reflective about those feelings. Conversely, experiential is more focused on situations and experiences. These personas have a knowledge base which is based off the counsel chat dataset. They store a couple sentences describing their personality and a short dialogue history. When a message is received from a user, the agent will combine the content of this knowledge base with the message input to generate a reply.

3. Once our XLNET model outputs the prediction for which persona to assign, that input is feed into the respective persona model for that conversation where the interaction continues.

# Steps to train
1. Download the pretrained xlnet model in /xlnet
2. Convert the xlnet tensorflow model to a pytorch model
3. Run Prep.ipynb to set up training and test data for personas and the xlnet model
4. Run TrainXLNET.ipynb to train the XLNET model on our data and Train.ipynb to train my personas
5. Run pipeline.ipynb, which should allow you to chat with a persona through my inference pipeline

# Notes
It's optimal to run with NVIDIA V100 gpu because of pytorch.
Both google and aws won't let me exceed service quota so training on mutliple gpus, hasn't been an option. I apparently have to wait 24-48 hours after starting a project to request a service limit increase. I've always used GCP and AWS on company/research institution accounts so this has never been an issue. As a result I'm training very big models with a limited amount of computer power and memory, so I've had to minimize parameters to build the model.
A simple random forest would likely perform better for this given training set, but I wanted a more creative and fun solution.

# Ideas 
1. Maybe we can use therapist url to do webscraping to help build more personalized personalities
2. We might be able to use upvotes, and phrase extraction('seems like') to add more features to the training data, for more meaningful predictions. 
3. We could engineer more training data based on category searches.
