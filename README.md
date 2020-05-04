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

# Challenge and Drawbacks
1. Limited training data, also not particularly clean data. Transfer Learning is a good way to deal, but with only 2000ish potential samples, it's still a challenge. I wish I'd spent some time engineering or scraping more data. Looking back, I may have been able to supplement this data well by scraping a mental health reddit. Of course abstracting away classes of topics to just Emotional and Experiential was one way to help deal with this. 
2. COMPUTE POWER. I'm on a new laptop which apparently couldn't handle retraining any of the massive pretrained NNs I was using. I'm so used to having access to all the compute power I need on GCP or AWS, I didn't realize service limits were a thing for individual users, and had to further minimize the parameters and training data just to train (very slowly) what I had. With a single p3.2xlarge instance on sagemaker I could have had each of these models train in about an hour, and being able to test and optimize them more. 
3. Inference speed. While I haven't been able to test my models yet, I know from experience, that they can take a few seconds too many to load before inference. This obviously depends a lot on the compute power of the endpoint, but I have multiple models loaded and running during inference, which wouldn't be an ideal setup for a chatbot application, where the end_user likely expects a fast response. There are some ways around this though.
4. Class imbalance. I didn't take the time to do extensive EDA for this, but I did a little bit, the classes are definitely imbalanced in this tiny training set. Not really sure whether or not I would oversample or undersample to resolve that, I'd have to look at the data and think about it more.

# ToDos
1. Stopwords, we're using deep learning models and this is a very context sensitive application, so removing all stop words for training and inference is an awful idea. "I'm feeling overworked and stressed because I didn't finish a project in time." and "I'm not feeling overworked and stressed because I did finish a project in time" might both be read as "feeling overworked stressed because finish project time." No bueno. Need a custom stopwords dictionary.
2. More Cleaning, lots of non utf-8 characters and homoglyphs which should probably be substituted.
3. Evaluation, I don't trust the internal metrics of ml frameworks by themselves, I prefer to use them in conjuction with scripts I can use to evaluate any given test set. 
4. Inference pipeline. I know that xlnet predictor probably produces a tuple from experience rather than a boolean. Once that model is finished I'll need to make sure I'm return the correct input from that tuple to feed into the persona model. 
