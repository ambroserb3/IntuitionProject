# This module will prep the data set for training
import pandas as pd
import bz2file as bz2
import json

def loadData():
    '''
    Returns the dataframe after loading from raw data, while remove unnecessary columns
    '''
    # Load data into dataframe
    data = bz2.open("Data/20200325_counsel_chat.csv.bz2")
    df = pd.read_csv(data)
    # Remove unneccessary input data
    df = df.drop(columns=['questionID', 'upvotes', 'views', 'questionLink', 'therapistInfo', 'therapistURL'])
    # fun idea, use therapist url to do webscraping to build the persona.
    
    df.insert(0, "TherapyPersona", "Emotion")  ### This is an abstraction over topic to deal with classimbalance/limited data
    return df
    

def switch(topic):
    '''
	Returns a the Therapy Persona type corresponding to the topic
	Arguments:
	- topic: Dataframe with comments and their corresponding toxicities 
    '''
    switcher = {
        #emotion categories
        "anxiety": "Emotion",
        "anger-management": "Emotion",
        "depression": "Emotion",
        "stress": "Emotion",
        "spirituality": "Emotion",
        "human-sexuality": "Emotion",
        "self-esteem": "Emotion",
        "intimacy": "Emotion",
        "children-adolescents": "Emotion",
        "behavioral-change": "Emotion",
        "counseling-fundamentals": "Emotion",
        "relationships": "Emotion",
        "grief-and-loss": "Emotion",

        #experiential categories
        "legal-regulatory": "Experiential",
        "trauma": "Experiential",
        "workplace-relationships": "Experiential",
        "substance-abuse": "Experiential",
        "lgbtq": "Experiential",
        "addiction": "Experiential",
        "parenting": "Experiential",
        "social-relationships": "Experiential",
        "sleep-improvement": "Experiential",
        "relationship-dissolution": "Experiential",
        "military-issues": "Experiential",
        "diagnosis": "Experiential",
        "family-conflict": "Experiential",
        "eating-disorders": "Experiential",
        "marriage":"Experiential",
        "domestic-violence": "Experiential",
        "self-harm": "Experiential",
        "professional-ethics": "Experiential"
    }
    return switcher.get(topic, "Emotion")

def removeEmpty(df):
    '''
	Returns the data frame after removing empty rows from data
	Arguments:
	- df: Dataframe with the questions, answers, and topics
    '''
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(subset = ["questionText", "questionTitle", "answerText"], inplace=True)
    return df
    
def setPersona(df):
    for index, row in df.iterrows():
        df.at[index, 'TherapyPersona'] = switch(row['topic'])
    return df

#Split Data
def SplitandSave(df):
    print(df.groupby('split').count())

    train_df = df[df.split == 'train']
    test_df = df[df.split == 'test']
    val_df = df[df.split == 'val']

    #Save sets
    train_df.to_csv ('Data/Train.csv', index = False, header=True)
    test_df.to_csv ('Data/Test.csv', index = False, header=True)
    val_df.to_csv ('Data/Val.csv', index = False, header=True)
    
    print(train_df)

def PrepData():
    df = loadData()
    df = removeEmpty(df)
    df = setPersona(df)
    SplitandSave(df)
    return df

def setupEmotion(df):
    EmotionDF = df[df.TherapyPersona == 'Emotion']
    data_set = {"Personality": ["How did that make you feel?", "I want to get to the source of these feelings."], 
                "utterances": [{"candidates": [], "history": []}]}
    Prev = ''
    i = 0
    # print(EmotionDF['questionText'].nunique())
    for index, row in EmotionDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        if(Prev == Question):
            # print(len(data_set["utterances"]))
            data_set["utterances"][i]["candidates"].append(Answer)
            Prev = Question
        else:
            data_set["utterances"].append({"candidates": [], "history": []})
            data_set["utterances"][i]["history"].append(Question)
            i += 1
            data_set["utterances"][i]["candidates"].append(Answer)
            Prev = Question
            
    json_dump = json.dumps(data_set)
    EmotionPersona = json.loads(json_dump)

    with open('Data/EmotionPersona.json', 'w') as outfile:
        json.dump(EmotionPersona, outfile)

def setupExperential(df):
    ExperentialDF = df[df.TherapyPersona == 'Experiential']
    data_set = {"Personality": ["Tell me more about the situation?", "Has this happened before in the past?"], 
                "utterances": [{"candidates": [], "history": []}]}
    Prev = ''
    i = 0
    for index, row in ExperentialDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        if(Prev == Question):
            # print(len(data_set["utterances"]))
            data_set["utterances"][i]["candidates"].append(Answer)
            Prev = Question
        else:
            data_set["utterances"].append({"candidates": [], "history": []})
            data_set["utterances"][i]["history"].append(Question)
            i += 1
            data_set["utterances"][i]["candidates"].append(Answer)
            Prev = Question
            
    json_dump = json.dumps(data_set)
    ExperentialPersona = json.loads(json_dump)

    with open('Data/ExperentialPersona.json', 'w') as outfile:
        json.dump(ExperentialPersona, outfile)


def PrepPersonaData():
    df = loadData()
    df = removeEmpty(df)
    df = setPersona(df)
    setupEmotion(df)
    setupExperential(df)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
PrepPersonaData()


# candidates will be the answer answerText
# history will be the questionText + answerText
# 1. Filter duplicate questions. 
# 2. set candidates for each utterance will be the answer text from a questionText
# 3. The history will be that questionText

    
    


