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
	- topic: the topic/category of the row
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
    '''
	Returns the dataframe with a Therapy Persona row filled based on the corresponding topic
	Arguments:
	- df: Dataframe with the questions, answers, and topics
    '''
    for index, row in df.iterrows():
        df.at[index, 'TherapyPersona'] = switch(row['topic'])
    return df

#Split Data
def SplitandSave(df):
    '''
	Splits the dataset into Train, Test, and Validation subsets
	Arguments:
	- df: Dataframe with the questions, answers, and topics
    '''
    print(df.groupby('split').count())

    train_df = df[df.split == 'train']
    test_df = df[df.split == 'test']
    val_df = df[df.split == 'val']

    #Save sets
#     train_df.to_csv ('Data/Train.csv', index = False, header=True)
#     test_df.to_csv ('Data/Test.csv', index = False, header=True)
#     val_df.to_csv ('Data/Val.csv', index = False, header=True)
    
    return train_df, test_df, val_df

def setupEmotion(train_df, test_df, val_df):
    '''
	Loads data from the dataframe and dumps to a json for the emotion persona training set
	Arguments:
	- train_df: Train set Dataframe with the questions, answers, and topics for the emotion persona
	- test_df: Testing set Dataframe with the questions, answers, and topics for the emotion persona
	- val_df: Validation set Dataframe with the questions, answers, and topics for the emotion persona
    '''
    #####################Train Data##########################
    EmotionDF = train_df[train_df.TherapyPersona == 'Emotion']
    data_set = [{"personality": ["How did that make you feel?", "I want to get to the source of these feelings."], 
                "utterances": [{"candidates": [], "history": []}]}]
    i = 0
    for index, row in EmotionDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        numCandidates = len(data_set[0]["utterances"][i]["candidates"])
        if(i==100):
            break
        if(i != len(EmotionDF) - 1):
            data_set[0]["utterances"].append({"candidates": [], "history": []})
        if(numCandidates < 1):
            data_set[0]["utterances"][i]["candidates"].append(Answer)
            data_set[0]["utterances"][i]["history"].append(Question)
        i += 1


    json_dump = json.dumps(data_set)
    EmotionPersona = json.loads(json_dump)

    with open('Data/EmotionPersona.json', 'w') as outfile:
        json.dump(EmotionPersona, outfile)
    setupEmotionTest(test_df)

def setupEmotionTest(test_df):
    '''
	Loads data from the dataframe and dumps to a json for the test set of the emotion persona
	Arguments:
	- test_df: Testing set Dataframe with the questions, answers, and topics for the emotion persona
    '''
    #####################Test Data##########################
    test_EmotionDF = test_df[test_df.TherapyPersona == 'Emotion']
    tdata_set = [{"personality": ["How did that make you feel?", "I want to get to the source of these feelings."], 
                "utterances": [{"candidates": [], "history": []}]}]
    i = 0
    for index, row in test_EmotionDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        numCandidates = len(tdata_set[0]["utterances"][i]["candidates"])
        if(i==100):
            break
        if(i != len(test_EmotionDF) - 1):
            tdata_set[0]["utterances"].append({"candidates": [], "history": []})
        if(numCandidates < 1):
            tdata_set[0]["utterances"][i]["candidates"].append(Answer)
            tdata_set[0]["utterances"][i]["history"].append(Question)
        i += 1


    json_dump = json.dumps(tdata_set)
    test_EmotionPersona = json.loads(json_dump)

    with open('Data/EmotionPersonaTest.json', 'w') as outfile:
        json.dump(test_EmotionPersona, outfile)
        
def setupExperential(train_df, test_df, val_df):
    '''
	Loads data from the dataframe and dumps to a json for experential persona training set
	Arguments:
	- train_df: Train set Dataframe with the questions, answers, and topics for the emotion persona
	- test_df: Testing set Dataframe with the questions, answers, and topics for the emotion persona
	- val_df: Validation set Dataframe with the questions, answers, and topics for the emotion persona
    '''
    #####################Train Data##########################
    ExperentialDF = train_df[train_df.TherapyPersona == 'Experiential']
    data_set = [{"personality": ["Tell me more about the situation?", "Has this happened before in the past?"], 
                "utterances": [{"candidates": [], "history": []}]}]
    i = 0
    for index, row in ExperentialDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        numCandidates = len(data_set[0]["utterances"][i]["candidates"])
        if(i==100):
            break
        if(i != len(ExperentialDF) - 1):
            data_set[0]["utterances"].append({"candidates": [], "history": []})
        if(numCandidates < 1):
            data_set[0]["utterances"][i]["candidates"].append(Answer)
            data_set[0]["utterances"][i]["history"].append(Question)
        i += 1

            
    json_dump = json.dumps(data_set)
    ExperentialPersona = json.loads(json_dump)

    with open('Data/ExperentialPersonaTest.json', 'w') as outfile:
        json.dump(ExperentialPersona, outfile)

    setupExpTest(test_df)

def setupExpTest(test_df):
    '''
	Loads data from the dataframe and dumps to a json for the testing set for the experential persona
	Arguments:
	- test_df: Testing set Dataframe with the questions, answers, and topics for the experential persona
    '''
    #####################Test Data##########################
    test_ExperentialDF = test_df[test_df.TherapyPersona == 'Experiential']
    tdata_set = [{"personality": ["Tell me more about the situation?", "Has this happened before in the past?"], 
                "utterances": [{"candidates": [], "history": []}]}]
    i = 0
    for index, row in test_ExperentialDF.iterrows():
        Question = row['questionText']
        Answer =  row['answerText']
        numCandidates = len(tdata_set[0]["utterances"][i]["candidates"])
        if(i==100):
            break
        if(i != len(test_ExperentialDF) - 1):
            tdata_set[0]["utterances"].append({"candidates": [], "history": []})
        if(numCandidates < 1):
            tdata_set[0]["utterances"][i]["candidates"].append(Answer)
            tdata_set[0]["utterances"][i]["history"].append(Question)
        i += 1

            
    json_dump = json.dumps(tdata_set)
    test_ExperentialPersona = json.loads(json_dump)

    with open('Data/ExperentialPersona.json', 'w') as outfile:
        json.dump(test_ExperentialPersona, outfile)

def PrepPersonaData():
    '''
	loads data into dataframe, removes empty rows, sets the persona type for corresponding categories, splits and saves the 
    datasets, then sets up the dataset used for training and testing each persona
    '''
    df = loadData()
    df = removeEmpty(df)
    df = setPersona(df)
    train_df, test_df, val_df = SplitandSave(df)
    setupEmotion(train_df, test_df, val_df)
    setupExperential(train_df, test_df, val_df)

# pd.set_option("display.max_rows", None, "display.max_columns", None)
PrepPersonaData()
