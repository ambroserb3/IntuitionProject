### This module is used for some basic exploration of data
import pandas as pd 
import bz2file as bz2

#Load data into dataframe
data = bz2.open("Data/20200325_counsel_chat.csv.bz2")
df = pd.read_csv(data)

#count unique values
print(df.nunique())

#check class balance
print(df.groupby('topic').count())

### Some Results ####

## Unique items
# questionID        885
# questionTitle     890
# questionText      826
# questionLink      897
# topic              31    
# therapistInfo     315
# therapistURL      315
# answerText       2126
# upvotes             9
# views             549
# split               3
# questionTitle     890
# questionText      826
# topic              31
# answerText       2126
# split               3

### Number of Each Topic ###
#                           questionID
# topic                               
# addiction                          9
# anger-management                  40
# anxiety                          255
# behavioral-change                 59
# children-adolescents               6
# counseling-fundamentals          245
# depression                       341
# diagnosis                         25
# domestic-violence                 21
# eating-disorders                  13
# family-conflict                  120
# grief-and-loss                    29
# human-sexuality                    7
# intimacy                         219
# legal-regulatory                  13
# lgbtq                             53
# marriage                          46
# military-issues                    3
# parenting                        150
# professional-ethics               39
# relationship-dissolution          82
# relationships                    181
# self-esteem                       91
# self-harm                         12
# sleep-improvement                 10
# social-relationships              22
# spirituality                      34
# stress                             7
# substance-abuse                   40
# trauma                            72
# workplace-relationships           27



###Quick Thoughts####
# I have a tiny dataset and severe class imbalance.
# I'm going to use transfer learning, and also create an abstraction of these topics
# into emotional and experential, so i have two chat therapy personas.