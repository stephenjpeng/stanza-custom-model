import argparse
import math
import numpy as np
import nltk
nltk.download('punkt')
import os
import pandas as pd
import random
import re
# import stanza

from stanza.utils.datasets.ner.utils import write_dataset
from transform_weight_date import number_to_words, date_to_formats

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_data_file', type=str, default='', help='Name of CSV file to be converted to BIO format.')
    parser.add_argument('--savedir', type=str, default= os.getcwd() + '/Processed_Data', help='Directory to save outputs to.')
    parser.add_argument('--save_prefix', type=str, default='output', help='Prefix of output.')

    args = parser.parse_args(args=args)
    args = vars(args)
    return args

# Use re to replace any instances of "####kg" with "#### kg" where #### is any continuous 
# sequence of numbers and unit is one of those listed below
def separate_weight_unit(row):
    return re.sub(r'([0-9]+)(kgs|kg|lbs|lb|pounds|kilograms)', r"\1 \2", row)

# Function to remove spaces (e.g. "Take 3" -> "Take3")
def remove_spaces(text):
    return text.replace(" ", "")

# Function to replace long hyphen ASCII code with short hyphen '-' ASCII code
def character_norm(text):
    return text.replace(chr(8211), "-")

# Word tokenizer splits ',' into separate token, so we have this function to do the same
def add_comma_token(text):
    return text.replace(",", " ,")

# Split '/' into its own token   JOE TO UPDATE THIS TINY EDGE CASE
def add_slash_token(text):
    return text.replace(chr(47), " / ")

# Word tokenizer splits ',' into separate token, so we have this function to do the same for our dates list
def add_date_var_comma_token(list):
    new_list = []
    for i in list:
        new_list.append(add_comma_token(i))
    return new_list

# Gets the first token of each date variation, to allow for faster downstream computation 
def get_first_token_set(list):
    new_set = set()
    for i in list:
        new_set.add(i.split()[0])
    return new_set

def get_item_set(row):
    item_set = set([])
    for i in row['item1'].split():
        item_set.add(i.lower())
    for j in row['item2'].split():
        item_set.add(j.lower())
    if 'nan' in item_set:
        item_set.remove('nan')
    return item_set

def preprocess(df):
    # Assign appropriate types
    string_cols = ["item1", "item2", "location", "organization", "date"]
    df[string_cols] = df[string_cols].astype(str)
    int_cols = ["weight1", "weight2"]
    for i in int_cols:
        df[i] = df[i].astype('Int64')

    # Normalize text columns to match tokenizer 
    df['text'] = df['text'].apply(lambda x: separate_weight_unit(x))
    for i in string_cols:
        df[i] = df[i].apply(lambda x: character_norm(x))
    df['text'] = df['text'].apply(lambda x: x.strip())

    # Tokenize text
    df['text_split'] = df['text'].apply(lambda x: nltk.word_tokenize(x))

    # Preprocess orgs and locations 
    df['org_no_space'] = df['organization'].apply(lambda x: remove_spaces(x))
    df['loc_no_space'] = df['location'].apply(lambda x: remove_spaces(x))

    # Preprocess ',' and '/' tokens
    for i in string_cols:
        df[i] = df[i].apply(lambda x: add_comma_token(x))
    for i in ["item1", "item2"]:
        df[i] = df[i].apply(lambda x: add_slash_token(x))

    # Create set of trash items of interest for each text
    df['item_set'] = df.apply(get_item_set, axis = 1)

    # Compute variations of date and weight formats and preprocess into desired formats
    df['date_vars'] = df['date'].apply(lambda x: date_to_formats(x) if x != 'nan' else str(x))
    df['weight1_text'] = df['weight1'].apply(lambda x: number_to_words(x)[1] if pd.notnull(x) else "")
    df['weight2_text'] = df['weight2'].apply(lambda x: number_to_words(x)[1] if pd.notnull(x) else "")
    df['date_vars'] = df['date_vars'].apply(lambda x: add_date_var_comma_token(x))
    df['date_vars_first_token'] = df['date_vars'].apply(lambda x: get_first_token_set(x))

    # Make string columns lowercase for downstream comparisons
    lowercase_cols = string_cols + ['organization', 'org_no_space', 'loc_no_space', 'weight1_text', 'weight2_text']
    for i in lowercase_cols:
        df[i] = df[i].apply(lambda x: x.lower())


def assign_entity_types(row):
    units = set(["kilograms", "kilogram", "kgs", "kg", "lb", "lbs", "pounds", "pound"])
    filler_words = set(["and", "the", "a", "an", ",", "/", "with"])

    words = row['text_split']
    new_tags = []
    prev_item_tag = False

    idx = 0
    while (idx < len(words)):
        loc_length = len(row['location'].split())
        org_length = len(row['organization'].split())
        weight1_text_length = len(row['weight1_text'].split())
        if row['weight2_text'] != None:
            weight2_text_length = len(row['weight2_text'].split())
        else:
            weight2_text_length = -1
        
        # Assign location labels
        # Checks for consecutive word matching for full location name (normalizing all words to lowercase)
        # Does not handle extraneous locations not provided in prompt!
        if ((idx <= len(words) - loc_length) and 
            [x.lower() for x in words[idx : idx + loc_length]] == row['location'].split()):
            new_tags.append("B-LOC")
            idx += 1
            for i in range(1, loc_length):
                new_tags.append("I-LOC")
                idx += 1
        elif (words[idx].lower() == row['loc_no_space']):
            new_tags.append("B-LOC")
            idx += 1

        # Assign organization labels
        # Checks for consecutive word matching for full location name (normalizing all words to lowercase)
        elif ((idx <= len(words) - org_length) and 
            [x.lower() for x in words[idx : idx + org_length]] == (row['organization'].lower().split())):
            new_tags.append("B-ORG")            # idea for later: tag acronyms for Orgs?
            idx += 1                            
            for i in range(1, org_length):
                new_tags.append("I-ORG")
                idx += 1
        elif (words[idx].lower() == row['org_no_space']):
            new_tags.append("B-ORG")      
            idx += 1
            
        # Assign unit labels
        elif words[idx].lower() in units:   
            new_tags.append("B-UNT")
            idx += 1
        
        # Assign weight labels for numeric and text numbers (consider '-' and non- '-' versions of written numbers?)
        elif (words[idx] == str(row['weight1']) or 
            (not pd.isna(row['weight2']) and words[idx] == str(row['weight2']))): 
            new_tags.append("B-WEI")
            idx += 1
        elif (not pd.isna(row['weight1']) and (idx <= len(words) - weight1_text_length) and 
                [x.lower() for x in words[idx : idx + weight1_text_length]] == row['weight1_text'].split()):
            new_tags.append("B-WEI")
            idx += 1
            for i in range(1, weight1_text_length):
                new_tags.append("I-WEI")
                idx += 1
        elif ((weight2_text_length > 0) and (idx <= len(words) - weight2_text_length) and 
                [x.lower() for x in words[idx : idx + weight2_text_length]] == row['weight2_text'].split()):
            new_tags.append("B-WEI")
            idx += 1
            for i in range(1, weight1_text_length):
                new_tags.append("I-WEI")
                idx += 1

        # Assign item labels (dont look for consecutive matches here)
        # Does not handle extraneous trash items not provided in prompt!
        elif (words[idx].lower() in row['item_set'] and words[idx].lower() not in filler_words):
            if prev_item_tag: 
                new_tags.append("I-ITM")
            else:
                new_tags.append("B-ITM")
                prev_item_tag = True
            idx += 1
        # Assign date labels (check only first token to minimize computation on each word)
        elif (words[idx] in row['date_vars_first_token']):
            # Check for complete consecutive match with any of the possible date variations 
            date_found = False
            for date_var in row['date_vars']:
                if ((idx <= len(words) - len(date_var.split())) and 
                    [x.lower() for x in words[idx : idx + len(date_var.split())]] == date_var.lower().split()):
                    new_tags.append("B-DAT")
                    idx += 1
                    for i in range(1, len(date_var.split())):
                        new_tags.append("I-DAT")
                        idx += 1
                    date_found = True
                    break
            # If the text matches with none of the date_vars, we need to append "O"
            if not date_found:
                new_tags.append("O")
                prev_item_tag = False
                idx += 1
        
        else:
            new_tags.append("O")
            prev_item_tag = False
            idx += 1

    return list(zip(words, new_tags))


# Compiles all sentences into a single list of lists (sentences) of word-pairs (word, NER tag)
def get_all_sentences(df):
    end_sentence = set(['.', '!', '?', '\n'])

    all_sentences = []
    for i in range(len(df)):
        idx = 0
        text_length = len(df.iloc[i]['tagged_entities'])
        # print("text length:", text_length)
        while idx < text_length:
            end = text_length - 1
            for j in range(idx, text_length):
                if df.iloc[i]['tagged_entities'][j][0] in end_sentence:
                    end = j
                    # print(j)
                    break
            
            # print("end", end)
            new_sentence = list(df.iloc[i]['tagged_entities'][idx : end + 1])
            all_sentences.append(new_sentence)
            idx = end + 1
    return all_sentences


def main(args=None):
    args = parse_args(args=args)
    input_file = args['csv_data_file']
    output_dir = args['savedir']
    save_prefix = args['save_prefix']

    df = pd.read_csv(args['csv_data_file'])
    df = preprocess(df)
    df['tagged_entities'] = df.apply(assign_entity_types, axis =1)
    all_sentences = get_all_sentences(df)
    print("# of sentences tagged: ", len(all_sentences))

    # Divide data into datasets = (train_sentences, dev_sentences, test_sentences)
    DEV_SPLIT = 0.1
    TEST_SPLIT = 0.1

    random.seed(1234)
    random.shuffle(all_sentences)

    train_sentences = all_sentences[ : int(len(all_sentences)*(1-DEV_SPLIT-TEST_SPLIT))]
    dev_sentences = all_sentences[int(len(all_sentences)*(1-DEV_SPLIT-TEST_SPLIT)) : int(len(all_sentences)*(1-TEST_SPLIT))]
    test_sentences = all_sentences[int(len(all_sentences)*(1-TEST_SPLIT)) : ]

    datasets = (train_sentences, dev_sentences, test_sentences)
    
    # Convert file and write to JSON file needed for Stanza modelling
    # out_directory = os.getcwd() + '/Processed_Data'
    write_dataset(datasets, out_directory, DATA_SELECTION)

if __name__ == '__main__':
    main()
