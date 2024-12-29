import time
import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *
from extract_features import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 1024),
    "detr": (100, 256),
    "vit": (145, 1024),
}



class CustomDatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, 
        df, 
        tokenizer, 
        name_map , 
        source_len, 
        target_len, 
        image_features, 
        mode, 
        args,
        dataset_type='TRAIN', 
        test_le=None
    ):  
        # print(f'CustomDatasetImg.__init__: START')
        self.df = df
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.summ_len = target_len
        self.name_map = name_map
        self.image_features = image_features
        self.id2target_text = {}
        self.id2source_text = {}
        self.id2image_ids = {}
        self.execution_mode = mode
        self.args = args

        # print(f'******************** image_features: {self.image_features.shape}')
        
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None

        print(f'df.index: {df.index}')
        for idx in range(df.shape[0]):

            
            if mode.strip() in ['testGDescGeneration']: # bs - 512

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                inst = f'Instruction: Please generate a description of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                op_stmt = f"*"
                
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['testContextGeneration']: # bs - 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                inst = f'Instruction: Please describe the context of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                op_stmt = f"*"
                
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            
            if mode.strip() in ['testGDescContextGeneration']: # 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                json_data = json.load( open(f'./results/testConfounder_memeGDescGeneration_output.json'))
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                
                ocr = df.loc[idx, 'OCR'].strip()

                gdesc = json_data[id_]['pred_memeGDesc'].strip().replace('[END]', '')
                inst = f'Instruction: Please describe the context of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'
                
                op_stmt = f"*"
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt


            # # allQAPairs 
            if mode.strip() in ['testGDescAllQAPairs']: # bs - 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                json_path = f'./results/testConfounder_memeGDescGeneration_output.json'
                json_data = json.load(open(json_path))
                
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                
                ocr = json_data[id_]['OCR'].strip()
                gdesc = json_data[id_]['pred_memeGDesc'].strip().replace('[END]', '')
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'                
                op_stmt = '*'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['testContextAllQAPairs']: # 256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                json_path = f'./results/testConfounder_memeContextGeneration_output.json'
                json_data = json.load(open(json_path))
                
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID']
                
                ocr = df.loc[idx, 'OCR'].strip()
                context = json_data[id_]['pred_memeContext'].strip().replace('[END]', '')
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, and image context.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage Context: {context}'                
                op_stmt = '*'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt
                

            if mode.strip() in ['testGDescContextAllQAPairs']: # bs - 256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                
                json_path = f'./results/testConfounder_memeContextGeneration_output.json'
                json_dataContext = json.load(open(json_path))
                context = json_dataContext[id_]['pred_memeContext'].strip().replace('[END]', '')

                json_path = f'./results/testConfounder_memeGDescGeneration_output.json'
                json_dataGDesc = json.load(open(json_path))
                gdesc = json_dataGDesc[id_]['pred_memeGDesc'].strip().replace('[END]', '')
                ocr = json_dataGDesc[id_]['OCR'].strip()
                
                
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description and image context.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nImage Context: {context}'                
                op_stmt = '*'
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt
                

            # allQueryGen
            if mode.strip() in ['testAllQueryGen']: # bs - 128

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                
                ocr = df.loc[idx, 'OCR'].strip()
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                
                op_stmt = '*'
                
                inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and the image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt


            if mode.strip() in ['testGDescAllQueryGen']: # bs - 128

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'PRED_GDESC'].strip().replace('[SEP]', '').replace('[END]', '')
                op_stmt = '*'
                
                inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and image description .'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            
            # Context2CLS
            if mode.strip() in ['testContext2CLS']: # bs-16
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                
                json_path = f'./results/testConfounder_memeContextGeneration_output.json'
                json_dataContext = json.load(open(json_path))
                context = json_dataContext[id_]['pred_memeContext'].strip().replace('[END]', '')

                op_stmt = f"*"
                
                inst = f'Instruction: The given meme can belong to any one of three following category - explicit hate, implicit hate, and benign. Please classify the input meme w.r.t. the given image text, and image context.'
                query = f'What kind of hate is it?'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage context: {context}\nQuestion: {query}'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['testGDescContext2CLS']: # bs - 16
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                
                json_path = f'./results/testConfounder_memeGDescContextGeneration_output.json'
                json_dataContext = json.load(open(json_path))
                context = json_dataContext[id_]['pred_memeGDescContext'].strip().replace('[END]', '')

                json_path = f'./results/testConfounder_memeGDescGeneration_output.json'
                json_dataGDesc = json.load(open(json_path))
                gdesc = json_dataGDesc[id_]['pred_memeGDesc'].strip().replace('[END]', '')
                
                op_stmt = f"*"
                
                inst = f'Instruction: The given meme can belong to any one of three following category - explicit hate, implicit hate, and benign. Please classify the input meme w.r.t. the given image text, image description, and image context.'
                query = f'What kind of hate is it?'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nImage Context: {context}\nQuestion: {query}'
                
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt


            # 1Q1A
            if mode.strip() in ['testAllQueryGen_1Q1A']: # ol-85
                
                ocr = df.loc[idx, 'OCR'].strip()
                query = df.loc[idx, 'PRED_QUERY'].strip()
                op_stmt = '*'

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nQuestion: {query}'

                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
                
            # 1Q1A
            if mode.strip() in ['testGDescAllQueryGen_1Q1A']: # bs-256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'PRED_GDESC'].strip()
                query = df.loc[idx, 'PRED_QUERY'].strip()
                op_stmt = '*'

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given given caption and image description.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nQuestion: {query}'

                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
                    
                


    def __len__(self):
        """returns the length of dataframe"""

        return self.df.shape[0]

    def __getitem__(self, index):
        
        """return the input ids, attention masks and target ids"""
        # print(f'CustomDatasetImg.__getitem__: START')

        
        # description generation
        if self.execution_mode in ['testGDescGeneration']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()


        # context generation
        if self.execution_mode in ['testContextGeneration']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()

        if self.execution_mode in ['testGDescContextGeneration']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
        



        # allQAPairs
        if self.execution_mode in ['testGDescAllQAPairs']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()

        if self.execution_mode in ['testContextAllQAPairs']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()

        if self.execution_mode in ['testGDescContextAllQAPairs']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
        
        # allQueryGen 
        if self.execution_mode in ['testAllQueryGen', 'testGDescAllQueryGen']: # bs - 128
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
        
        # 1Q1A
        if self.execution_mode in ['testAllQueryGen_1Q1A', 'testGDescAllQueryGen_1Q1A']:
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip().split('/')[-1].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()

        # context2CLS
        if self.execution_mode in ['testContext2CLS']: # bs - 16
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
        
        if self.execution_mode in ['testGDescContext2CLS']: # bs - 16
            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip()
        



        
        image_ids = int(self.name_map[img_id])

        source_text = str( self.id2source_text[id_])
        target_text = str( self.id2target_text[id_])

        print()
        print(f'*****   I/P: {source_text}')
        print(f'*****   O/P: {target_text}')
        print()

        # 1/0
        

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        
        # print(f'self.image_features: {self.image_features.shape}')
        image_vector = self.image_features[image_ids]   
        
        # print(self.args.img_type.upper())
        print(f'self.image_features: {self.image_features.shape}')
        print(f'self.image_vector: {image_vector.shape}')
        
        if 'CLIP' not in self.args.img_type.upper():
            image_vector = torch.tensor(image_vector).squeeze()
        # print(f'self.image_features: {image_vector.shape}')
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_vector,
            "labels": target_ids,
        }
