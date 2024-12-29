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

            
            if mode.strip() in ['trainGDescGeneration', 'traingdescgeneration', 'TRAINGDESCGENERATION']: # bs - 512

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                inst = f'Instruction: Please generate a description of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                op_stmt = f"{gdesc} [END]"
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['trainContextGeneration']: # bs - 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                inst = f'Instruction: Please describe the context of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                op_stmt = f"{context} [END]"
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            
            if mode.strip() in ['trainGDescContextGeneration']: # 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                inst = f'Instruction: Please describe the context of the given image.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'
                
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                op_stmt = f"{context} [END]"
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt


            # allQAPairs 
            if mode.strip() in ['trainGDescAllQAPairs']: # bs - 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'                
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt)
                op_stmt += '[END]'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['trainContextAllQAPairs']: # 256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, and image context.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage Context: {context}'                
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt)
                op_stmt += '[END]'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt
                

            if mode.strip() in ['trainGDescContextAllQAPairs']: # bs - 256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                
                inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description and image context.'
                
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nImage Context: {context}'                
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt)
                op_stmt += '[END]'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt
                

            # allQueryGen
            if mode.strip() in ['trainAllQueryGen']: # bs - 128

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                # gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                # context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                # if len(context) < 10:
                #     print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                #     1/0
                
                all_qas = df.loc[idx, 'QA'].strip().split('\n')[:-1]
                
                all_query = []
                for qa in all_qas:
                    
                    if len(qa.strip()) == 0:
                        continue
                    all_query.append( qa.split('#')[0] )

                op_stmt = ' [SEP] '.join(all_query) + ' [END]'
                
                inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and the image.'
                ip_stmt = f'{inst}\nCaption: {ocr}'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt


            if mode.strip() in ['trainGDescAllQueryGen']: # bs - 128

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                # context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                # if len(context) < 10:
                #     print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                #     1/0
                
                all_qas = df.loc[idx, 'QA'].strip().split('\n')[:-1]
                
                all_query = []
                for qa in all_qas:
                    
                    if len(qa.strip()) == 0:
                        continue
                    all_query.append( qa.split('#')[0] )

                op_stmt = ' [SEP] '.join(all_query) + ' [END]'
                
                inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and image description .'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            
            # Context2CLS
            if mode.strip() in ['trainContext2CLS']: # bs-16
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                
                op_stmt = f"{df.loc[idx, 'LABEL']} [END]"
                
                inst = f'Instruction: The given meme can belong to any one of three following category - explicit hate, implicit hate, and benign. Please classify the input meme w.r.t. the given image text, and image context.'
                query = f'What kind of hate is it?'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage context: {context}\nQuestion: {query}'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['trainGDescContext2CLS']: # bs - 16
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                
                context = df.loc[idx, 'QA'].strip().split('\n')[-1].strip().split('#')[-1].strip()
                if len(context) < 10:
                    print(f"******  Error in context: {df.loc[idx, 'LABEL']}: {df.loc[idx, 'PATH']}")
                    1/0
                
                op_stmt = f"{df.loc[idx, 'LABEL']} [END]"
                
                inst = f'Instruction: The given meme can belong to any one of three following category - explicit hate, implicit hate, and benign. Please classify the input meme w.r.t. the given image text, image description, and image context.'
                query = f'What kind of hate is it?'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nImage Context: {context}\nQuestion: {query}'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt
                
                
            # 1Q1A
            
            if mode.strip() in ['trainAllQueryGen_1Q1A']: # bs-85
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                query = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'ANSWER'].strip() + ' [END]'

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nQuestion: {query}'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            if mode.strip() in ['trainGDescAllQueryGen_1Q1A']: # bs-85
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                query = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'ANSWER'].strip() + ' [END]'

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image description.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nQuestion: {query}'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
            
            if mode.strip() in ['trainGDescContextAllQueryGen_1Q1A']: # bs-85
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                context = df.loc[idx, 'CONTEXT'].strip()
                query = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'ANSWER'].strip() + ' [END]'

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption, image description and image context.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nImage context: {context}\nQuestion: {query}'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
            
            # v3

            if mode.strip() in ['train_QASummaryGeneration']: # bs- 350
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                op_stmt = df.loc[idx, 'QA_SUMMARY'].strip() + ' [END]'
                
                inst = f'Instruction: Please generate a precise summary against the given meme and corresponding caption.'
                ip_stmt = f'{inst}\nCaption: {ocr}'

                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
            
            if mode.strip() in ['train_QASummaryToQApairsgeneration']: # bs-256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                qa_summary = df.loc[idx, 'QA_SUMMARY'].strip()
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt) + ' [END]'
                
                inst = f'Instruction: Please generate a series of relevant question, answer pairs against the give input summary.'
                ip_stmt = f'{inst}\nSummary: {qa_summary}'

                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            if mode.strip() in ['train_QASummaryToQuestionsGeneration']: # bs-256
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                qa_summary = df.loc[idx, 'QA_SUMMARY'].strip()
                all_qaPairs = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                
                all_questins = []
                for pair in all_qaPairs:
                    
                    pair = pair.strip()
                    if len(pair) == 0:
                        continue

                    try:
                        query, answer = pair.split(f'#')
                        query  = query.strip()
                        answer = answer.strip()
                    
                    except Exception as e:
                        print(f'Exception {e}: {pair}')
                        query = pair
                        1/0

                    all_questins.append(query)

                op_stmt = ' [SEP] '.join(all_questins) + ' [END]'
                
                inst = f'Instruction: Please generate a series of relevant question against the give input summary.'
                ip_stmt = f'{inst}\nSummary: {qa_summary}'

                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            if mode.strip() in ['train_QASummaryQuestionsTo1Q1A']: # bs-85
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
               
                query = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'ANSWER'].strip() + ' [END]'
                qa_summary = df.loc[idx, 'QA_SUMMARY'].strip()

                inst = f'Instruction: Please generate an answer against the given question w.r.t. the given summary.'
                ip_stmt = f'{inst}\nSummary: {qa_summary}\nQuestion: {query}'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            
            if mode.strip() in ['train_QASummaryGenerationToCls']: # bs-16
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
               
                # ocr = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'LABEL'].strip() + ' [END]'
                qa_summary = df.loc[idx, 'QA_SUMMARY'].strip()

                inst = f'Instruction: Please classify the given summary into one of the following categories - explicit hate, implicit hate and benign.'
                ip_stmt = f'{inst}\nSummary: {qa_summary}'

                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            #
            ###
            ####            
            if mode.strip() in ['trainContextBasedAllQueryGen']: # ol-30
                
                #    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUESTION_CONTEXT', 'QUERY']

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
               
                query = df.loc[idx, 'QUERY'].strip()
                context = df.loc[idx, 'QUESTION_CONTEXT'].strip()
                inst = f'Instruction: Please generate the next question w.r.t. the given context.'
                ip_stmt = f'{inst}\n{context}'
                op_stmt = df.loc[idx, 'QUERY'] + ' [END]'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt


            
            if mode.strip() in ['trainContextBasedAnswerGen_1Q1A']: # ol-85
                
                #    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUESTION_CONTEXT', 'ANSWER']

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
               
                exp_answer = df.loc[idx, 'ANSWER'].strip()
                context = df.loc[idx, 'QUESTION_CONTEXT'].strip()
                inst = f'Instruction: Please generate the appropriate answer w.r.t. the given context.'
                ip_stmt = f'{inst}\n{context}'
                op_stmt = exp_answer + ' [END]'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt
            
                


    def __len__(self):
        """returns the length of dataframe"""

        return self.df.shape[0]

    def __getitem__(self, index):
        
        """return the input ids, attention masks and target ids"""
        # print(f'CustomDatasetImg.__getitem__: START')

        if self.execution_mode in ['train_QASummaryToQuestionsGeneration']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['train_QASummaryToQApairsgeneration']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['train_QASummaryGeneration']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        # description generation
        if self.execution_mode in ['trainGDescGeneration', 'traingdescgeneration', 'TRAINGDESCGENERATION']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()


        # context generation
        if self.execution_mode in ['trainContextGeneration']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['trainContextAllQAPairs']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['trainGDescContextGeneration']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        

        # allQAPairs
        if self.execution_mode in ['trainGDescAllQAPairs']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        if self.execution_mode in ['trainGDescContextAllQAPairs']:
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        # allQueryGen 
        if self.execution_mode in ['trainGDescAllQueryGen', 'trainAllQueryGen']: # bs - 128
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        # 1Q1A
        if self.execution_mode in ['trainGDescAllQueryGen_1Q1A', 'trainGDescContextAllQueryGen_1Q1A', 'trainAllQueryGen_1Q1A']:
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        
        if self.execution_mode in ['train_QASummaryQuestionsTo1Q1A']:
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['train_QASummaryQuestionsTo1Q1A']:
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        
        # context2CLS
        if self.execution_mode in ['trainContext2CLS']: # bs - 16
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        if self.execution_mode in ['trainGDescContext2CLS']: # bs - 16
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        if self.execution_mode in ['train_QASummaryGenerationToCls']: # bs - 16
            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        #
        if self.execution_mode in ['trainContextBasedAllQueryGen']: # ol -30
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in ['trainContextBasedAnswerGen_1Q1A']: # ol -85
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        
        
        image_ids = int(self.name_map[img_id])

        source_text = str( self.id2source_text[id_])
        target_text = str( self.id2target_text[id_])

        # print()
        # print(f'*****   I/P: {source_text}')
        # print(f'*****   O/P: {target_text}')
        # print()

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
        # print(f'self.image_features: {image_vector.shape}')
        
        if 'CLIP' not in self.args.img_type.upper():
            image_vector = torch.tensor(image_vector).squeeze()
        # print(f'self.image_features: {image_vector.shape}')
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_vector,
            "labels": target_ids,
        }
