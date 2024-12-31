import time
import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
# from utils_prompt import *
from extract_features import *
import logging
# import timm
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 1024),
    "detr": (100, 256),
    "vit": (145, 1024),
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = f"./logFiles/mmCoT_singleCardTrain.log"
        
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
        category,
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
        self.category = category

        # print(f'******************** image_features: {self.image_features.shape}')
        
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)
        logger.info(f"START for util_data: Test for {self.category}")
            
        # logger.info(f'df.shape: {df.shape}')

        try:
            df['TARGET'] = df['TARGET'].str.upper()
        except Exception as e:
            logger.info(f'In CustomDatasetImg: {e}')

        self.category2save_dict = {

        'ISLAM': 'ISLAM',
        'DISABILITY': 'DISABILITY',
        'THE_LGBTQ_COMMUNITY': 'LGBTQ', 
        'WOMEN_IN_GENERAL': 'WOMEN',
        'MEN_IN_GENERAL': 'MEN',
        'THE_BLACK_COMMUNITY': 'BLACK',
        'OTHERS': 'OTHERS',
        'THE_JEWISH_COMMUNITY': 'JEWS',
        'THE_WHITE_COMMUNITY': 'WHITE',
        'THE_PEOPLE_WITH_DISABILITY': 'DISABILITY',
        'THE_IMMIGRANT_PEOPLE': 'IMMIGRANT',    
        'NO_BODY': 'NO_BODY',
        'GENERAL': 'GENERAL',

        
        'THE LGBTQ COMMUNITY': 'LGBTQ', 
        'WOMEN IN GENERAL': 'WOMEN',
        'MEN IN GENERAL': 'MEN',
        'NO BODY': 'NOBODY',
        'THE BLACK COMMUNITY': 'BLACK',
        'THE JEWISH COMMUNITY': 'JEWS',
        'THE WHITE COMMUNITY': 'WHITE',
        'THE PEOPLE WITH DISABILITY': 'DISABILITY',
        'THE DISABLED PEOPLE': 'DISABILITY',
        'THE IMMIGRANT PEOPLE': 'IMMIGRANT',
        'CHRISTIANITY': 'OTHERS',
        'HINDUISM': 'OTHERS',
        'NO BODY': 'NO_BODY',
        'NOBODY': 'NO_BODY',
        'BLACK':'BLACK',
        

        }

        
        for idx in range(df.shape[0]):
            
            # if mode.strip() in ['train_helperModuleAllQuestionGenCategorySpecific', 'train_helperModuleAllQuestionGenCombined']: # ol -90

            #     logger.info(f'CustomDatasetImg: {mode}: {dataset_type}')
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     all_qas = df.loc[idx, 'QA'].strip().split('\n')[:-1]
            #     all_query = []
            #     for qa in all_qas:
                    
            #         if len(qa.strip()) == 0:
            #             continue
            #         all_query.append( qa.split('#')[0] )

            #     op_stmt = ' [SEP] '.join(all_query) + ' [END]'
                
            #     inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and the image.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}'
                
            #     id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
            #     self.id2target_text[id_] = op_stmt
            #     self.id2source_text[id_] = ip_stmt

            if mode.strip() in ['train_gDescGeneration_singleCards']:

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                op_stmt = df.loc[idx, 'GENERAL_DESC'].strip() + ' [END]'

                inst = f'Instruction: Please generate an description of the given meme instance.'
                ip_stmt = f'{inst}\nCaption: {ocr}'

                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

                
                
            #

            if mode.strip() in [
                'train_noGDescSingleCards_1Q1A', 
                'train_withGDescSingleCards_1Q1A',
            ]: # bs-85
                
                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                query = df.loc[idx, 'QUERY'].strip()
                op_stmt = df.loc[idx, 'ANSWER'].strip() + ' [END]'

                
    
                if 'withGDesc' in mode: # For with GDesc
                    inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image description.'
                    ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nQuestion: {query}'
                else:
                    inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image.'
                    ip_stmt = f'{inst}\nCaption: {ocr}\nQuestion: {query}'

                id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt


            if mode.strip() in [
                'train_noGDescSingleCardsAllQAGen',
                'train_withGDescSingleCardsAllQAGen',
            ]: # bs - 256

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
                
                if 'withGDesc' in mode: # For with GDesc
                    inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description.'
                    ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'                
                
                else:  # For no GDesc
                    inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text.'
                    ip_stmt = f'{inst}\nCaption: {ocr}'                
                
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt)
                op_stmt += '[END]'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            if mode.strip() in [
                'train_noGDescSingleCardsAllQueriesGen',
                'train_withGDescSingleCardsAllQueriesGen',
            ]: # bs - 90

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip()

                all_qas = df.loc[idx, 'QA'].strip().split('\n')[:-1]
                all_query = []
                for qa in all_qas:
                    
                    if len(qa.strip()) == 0:
                        continue
                    all_query.append( qa.split('#')[0] )

                op_stmt = ' [SEP] '.join(all_query) + ' [END]'
                
                if 'withGDesc' in mode: # For with GDesc
                    inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and the image.'
                    ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'
                
                else:  # For no GDesc
                    inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image.'
                    ip_stmt = f'{inst}\nCaption: {ocr}'
                
                op_stmt = df.loc[idx, 'QA'].strip().split('\n')[:-1] 
                op_stmt = ' [SEP] '.join(op_stmt)
                op_stmt += '[END]'
                
                id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
                self.id2target_text[id_] = op_stmt
                self.id2source_text[id_] = ip_stmt

            # Test
            # if mode.strip() in ['test_helperModuleAllQAGenCombined_GDesc']: # 256

            #     print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     gdesc = df.loc[idx, 'PRED_GDESC'].strip()
                
            #     op_stmt = '*'

            #     inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'                
                


            #     id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
            #     self.id2source_text[id_] = ip_stmt
            #     self.id2target_text[id_] = op_stmt

            # if mode.strip() in ['test_helperModule1Q1AGenCombined_GDesc']: # ol-85
                
            #     print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     gdesc = df.loc[idx, 'PRED_GDESC'].strip()
            #     query = df.loc[idx, 'PRED_QUERY'].strip()
            #     op_stmt = '*'

            #     inst = f'Instruction: Please generate an answer against the given question w.r.t. the given caption and image description.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}\nQuestion: {query}'



            #     id_ = 'rmmhs_' +  df.loc[idx, 'ID'].strip().split('/')[-1].strip()
            #     self.id2source_text[id_] = ip_stmt
            #     self.id2target_text[id_] = op_stmt

            # if mode.strip() in ['test_helperModuleGDescCombined']: # ol - 128
                
            #     logger.info(f'CustomDatasetImg: {mode}: {dataset_type}')
                
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     inst = f'Instruction: Please generate a description of the given image.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}'
                
            #     gdesc = df.loc[idx, 'GENERAL_DESC'].strip()
            #     op_stmt = f"{gdesc} [END]"
                
            #     id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
            #     self.id2target_text[id_] = op_stmt
            #     self.id2source_text[id_] = ip_stmt

            # if mode.strip() in ['test_helperModuleAllQuestionGenCombined']: # ol -90 # No Gdesc

            #     logger.info(f'CustomDatasetImg: {mode}: {dataset_type}')
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     # all_qas = df.loc[idx, 'QA'].strip().split('\n')[:-1]
            #     all_qas = '*'
            #     all_query = []
            #     for qa in all_qas:
                    
            #         if len(qa.strip()) == 0:
            #             continue
            #         all_query.append( qa.split('#')[0] )

            #     op_stmt = ' [SEP] '.join(all_query) + ' [END]'
                
            #     inst = f'Instruction: Please generate all questions that is required to explain the context w.r.t. the given image text, and the image.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}'
                
            #     id_ = 'rmmhs_' +  df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
                
            #     self.id2target_text[id_] = op_stmt
            #     self.id2source_text[id_] = ip_stmt





    def __len__(self):
        """returns the length of dataframe"""

        return self.df.shape[0]

    def __getitem__(self, index):
        
        """return the input ids, attention masks and target ids"""
        logger.info(f'CustomDatasetImg.__getitem__: START: {self.execution_mode}')

        if self.execution_mode.strip() in ['train_gDescGeneration_singleCards']:

            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        if self.execution_mode.strip() in [
            'train_noGDescSingleCardsAllQAGen',
            'train_withGDescSingleCardsAllQAGen'
        ]:

            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()


        if self.execution_mode in [
            'train_noGDescSingleCardsAllQueriesGen',
            'train_withGDescSingleCardsAllQueriesGen',
        ]: # ol - 90

            id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        # if self.execution_mode in ['train_helperModuleGDescCategorySpecific', 'train_helperModuleGDescCombined']:
        #     id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        if self.execution_mode in [
            'train_noGDescSingleCards_1Q1A', 
            'train_withGDescSingleCards_1Q1A',
        ]:
            id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
            img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        
        
        # if self.execution_mode in ['train_helperModuleAllQACategorySpecific', 'train_helperModuleAllQAGenCombined']:
        #     id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        # # *** Test ***
        
        # if self.execution_mode.strip() in ['test_helperModuleAllQAGenCombined_GDesc']: # 256
            
        #     id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        # if self.execution_mode.strip() in ['test_helperModule1Q1AGenCombined_GDesc']: # ol-85
            
        #     id_ = 'rmmhs_' + self.df.loc[index, 'ID'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()


        # if self.execution_mode.strip() in ['test_helperModuleAllQuestionGenCombined']: # ol -90 # No Gdesc
            
        #     id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()



        # if self.execution_mode in ['test_helperModuleGDescCombined']:
            
        #     id_ = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()
        #     img_id = 'rmmhs_' + self.df.loc[index, 'PATH'].strip().split('/')[-1].strip()

        image_ids = int(self.name_map[img_id])

        source_text = str( self.id2source_text[id_])
        target_text = str( self.id2target_text[id_])

        logger.info(f'\n*** {self.execution_mode}: i/p: {source_text}: {self.execution_mode}')
        logger.info(f'*** {self.execution_mode}: o/p: {target_text}: {self.execution_mode}')
        logger.info(f'*** {self.execution_mode}: image_ids: {image_ids}')

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
