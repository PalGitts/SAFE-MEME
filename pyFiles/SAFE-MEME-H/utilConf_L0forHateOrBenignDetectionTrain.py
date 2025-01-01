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
log_file = f"./logFiles/test_benignOrNOT_fullFT_noCards.log"
        
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
            

            if mode.strip() in [
                'train_benignOrNOT_partialFT_singleCard', 
                'train_benignOrNOT_fullFT_noCards',
            ]:

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx, 'GENERAL_DESC'].strip() 
                label = df.loc[idx, 'LABEL'].strip().upper()
                # logger.info(f'*** Original label: {label}')

                inst = f'Instruction: Is the input instance hateful or benign? Response HATEFUL if it contains hate else BENIGN w.r.t. the given caption and image description.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'

                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                
                op_stmt = 'BENIGN [END]'
                if 'EXP' in label or 'IMP' in label:
                    op_stmt = 'HATEFUL [END]'
                 
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

                
                
            # Test
            if mode.strip() in [
                'test_benignOrNOT_fullFT_noCards', 
                'test_benignOrNOT_partialFT_singleCard'
                
            ]:

                print(f'CustomDatasetImg: {mode}: {dataset_type}')
                id_ = 'cnf_' +  df.loc[idx, 'IMG_ID'].strip().split('/')[-1].strip()
                
                # print(f'df: {df.columns}')
                ocr = df.loc[idx, 'OCR'].strip()
                gdesc = df.loc[idx,'GENERAL_DESC'].strip()
                label = df.loc[idx, 'LABEL'].strip().upper()
                logger.info(f'*** Original label: {label}')

                inst = f'Instruction: Is the input instance hateful or benign? Response HATEFUL if it contains hate else BENIGN w.r.t. the given caption and image description.'
                ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'

                
                
                op_stmt = '*'#'FALSE [END]'
                self.id2source_text[id_] = ip_stmt
                self.id2target_text[id_] = op_stmt

            # if mode.strip() in ['test_helperModuleAllQAGenCombined_GDesc']: # 256

            #     print(f'CustomDatasetImg: {mode}: {dataset_type}')
                
            #     ocr = df.loc[idx, 'OCR'].strip()
            #     gdesc = df.loc[idx, 'PRED_GDESC'].strip()
                
            #     op_stmt = '*'

            #     inst = f'Instruction: Please generate all questions and corresponding answers w.r.t. the given image text, image description.'
            #     ip_stmt = f'{inst}\nCaption: {ocr}\nImage description: {gdesc}'                
                



    def __len__(self):
        """returns the length of dataframe"""

        return self.df.shape[0]

    def __getitem__(self, index):
        
        """return the input ids, attention masks and target ids"""
        logger.info(f'CustomDatasetImg.__getitem__: START: {self.execution_mode}')

        if self.execution_mode.strip() in [
                'train_benignOrNOT_partialFT_singleCard', 
                'test_benignOrNOT_partialFT_singleCard',
                
                'train_benignOrNOT_fullFT_noCards',
                'test_benignOrNOT_fullFT_noCards',
        ]:

            id_ = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip().split('/')[-1].strip()
            img_id = 'cnf_' + self.df.loc[index, 'IMG_ID'].strip().split('/')[-1].strip()
        

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
