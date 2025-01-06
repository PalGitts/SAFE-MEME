import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from model import T5ForMultimodalGeneration
from utils_dataTrainUnitModels import img_shape, CustomDatasetImg
# from utils_prompt import *
# from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import time


'''
***** ATTENTION ***
The purpose of the main_trainUnitModels.py is to finetune the unit modules.It uses T5-base.  
Train the models using the commands as per the mode.

CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainAllQueryGen     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 128     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainAllQueryGen_1Q1A     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval


CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescAllQueryGen     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 128     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescGeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainContextGeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescContextGeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescAllQAPairs     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainContextAllQAPairs     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescContextAllQAPairs     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescContextAllQAPairs     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescAllQueryGen_1Q1A     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainGDescContextAllQueryGen_1Q1A     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  train_QASummaryGeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 350     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  train_QASummaryGenerationToCls     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 16     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  train_QASummaryToQApairsgeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 256     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  train_QASummaryToQuestionsGeneration     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 128     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  train_QASummaryQuestionsTo1Q1A     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainContextBasedAllQueryGen     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 30     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval
CUDA_VISIBLE_DEVICES=0 python3 main_trainUnitModels.py    --full_FT YES     --img_type vit     --execution_mode  trainContextBasedAnswerGen_1Q1A     --user_msg rationale     --epoch 50 --lr 5e-5 --output_len 85     --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval


'''
def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # List of valid modes
    mode_list = [
        'trainGDescGeneration', # i/p: OCR + Image => GDesc Generation. Expected o/p length - 85 
        
        # context realted
        'trainContextGeneration', # i/p: OCR + Image => Context. Expected o/p length - 256
        'trainGDescContextGeneration', # i/p: OCR + Image + GDesc => Context. Expected o/p length - 256
        
        #
        'trainGDescAllQAPairs', # i/p: OCR + Image + Gdesc => All relevant question, answer pairs. Expected o/p length - 256
        'trainContextAllQAPairs', # i/p: OCR + Image + Context => All relevant question, answer pairs. Expected o/p length - 256
        'trainGDescContextAllQAPairs', # i/p: OCR + Image + Gdesc + Context => All relevant question, answer pairs. Expected o/p length - 256

        # allQueryGen
        'trainAllQueryGen', # i/p: OCR + Image => All relevant questions. Expected o/p length - 128
        'trainGDescAllQueryGen', # i/p: OCR + Image + Gdesc => All relevant questions. Expected o/p length - 128
        
        
        # contextGen 2 Cls
        'trainContext2CLS', # i/p: OCR + Image + Context => Classification, Expected o/p length - 16
        'trainGDescContext2CLS', # i/p: OCR + Image + Gdesc + Context => Classification,  Expected o/p length - 16

        # 1Q1A
        'trainAllQueryGen_1Q1A', # i/p: OCR + Image + QuestionSet => AnswerSet. Expected o/p length - 85 
        'trainGDescAllQueryGen_1Q1A', # i/p: OCR + Image + GDesc + QuestionSet => AnswerSet. Expected o/p length - 85
        'trainGDescContextAllQueryGen_1Q1A', # i/p: OCR + Image + GDesc + QuestionSet => AnswerSet. Expected o/p length - 85

        'train_QASummaryGeneration', # i/p: OCR + Image + SummaryFromQAColumns => QAColumnsSummary. Expected o/p length - 350
        'train_QASummaryGenerationToCls', # i/p: QAColumnsSummary => Classification. Expected o/p length - 16
        'train_QASummaryToQApairsgeneration', # OCR + Image + SummaryFromQAColumns => All relevant question, answer pairs. Expected o/p length - 256
        'train_QASummaryToQuestionsGeneration', # OCR + Image + SummaryFromQAColumns => All relevant question. Expected o/p length - 128
        'train_QASummaryQuestionsTo1Q1A', # OCR + Image + QASummaryQuestions => All answers. Expected o/p length - 85

        'trainContextBasedAllQueryGen', # OCR + Image + Context => All relevant questions generation. Expected o/p length 30
        'trainContextBasedAnswerGen_1Q1A', # OCR + Image + Context + QuestionSet=> AnswerSet generation. Expected o/p length 85

    ]

    # parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--execution_mode', type=str, choices=mode_list)
    parser.add_argument('--full_FT', type=str, choices=['YES', 'NO'])
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default=f'unit_models/mm-cot-base-rationale/checkpoint-873/') 
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=12)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=512)
    parser.add_argument('--eval_bs', type=int, default=4)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=f'unit_models/mm-cot-base-rationale/checkpoint-873/', help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default='QCM-A', help='prompt format template',
                        choices=['QCM-A', 'QCM-E', 'QCM-LE', 'QCMG-A', 'QCM-LEA', 'QCM-ALE'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args

import pandas as pd
    
def get_df(id_list):
    
    df_initialized = False
    for idx_dataset in id_list:

        path = f'/home2/palash/p0_ImplicitHateDetection/EMNLP_2024/usable_datasets/RMMHS_F/RMMHS_{idx_dataset}.xlsx' # Please path to DatasetA-Regular.xlsx, *** ATTENTION ***
        temp_df = pd.read_excel(path)
        
        if df_initialized == False:
            df = temp_df
            df_initialized = True
        else:
            df = pd.concat([df, temp_df], axis=0, ignore_index=True)

        print(f'idx_dataset: {idx_dataset}: {df.shape}')
        
    return df

def reform_df(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUERY', 'ANSWER', 'CONTEXT']
             
    for idx in range(df.shape[0]):
        
        ocr = df.loc[idx, 'OCR']
        label = df.loc[idx, 'LABEL']
        path = df.loc[idx, 'PATH']    
        gdesc = df.loc[idx, 'GENERAL_DESC']
        id_ = df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
        
        context = df.loc[idx, 'QA'].strip().split('\n')[-1].split('#')[-1]
        qa_pairs = df.loc[idx, 'QA'].strip().split('\n')[1:-1]
        for idx_qa, qa_pair in enumerate( qa_pairs):
            if len(qa_pair) == 0:
                continue
            query, answer = qa_pair.split('#')
            query = query.strip()
            answer = answer.strip()
            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc,  query, answer, context]) 


    df = pd.DataFrame(all_rows, columns=col_names)
    return df
        

def reform_df_2(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUERY', 'ANSWER', 'CONTEXT', "QA_SUMMARY"]
             
    for idx in range(df.shape[0]):
        
        ocr = df.loc[idx, 'OCR']
        label = df.loc[idx, 'LABEL']
        path = df.loc[idx, 'PATH']    
        gdesc = df.loc[idx, 'GENERAL_DESC']
        id_ = df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
        qa_summary = df.loc[idx, 'QA_SUMMARY'].strip()
        
        context = df.loc[idx, 'QA'].strip().split('\n')[-1].split('#')[-1]
        qa_pairs = df.loc[idx, 'QA'].strip().split('\n')[1:-1]
        for idx_qa, qa_pair in enumerate( qa_pairs):
            if len(qa_pair) == 0:
                continue
            query, answer = qa_pair.split('#')
            query = query.strip()
            answer = answer.strip()
            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc,  query, answer, context, qa_summary]) 

    df = pd.DataFrame(all_rows, columns=col_names)

    return df
        

def reform_df_contextBasedQuestionGeneration(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUESTION_CONTEXT', 'QUERY']
             
    for idx in range(df.shape[0]):
        
        ocr = df.loc[idx, 'OCR']
        label = df.loc[idx, 'LABEL']
        path = df.loc[idx, 'PATH']    
        gdesc = df.loc[idx, 'GENERAL_DESC']
        id_ = df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
        
        qa_pairs = df.loc[idx, 'QA'].strip().split('\n')[1:-1]
        
        prev_query = ''
        prev_answer = ''
        contextBasedQuestionsUpto = ''

        for idx_qa, qa_pair in enumerate( qa_pairs):

            if len(qa_pair) == 0:
                continue

            query, answer = qa_pair.split('#')
            query =  query.strip()
            answer = 'Answer: ' + answer.strip()

            contextBasedQuestionsUpto += f'\n{prev_query}\n{prev_answer}\nQuestion: '

            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc, contextBasedQuestionsUpto, query])
            
            prev_query = query
            prev_answer = answer

    df = pd.DataFrame(all_rows, columns=col_names)
    return df

def reform_df_contextBasedAnswerGeneration_1Q1A(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUESTION_CONTEXT', 'ANSWER']
             
    for idx in range(df.shape[0]):
        
        ocr = df.loc[idx, 'OCR']
        label = df.loc[idx, 'LABEL']
        path = df.loc[idx, 'PATH']    
        gdesc = df.loc[idx, 'GENERAL_DESC']
        id_ = df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
        
        qa_pairs = df.loc[idx, 'QA'].strip().split('\n')[1:-1]
        
        prev_query = ''
        prev_answer = ''
        contextUpto = ''

        for idx_qa, qa_pair in enumerate( qa_pairs):

            if len(qa_pair) == 0:
                continue

            query, answer = qa_pair.split('#')
            query =  query.strip()
            answer = answer.strip()

            contextUpto += f'Question: {query}\nAnswer: '
            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc, contextUpto, answer])
            
            contextUpto += answer + '\n'

    df = pd.DataFrame(all_rows, columns=col_names)
    return df
        








def T5Trainer(args):
    print(f'args: {args}')
    print()

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    print(f'args.evaluate_dir: {args.evaluate_dir}')
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    print(f'*****   model: {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    exp_trn_list = [1,2,3,4,5,6,7,8]; exp_val_list = [9]; exp_test_list = [11, 12, 13]
    imp_trn_list = [14,15,16,17,18,19, 20]; imp_val_list =[21]; imp_test_list = [28, 29, 30, 31]
    ben_trn_list = [33, 34, 35, 36, 37, 38, 39]; ben_val_list =  [40]; ben_test_list= [43]
    
    execution_mode = args.execution_mode.strip()
    print(f'*****   execution_mode: {execution_mode}')

    if args.full_FT.upper() == 'YES':
        save_dir = f'unit_models/{args.img_type.upper()}_T5Base_FullFT_{execution_mode}'
    else:
        save_dir = f'unit_models/{args.img_type.upper()}_T5Base_partialFT_{execution_mode}'


    ###

    print(f'*****   save_dir: {save_dir}')
    time.sleep(2)    
    # 1/0
    
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
         
        
        # check
        if args.img_type == "resnet":
            image_features = np.load('vision_features/resnet.npy')
            image_features = np.expand_dims(image_features, axis=1)
            image_features = image_features.repeat(512, axis=1)
        
        if args.img_type == "clip":
            # image_features = np.load('vision_features/clip.npy')
            name_maps = json.load(open('vision_features/name_map_clip_trainC.json'))
            image_features = torch.load('vision_features/clip_trainC.pth')
            print(f'*****   image_feature: clip_trainC.pth: {image_features.shape}')
            model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, ignore_mismatched_sizes=True)
            print(f'*****   model loaded: {args.model}: CLIP')
            # time.sleep(5)


        if args.img_type == "detr":
            name_maps = json.load(open('vision_features/name_map_detr_trainC.json'))
            image_features = torch.load('vision_features/detr_trainC.pth')
            model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size, ignore_mismatched_sizes=True)
            print(f'*****   model loaded: {args.model}: DETR')
            # time.sleep(5)

        if args.img_type == "vit":
            image_features = torch.load("vision_features/vit_trainC.pth")
            print(f'image_model: vision_features/vit_trainC.pth')
            model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)
            name_maps = json.load(open('vision_features/name_map_vit_trainC.json'))
            print(f'*****   model loaded: {args.model}: VIT')
        
        # Freeze model
        if args.full_FT.upper() in ['NO']:
            for name, param in model.named_parameters():
                if 'gate_dense' in name or 'block.11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
                print(f'{name}: {param.requires_grad}')

        #
        # train_path = f'./DatasetA-Regular/DatasetA-Regular_train.xlsx'
        # val_path = f'./DatasetA-Regular/DatasetA-Regular_validation.xlsx'
        
        # train_df = pd.read_excel(train_path)
        # eval_df = pd.read_excel(val_path)

        train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
        eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
        
        if execution_mode in [
            'train_QASummaryGeneration', 
            'train_QASummaryToQApairsgeneration', 
            'train_QASummaryToQuestionsGeneration',
            'train_QASummaryQuestionsTo1Q1A',
            'train_QASummaryGenerationToCls'
        ]:
            
            path = '' # Please path to DatasetA-Regular.xlsx, *** ATTENTION ***
            train_df = pd.read_excel(path)
            
        
        if execution_mode in [ 'trainGDescAllQueryGen_1Q1A', 'trainGDescContextAllQueryGen_1Q1A', 'trainAllQueryGen_1Q1A']:
            train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
            train_df = reform_df(train_df)
            
        if execution_mode in ['train_QASummaryQuestionsTo1Q1A']:
            train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
            train_df = reform_df_2(train_df)
            

        if execution_mode in ['trainContextBasedAllQueryGen']:
            train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
            train_df = reform_df_contextBasedQuestionGeneration(train_df)

        if execution_mode in ['trainContextBasedAnswerGen_1Q1A']:
            train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
            train_df = reform_df_contextBasedAnswerGeneration_1Q1A(train_df)

        train_set = CustomDatasetImg(
            train_df,
            tokenizer,
            name_maps,
            args.input_len,
            args.output_len,
            image_features,
            execution_mode,
            args,
            'TRAIN',
            args.eval_le
        )
        print(f'Training set is created.')
        
        if execution_mode in ['trainGDescAllQueryGen_1Q1A', 'trainGDescContextAllQueryGen_1Q1A', 'trainAllQueryGen_1Q1A']:
            eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
            eval_df = reform_df( eval_df)
            

        if execution_mode in ['train_QASummaryQuestionsTo1Q1A']:
            eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
            eval_df = reform_df_2(eval_df)

        if execution_mode in ['trainContextBasedAllQueryGen']:
            eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
            eval_df = reform_df_contextBasedQuestionGeneration(eval_df)

        if execution_mode in ['trainContextBasedAnswerGen_1Q1A']:
            eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
            eval_df = reform_df_contextBasedAnswerGeneration_1Q1A(eval_df)

        eval_set = CustomDatasetImg(
            eval_df,
            tokenizer,
            name_maps,
            args.input_len,
            args.output_len,
            image_features,
            execution_mode,
            args,
            'VAL',
            args.eval_le
        )
        print(f'Evaluation set is created.')
        

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())
    def extract_ans(ans):
        pattern = re.compile(r'The answer is \(([A-Z])\)')
        res = pattern.findall(ans)
        
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED" 
        return answer  

    # accuracy for answer inference
    def compute_metrics_acc(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        assert len(preds) == len(targets)
        for idx, pred in enumerate(preds):
            reference = targets[idx]
            reference = extract_ans(reference)
            extract_pred = extract_ans(pred)
            best_option = extract_pred
            if reference == best_option:
                correct +=1 
        return {'accuracy': 1.0*correct/len(targets)}
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    print(f'args.final_eval: {args.final_eval}')
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy='no',
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 10,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            # load_best_model_at_end=True,
            report_to="none",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 10,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size= 4,#args.bs,
            per_device_eval_batch_size= 4, #args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="none",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    print(f'\n *** Training: START: execution_mode: {execution_mode}: save_dir: {save_dir}')
    trainer.train()
    # trainer.save_model(save_dir)

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)


    T5Trainer(args = args)
