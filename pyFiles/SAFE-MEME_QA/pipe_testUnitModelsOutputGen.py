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
from utils_dataTestUnitModels import img_shape, CustomDatasetImg
# from utils_prompt import *
# from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import time
from tqdm import tqdm


'''
***** ATTENTION ****
The file 'pipe_testUnitModelsOutputGen' is used for inference of DatasetAB-Regular and NO training.
Kindly run the commands for testAllQueryGen, and testAllQueryGen_1Q1A sequentially.  

CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testAllQueryGen         --img_type vit         --output_len 85         --full_FT YES         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testAllQueryGen_1Q1A         --img_type vit         --output_len 85         --full_FT YES         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval


CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescGeneration        --img_type vit         --output_len 512         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testContextGeneration         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescContextGeneration         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescAllQAPairs         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testContextAllQAPairs         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescContextAllQAPairs         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testContext2CLS         --img_type vit         --output_len 16         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescContext2CLS         --img_type vit         --output_len 16         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval

CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescAllQueryGen         --img_type vit         --output_len 128         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval

CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  testGDescAllQueryGen_1Q1A         --img_type vit         --output_len 85         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  test_QASummaryGeneration         --img_type vit         --output_len 350         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  test_QASummaryToQApairsgeneration         --img_type vit         --output_len 256         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  test_QASummaryToQuestionsGeneration         --img_type vit         --output_len 128         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval
CUDA_VISIBLE_DEVICES=0 python3 pipe_testUnitModelsOutputGen.py         --execution_mode  test_QASummaryQuestionsTo1Q1A         --img_type vit         --output_len 85         --full_FT YES         --model declare-lab/flan-alpaca-large         --user_msg rationale         --epoch 50         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval

'''
def parse_args():

    all_modes = [
        'testGDescGeneration', 
        'testContextGeneration',
        'testGDescContextGeneration',

        'testGDescAllQAPairs',
        'testContextAllQAPairs',
        'testGDescContextAllQAPairs',

        # contextGen 2 Cls
        'testContext2CLS', 
        'testGDescContext2CLS', 

        # All QueryGen
        'testAllQueryGen', 
        'testGDescAllQueryGen',
        
        # 1Q1A
        'testAllQueryGen_1Q1A',
        'testGDescAllQueryGen_1Q1A', 
        # 'testGDescContextAllQueryGen_1Q1A' 

        'test_QASummaryGeneration', # 
        'test_QASummaryToQApairsgeneration', # 
        'test_QASummaryToQuestionsGeneration', # 
        'test_QASummaryQuestionsTo1Q1A', # 

    ]

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--execution_mode', type=str, choices=all_modes)
    parser.add_argument('--full_FT', type=str, choices=['YES', 'NO'],  default='YES')
    parser.add_argument('--dataset_name', type=str, choices=['rmmhs_v2', 'confounder'])
    
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=2)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
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
        
        # print(df.shape)
        # print(df.columns)
        
    return df
    
def reform_df(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QA']
    json_result = json.load(open('results/test_allQuestions_output.json'))
    
    for key, _dict in json_result.items():
        
        ocr = _dict['OCR']
        label = _dict['LABEL']
        path = _dict['PATH']    
        gdesc = _dict['GENERAL_DESC']
        id_ = str(_dict['ID']).strip().split('/')[-1].strip()
        qa_pairs = _dict['pred'].strip().split('[SEP]')

        for idx_qa, query in enumerate( qa_pairs):
            
            if len(query) == 0:
                continue
            query = query.strip()
            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc, query]) 


    df = pd.DataFrame(all_rows, columns=col_names)
    return df

def T5Trainer( args):
    print(f'args: {args}')
    print()

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # execution_mode = 'TEST_GDESC_GEN'
    execution_mode = args.execution_mode.strip()
    save_dir = './results/'

    #
    # The following code snippet loads fine-tuned model based on the execution mode
    # and specifies the name of the output JSON file.
    #
    if execution_mode in ['testGDescGeneration']:

        chkp_id = 'checkpoint-9520/'

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescGeneration/{chkp_id}'
        
        json_name = 'test_memeGDescGeneration_output.json'

    if execution_mode in ['testContextGeneration']:

        chkp_id = 'checkpoint-9940/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainContextGeneration/{chkp_id}'
        
        json_name = 'test_memeContextGeneration_output.json'

    if execution_mode in ['testGDescContextGeneration']:
        
        chkp_id = 'checkpoint-14025/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescContextGeneration/{chkp_id}'
        
        json_name = 'test_memeGDescContextGeneration_output.json'

    if execution_mode in ['testGDescAllQAPairs']:
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescAllQAPairs/{chkp_id}'
        
        json_name = 'test_memeGDescAllQAPairs_output.json'

    if execution_mode in ['testContextAllQAPairs']:
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainContextAllQAPairs/{chkp_id}'
        
        json_name = 'test_memeContextAllQAPairs_output.json'

    if execution_mode in ['testGDescContextAllQAPairs']:
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescContextAllQAPairs/{chkp_id}'

        json_name = 'test_memeGDescContextAllQAPairs_output.json'
    
    if execution_mode in ['testContext2CLS']: 
        chkp_id = 'checkpoint-14025/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainContext2CLS/{chkp_id}'

        json_name = 'test_memContextGen2Classification_output.json'

    if execution_mode in ['testGDescContext2CLS']: 
        chkp_id = 'checkpoint-14025/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescContext2CLS/{chkp_id}'

        json_name = 'test_memeGDescContextGen2Classification_output.json'

    if execution_mode in ['testAllQueryGen']:
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainAllQueryGen/{chkp_id}'

        json_name = 'test_memeAllQueryGen_output.json'

    if execution_mode in ['testGDescAllQueryGen']:
        chkp_id = 'checkpoint-14025/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescAllQueryGen/{chkp_id}'

        json_name = 'test_memeGDescAllQueryGen_output.json'

    if execution_mode in ['testAllQueryGen_1Q1A']: # bs - 85
        chkp_id = 'checkpoint-43450/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainAllQueryGen_1Q1A/{chkp_id}'

        json_name = 'test_memeAllQueryGen_1Q1A_output.json'

    if execution_mode in ['testGDescAllQueryGen_1Q1A']: # bs - 85
        chkp_id = 'checkpoint-43450/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainGDescAllQueryGen_1Q1A/{chkp_id}'

        json_name = 'test_memeGDescAllQueryGen_1Q1A_output.json'



    if execution_mode in ['test_QASummaryGeneration']: # ol - 350
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_train_QASummaryGeneration/{chkp_id}'

        json_name = 'test_QASummaryGeneration_output.json'

    if execution_mode in ['test_QASummaryToQApairsgeneration']: # ol - 350
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_train_QASummaryToQApairsgeneration/{chkp_id}'

        json_name = 'test_QASummaryToQAPairsGeneration_output.json'

    
    if execution_mode in ['test_QASummaryToQuestionsGeneration']: # ol - 350
        chkp_id = 'checkpoint-9350/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_train_QASummaryToQuestionsGeneration/{chkp_id}'

        json_name = 'test_QASummaryToQuestionsGeneration_output.json'
    
    if execution_mode in ['test_QASummaryQuestionsTo1Q1A']: # ol - 350
        chkp_id = 'checkpoint-43450/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_train_QASummaryQuestionsTo1Q1A/{chkp_id}'

        json_name = 'test_QASummaryQuestionsTo1Q1A_output.json'
    
    if execution_mode in ['test_ContextBasedAllQueryGen']: # ol - 30
        chkp_id = 'checkpoint-43450/' 

        if args.img_type in ['vit', 'VIT']:
            model_dir = f'./unit_models/VIT_T5Base_FullFT_trainContextBasedAllQueryGen/{chkp_id}'

        json_name = 'test_ContextBasedAllQueryGen_output.json'
    
    
    #
    #   The following code snippet load the tokenizer, pre-processed image embeddings and the fine-tuned 
    #   model from the saved location.
    #

    print(f'*****   args.execution_mode: {args.execution_mode}')
    print(f'*****   model_dir: {model_dir}')
    time.sleep(5)
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    console.log(f'tokenizer is loaded.')

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
        
    exp_trn_list = [1,2,3,4,5,6,7,8]; exp_val_list = [9]; exp_test_list = [11, 12, 13]
    imp_trn_list = [14,15,16,17,18,19, 20]; imp_val_list =[21]; imp_test_list = [28, 29, 30, 31]
    ben_trn_list = [33, 34, 35, 36, 37, 38, 39]; ben_val_list =  [40]; ben_test_list= [43]

    # # train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list )
    # # eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )

    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        model = T5ForMultimodalGeneration.from_pretrained(model_dir, patch_size=patch_size) 
        print(f'Model is loaded.')

        # check
        if args.img_type == "resnet":
            image_features = np.load('vision_features/resnet.npy')
            image_features = np.expand_dims(image_features, axis=1)
            image_features = image_features.repeat(512, axis=1)
        
        if args.img_type.lower() == "clip":
            image_features = torch.load('vision_features/clip_trainC.pth')
            name_maps = json.load(open('vision_features/name_map_clip_trainC.json'))
            
        
        if args.img_type == "detr":
            image_features = torch.load('vision_features/detr_trainC.pth')
            name_maps = json.load(open('vision_features/name_map_detr_trainC.json'))
            

        if args.img_type in ["vit", 'VIT']:

            image_features = torch.load("vision_features/vit_trainC.pth")
            name_maps = json.load(open('vision_features/name_map_vit_trainC.json'))
            print(f'image_model: vision_features/vit_trainC.pth')
        
                
        #
        #   The following code loads the test data from DatasetA-Regular or
        #   some output JSON file that is produced from other modules.
        #
        
        path = './DatasetA-Regular/DatasetA-Regular.xlsx'
        if execution_mode in ['test_QASummaryToQuestionsGeneration']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        if execution_mode in ['test_QASummaryToQApairsgeneration']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        if execution_mode in ['test_QASummaryGeneration']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        if execution_mode in ['testGDescGeneration']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)
            
        if execution_mode in ['testContextGeneration', 'testGDescContextGeneration']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        if execution_mode in ['testGDescAllQAPairs', 'testContextAllQAPairs', 'testGDescContextAllQAPairs']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)
        
        if execution_mode in ['testContext2CLS', 'testGDescContext2CLS']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        if execution_mode in ['testAllQueryGen']:
            test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
            # test_df = pd.read_excel(path)

        # Independent of excel sheet.
        if 'testGDescAllQueryGen' in execution_mode:
            #
            generate_context_json = './results/test_memeGDescGeneration_output.json'
            test_json = json.load( open(generate_context_json))
            
            all_lines = []
            for k, line in test_json.items():

                pred_memeGDesc = line['pred_memeGDesc'].replace(f'[END]', '')
                label = line['LABEL']
                path = line['PATH']
                ocr = line['OCR']
                all_lines.append( [pred_memeGDesc, ocr, label, path])

            test_df = pd.DataFrame(all_lines, columns=['PRED_GDESC', 'OCR', 'LABEL', 'PATH' ])


        # 1Q1A
        if execution_mode in ['testAllQueryGen_1Q1A']:
            #
            generate_context_json = './results/test_memeAllQueryGen_output.json'
            test_json = json.load( open(generate_context_json))
            
            
            all_lines = []
            for k, line in test_json.items():

                ocr = line['OCR']
                label = line['LABEL']
                path = line['PATH']
                pred_allQ = line['pred_allQueryGen'].replace('[END]', '').strip().split('[SEP]')

                for q_idx, query in enumerate(pred_allQ):

                    id_ = str(q_idx) + '_' + path.strip().split('/')[-1].strip()
                    all_lines.append( [ query, ocr, label, path, id_])

            test_df = pd.DataFrame(all_lines, columns=['PRED_QUERY',  'OCR', 'LABEL', 'PATH', 'ID' ])
            
            # 1/0


        if execution_mode in ['testGDescAllQueryGen_1Q1A']:
            #
            generate_context_json = './results/test_memeGDescAllQueryGen_output.json'
            test_json = json.load( open(generate_context_json))
            
            
            all_lines = []
            for k, line in test_json.items():

                pred_memeGDesc = line['PRED_GDESC']
                ocr = line['OCR']
                label = line['LABEL']
                path = line['PATH']
                pred_allQ = line['pred_gDescAllQueryGen'].replace('[END]', '').strip().split('[SEP]')

                for q_idx, query in enumerate(pred_allQ):

                    id_ = str(q_idx) + '_' + path.strip().split('/')[-1].strip()
                    all_lines.append( [ query, pred_memeGDesc, ocr, label, path, id_])

            test_df = pd.DataFrame(all_lines, columns=['PRED_QUERY', 'PRED_GDESC', 'OCR', 'LABEL', 'PATH', 'ID' ])
            
        
        if execution_mode in ['test_QASummaryQuestionsTo1Q1A']:
            #
            generate_context_json = './results/test_QASummaryToQuestionsGeneration_output.json'
            test_json = json.load( open(generate_context_json))
            
            
            all_lines = []
            for k, line in test_json.items():

                # pred_memeGDesc = line['pred_Query']
                ocr = line['OCR']
                label = line['LABEL']
                path = line['PATH']
                pred_allQ = line['pred_Query'].replace('[END]', '').strip().split('[SEP]')

                for q_idx, query in enumerate(pred_allQ):

                    id_ = str(q_idx) + '_' + path.strip().split('/')[-1].strip()
                    all_lines.append( [ query, ocr, label, path, id_])

            test_df = pd.DataFrame(all_lines, columns=['PRED_QUERY', 'OCR', 'LABEL', 'PATH', 'ID' ])
        

        print(f'test_df is created in {execution_mode} mode.')
    
    ###

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
        
        print(preds.shape)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    # Loading the training_args
    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="none",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            model_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="none",
        )

    # Loading the trainer
    tester = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    
    # generate the rationale for the eval set
    print(f'\n*** Prediction: STARTED: {execution_mode} \n')

    # print( type(eval_set) )
    if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
        
        torch.cuda.empty_cache()
        
        id2info = { }
        total_cases = test_df.shape[0]
        case_processed = 0

        jump = 1
        print(f'Total cases: {total_cases}: {test_df.shape}')
        

        test_df = test_df.reset_index()
        
        while case_processed < total_cases:

            print(f'At: {case_processed}/{test_df.shape[0]}')
            
            temp_df = test_df.loc[case_processed:case_processed, :].reset_index()
            # print(f'temp_df: {temp_df.shape}:\n{temp_df}')
            case_processed += jump
            # if execution_mode == 'TEST_QUESTION_SPECIFIC_ANSWER_GEN':
            #     temp_df = reform_df( temp_df)

            temp_set = CustomDatasetImg( temp_df, tokenizer, name_maps, args.input_len,
                    args.output_len, image_features, execution_mode, args, 'TEST', args.test_le )
            
            predict_results = tester.predict(test_dataset=temp_set, max_length=args.output_len) 
        
            if tester.is_world_process_zero():
                if args.use_generate:
                    preds, targets = predict_results.predictions, predict_results.label_ids
                else:
                    preds = predict_results.predictions[0]
                    targets = predict_results.label_ids
                    preds = preds.argmax(axis=2)
                

                print(f'preds: {preds.shape}')
                preds = tokenizer.batch_decode(
                    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                targets = tokenizer.batch_decode(
                    targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                preds = [pred.strip() for pred in preds]

                # print(f'***************** temp_df: {temp_df.shape}')
                for idx, path_i in enumerate(temp_df['PATH']):
                    
                    image_id = 'rmmhs_' + path_i.strip().split('/')[-1].strip()
                    if execution_mode in ['TEST_memeGDescAllQGenQ2AGen']:
                        image_id = temp_df.loc[0, 'ID'] 
                    
                    if execution_mode in ['testGDescGeneration']:
                        id2info[image_id] = {
                            'pred_memeGDesc': preds[idx], 
                            'exp_memeGDesc': temp_df.loc[0,'GENERAL_DESC'].strip(),
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }

                        # print(temp_df.columns)
                        

                    if execution_mode in ['testContextGeneration']:
                        id2info[image_id] = {
                            'pred_memeContext': preds[idx], 
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    
                    if execution_mode in ['testGDescContextGeneration']:
                        id2info[image_id] = {
                            'pred_memeGDescContext': preds[idx], 
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    
                    if execution_mode in ['testGDescAllQAPairs']:
                        id2info[image_id] = {
                            'pred_memeGDescAllQAPairs': preds[idx].upper(), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    
                    if execution_mode in ['testContextAllQAPairs']:
                        id2info[image_id] = {
                            'pred_memeConextAllQAPairs': preds[idx].upper(), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    
                    if execution_mode in ['testGDescContextAllQAPairs']:
                        id2info[image_id] = {
                            'pred_memeGDescContextAllQAPairs': preds[idx].upper(), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }

                    if execution_mode in ['testContext2CLS']:
                        id2info[image_id] = {
                            'pred_memeContext2Classification': preds[idx].upper(), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    if execution_mode in ['testGDescContext2CLS']:
                        id2info[image_id] = {
                            'pred_memeGdescContext2Classification': preds[idx].upper(), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }

                    if execution_mode in ['testAllQueryGen']:
                        id2info[image_id] = {
                            'pred_allQueryGen': preds[idx], 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                            'PATH': temp_df.loc[0,'PATH'].strip()
                        }
                    
                    if execution_mode in ['testGDescAllQueryGen']:
                        id2info[image_id] = {
                            'pred_gDescAllQueryGen': preds[idx], 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip(),
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                            # 'PRED_QUERY': temp_df.loc[0, 'PRED_QUERY'].strip(),
                            'PRED_GDESC': temp_df.loc[0,'PRED_GDESC'].strip()
                        }
                    
                    # 1Q1A
                    if execution_mode in ['testAllQueryGen_1Q1A', 'testGDescAllQueryGen_1Q1A']:
                        image_id = temp_df.loc[0, 'ID']
                        # print(f'*****: image_id: {image_id}')
                        id2info[image_id] = {
                            'OCR': temp_df.loc[0, 'OCR'].strip(),
                            'PRED_QUERY': temp_df.loc[0, 'PRED_QUERY'].strip(),
                            'pred_ANSWER': preds[idx].strip().replace('[END]', ''), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            
                        }

                    
                    if execution_mode in ['test_QASummaryGeneration']:
                        
                        id2info[image_id] = {
                            'OCR': temp_df.loc[0, 'OCR'].strip(),
                            'QA_SUMMARY': preds[idx],
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            
                        }

                    if execution_mode in ['test_QASummaryToQApairsgeneration']:
                        
                        id2info[image_id] = {
                            'OCR': temp_df.loc[0, 'OCR'].strip(),
                            'QA_PAIRS': preds[idx],
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip(),
                            
                        }

                    if execution_mode in ['test_QASummaryToQuestionsGeneration']:
                        
                        id2info[image_id] = {
                            'pred_Query': preds[idx], 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip(),
                            'OCR': temp_df.loc[0,'OCR'].strip(),
                        }
                    
                    
                    if execution_mode in ['test_QASummaryQuestionsTo1Q1A']:
                        image_id = temp_df.loc[0, 'ID']
                        # print(f'*****: image_id: {image_id}')
                        id2info[image_id] = {
                            'OCR': temp_df.loc[0, 'OCR'].strip(),
                            'PRED_QUERY': temp_df.loc[0, 'PRED_QUERY'].strip(),
                            'pred_ANSWER': preds[idx].strip().replace('[END]', ''), 
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            
                        }




                output_prediction_file = save_dir+ json_name
                with open(output_prediction_file, "w") as writer:
                    writer.write(json.dumps(id2info, indent=4))
                
                print(f'*****   Outputs are saved at: {output_prediction_file}')
                # 1/0
        

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

    # if args.img_type is not None:
    #     problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
    #     dataframe = {'problems':problems, 'qids':qids, 'name_maps': name_maps, 'image_features': image_features}
    # else:
    #     problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
    #     dataframe = {'problems':problems, 'qids':qids}

    T5Trainer( args = args)
