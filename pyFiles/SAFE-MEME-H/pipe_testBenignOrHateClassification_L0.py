import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
# from model import T5ForMultimodalGeneration
from util_L0forHateOrBenignDetectionTrain import CustomDatasetImg
# from util_trainCombinedCardTrain import CustomDatasetImg


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

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = f"./logFiles/test_benignOrNOT_fullFT_noCards.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)
logger.info(f"START for main.py")



'''
***** ATTENTION ****
The file 'pipe_testUnitModelsOutputGen' is used for inference of DatasetAB-Regular and NO training.
For MANTIS, kindly run the commands for testAllQueryGen, and testAllQueryGen_1Q1A sequentially.  

CUDA_VISIBLE_DEVICES=2 python3 pipe_testBenignOrHateClassification_L0.py         --execution_mode  test_benignOrNOT_partialFT_singleCard         --output_len 10         --img_type vit         --user_msg rationale         --epoch 1         --lr 5e-5         --use_caption         --use_generate         --prompt_format QCM-E         --output_dir experiments         --final_eval

'''
category2save_dict = {

    'ISLAM': 'ISALM',
    'WOMEN_IN_GENERAL': 'WOMEN',
    'MEN_IN_GENERAL': 'MEN',
    'THE_BLACK_COMMUNITY': 'BLACK',
    'OTHERS': 'OTHERS',
    'THE_LGBTQ_COMMUNITY': 'LGBTQ', 
    'THE_JEWISH_COMMUNITY': 'JEWS',
    'THE_WHITE_COMMUNITY': 'WHITE',
    'THE_PEOPLE_WITH_DISABILITY': 'DISABILITY',
    'THE_IMMIGRANT_PEOPLE': 'IMMIGRANT',    
    'GENERAL': 'GENERAL',

    'NO_BODY': 'NO_BODY',
    

    'THE LGBTQ COMMUNITY': 'LGBTQ', 
    'WOMEN IN GENERAL': 'WOMEN',
    'MEN IN GENERAL': 'MEN',
    'NO BODY': 'NOBODY',
    'THE BLACK COMMUNITY': 'BLACK',
    'THE JEWISH COMMUNITY': 'JEWS',
    'THE WHITE COMMUNITY': 'WHITE',
    'THE PEOPLE WITH DISABILITY': 'DISABILITY',
    'THE IMMIGRANT PEOPLE': 'IMMIGRANT',
    'CHRISTIANITY': 'CHRISTIANITY',
    'HINDUISM': 'HINDUISM',
    'NO BODY': 'NO_BODY',
    'NOBODY': 'NO_BODY',
    

}

def parse_args():

    mode_list = [

        'test_benignOrNOT_fullFT_noCards', # ol - 10 # no_cardmodel.py # model_noCards - GENERAL
        'test_benignOrNOT_partialFT_singleCard', # 10 # single-card # model_RMMHS - GENERAL

    ]


    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_root', type=str, default='data')
    # parser.add_argument('--model_variation', type=str, default='QCM-A', choices=['ZERO', 'ONE'])
    parser.add_argument('--category', type=str, default='QCM-A', choices=['OTHERS', 'GENERAL', 'NO_BODY', 'ISLAM', 'THE_PEOPLE_WITH_DISABILITY', 'THE_IMMIGRANT_PEOPLE', 'THE_WHITE_COMMUNITY', 'THE_BLACK_COMMUNITY', 'THE_JEWISH_COMMUNITY', 'THE_LGBTQ_COMMUNITY', 'WOMEN_IN_GENERAL', 'MEN_IN_GENERAL'])
    parser.add_argument('--execution_mode', type=str, choices=mode_list)
    
    # parser.add_argument('--full_FT', type=str, choices=['YES', 'NO'])
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

        logger.info(f'idx_dataset: {idx_dataset}: {df.shape}')
        
    return df

def get_filtered_dataframe(df, category):
    
    logger.info(f'get_filtered_dataframe: category: {category.upper()}')
    logger.info(f'Initially: df,shape: {df.shape}')
    df['TARGET'] = df['TARGET'].str.upper()

    if category.strip().upper() not in [ 'OTHERS', 'HINDUISM', 'CHRISTIANITY', 'GENERAL' ]:
        df = df[ df['TARGET'] == category.strip().upper() ]
        logger.info(f'df.shape: {df.shape}')
    
        logger.info(f'After df.shape: {df.shape}')
    
    if category.strip().upper() in [ 'OTHERS' ]:
        
        df_1 = df[ df['TARGET'] == 'OTHERS' ]
        df_2 = df[ df['TARGET'] == 'CHRISTIANITY' ]
        df_3 = df[ df['TARGET'] == 'HINDUISM' ]

        df = pd.concat( [df_1, df_2, df_3] )
        logger.info(f'After df.shape: {df.shape}')


    
    df = df.reset_index()
    logger.info(f'Finally: df.shape: {df.shape}')
    
    return df


def reform_df(df):

    all_rows = []    
    col_names = [ 'OCR', 'LABEL', 'ID', 'PATH', 'GENERAL_DESC', 'QUERY', 'ANSWER', 'CONTEXT', 'TARGET']
             
    for idx in range(df.shape[0]):
        
        ocr = df.loc[idx, 'OCR']
        label = df.loc[idx, 'LABEL']
        path = df.loc[idx, 'PATH']    
        gdesc = df.loc[idx, 'GENERAL_DESC']
        id_ = df.loc[idx, 'PATH'].strip().split('/')[-1].strip()
        target = df.loc[idx, 'TARGET'].strip()

        context = df.loc[idx, 'QA'].strip().split('\n')[-1].split('#')[-1]
        qa_pairs = df.loc[idx, 'QA'].strip().split('\n')[1:-1]
        for idx_qa, qa_pair in enumerate( qa_pairs):
            if len(qa_pair) == 0:
                continue
            query, answer = qa_pair.split('#')
            query = query.strip()
            answer = answer.strip()
            all_rows.append( [ocr, label, f'{idx_qa}_{id_}', path, gdesc,  query, answer, context, target]) 


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
        

def T5Trainer( args):
    print(f'args: {args}')
    print()

    img_shape = { "resnet": (512, 2048), "clip": (49, 1024), "detr": (100, 256), "vit": (145, 1024), }

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
    if execution_mode in ['test_benignOrNOT_fullFT_noCards', ]:

        json_path = f'./results/test_fullFT_noCard_gDescGeneration.json'
        # json_path = f'./results/test_partialFT_singleCard_gDescGeneration.json'

        testCases = json.load( open(json_path) )

        ocr_list = []
        path_list = []
        label_list = []
        gdesc_list = []
        
        for k, _dict in testCases.items():

            ocr_list.append(_dict['OCR'].strip())
            path_list.append(_dict['PATH'].strip())
            label_list.append(_dict['LABEL'].strip())
            gdesc_list.append(_dict['pred_memeGDesc'].strip())

        data = { 'OCR': ocr_list, 'PATH': path_list, 'LABEL': label_list, 'GENERAL_DESC': gdesc_list}
        test_df = pd.DataFrame(data)

        logger.info(f'*** test_df: {test_df.shape}')
        logger.info(test_df)

        from model_noCardTrain import T5ForMultimodalGeneration
        logger.info(f'*** model_def: model_noCardTrain')  
        
        chkp_id = 'checkpoint-976/'
        model_dir = f'./unit_models/VIT_T5Base_fullFT_train_benignOrNOT_fullFT_noCards_category_GENERAL/{chkp_id}'
    
        json_name = 'test_benignOrNOT_fullFT_noCards.json'
        
    if execution_mode in ['test_benignOrNOT_partialFT_singleCard', ]:

        # model_singleCard
        # ***** CARD_PATH: ./trained_cards/card_train_benignOrNOT_partialFT_singleCard_GENERAL
        # ***** save_dir: unit_models/VIT_T5Base_partialFT_train_benignOrNOT_partialFT_singleCard_category_GENERAL

        # json_path = f'./results/test_fullFT_noCard_gDescGeneration.json'
        # json_path = f'./results/test_partialFT_singleCard_gDescGeneration.json'
        


        # json_path = f'./results/test_fullFT_categoryCard_gDescGeneration_v0.json'
        # json_name = f'test_L0_usingfullFTCategoryCardbasedGDesc_v0.json'
        
        # json_path = f'./results/test_partialFT_categoryCard_gDescGeneration_v0.json'
        # json_name = f'test_L0_usingpartialFTCategoryCardbasedGDesc_v0.json'
        
        # json_path = f'./results/test_fullFT_categoryCard_gDescGeneration_v1.json'
        # json_name = f'test_L0_usingfullFTCategoryCardbasedGDesc_v1.json'
        
        # json_path = f'./results/test_partialFT_categoryCard_gDescGeneration_v1.json'
        # json_name = f'test_L0_usingpartialFTCategoryCardbasedGDesc_v1.json'
        
        json_path = f'./results/test_partialFT_categoryCard_gDescGeneration_v1.json'
        json_name = f'test_L0_usingpartialFTCategoryCardbasedGDesc_v1.json'
        


        testCases = json.load( open(json_path) )

        ocr_list = []
        path_list = []
        label_list = []
        gdesc_list = []
        
        for k, _dict in testCases.items():

            ocr_list.append(_dict['OCR'].strip())
            path_list.append(_dict['PATH'].strip())
            label_list.append(_dict['LABEL'].strip())
            gdesc_list.append(_dict['pred_memeGDesc'].strip())

        data = { 'OCR': ocr_list, 'PATH': path_list, 'LABEL': label_list, 'GENERAL_DESC': gdesc_list}
        test_df = pd.DataFrame(data)

        logger.info(f'*** test_df: {test_df.shape}')
        logger.info(test_df)

        from model_singleCard import T5ForMultimodalGeneration
        logger.info(f'*** model_def: model_noCardTrain')  
        
        chkp_id = 'checkpoint-12200/'
        model_dir = f'./unit_models/VIT_T5Base_partialFT_train_benignOrNOT_partialFT_singleCard_category_GENERAL/{chkp_id}'

        # json_name = 'test_benignOrNOT_partialFT_singleCard_basedOn_test_fullFT_noCard_gDescGeneration.json'    
        
        
    
    logger.info(f'**** json_name: {json_name}')

    patch_size = img_shape[args.img_type]
    model = T5ForMultimodalGeneration.from_pretrained(model_dir, patch_size=patch_size)
    logger.info(f'*****   model loaded: {args.model}: VIT')
    
    image_features = torch.load("vision_features/vit_trainC.pth")
    name_maps = json.load(open('vision_features/name_map_vit_trainC.json'))
    print(f'image_model: vision_features/vit_trainC.pth')


    #
    #   The following code snippet load the tokenizer, pre-processed image embeddings and the fine-tuned 
    #   model from the saved location.
    #

    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    console.log(f'tokenizer is loaded.')

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
        
    # exp_trn_list = [1,2,3,4,5,6,7,8]; exp_val_list = [9]; 
    exp_test_list = [11, 12, 13]
    # imp_trn_list = [14,15,16,17,18,19, 20]; imp_val_list =[21]; 
    imp_test_list = [28, 29, 30, 31]
    # ben_trn_list = [33, 34, 35, 36, 37, 38, 39]; ben_val_list =  [40]; 
    ben_test_list= [43]

    # if execution_mode in []:
    #     test_df = get_df(exp_test_list + imp_test_list + ben_test_list)
        
    # print(f'test_df is created in {execution_mode} mode.')

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
                        

                    if execution_mode in [
                        'test_benignOrNOT_fullFT_noCards', 
                        'test_benignOrNOT_partialFT_singleCard'
                    ]:
                        
                        # print(f'*****: image_id: {image_id}')
                        id2info[image_id] = {
                            'OCR': temp_df.loc[0, 'OCR'].strip(),
                            'pred_label': preds[idx],
                            'pred_gdesc': temp_df.loc[0,'GENERAL_DESC'].strip(),
                            'LABEL': temp_df.loc[0,'LABEL'].strip().upper(),
                            'PATH': temp_df.loc[0,'PATH'].strip(),
                            
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
