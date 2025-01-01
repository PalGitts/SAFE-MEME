import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration

# from model_RMMHS import T5ForMultimodalGeneration
# # from model_noCards import T5ForMultimodalGeneration


from util_L0forHateOrBenignDetectionTrain import img_shape, CustomDatasetImg
# from utils_prompt import *
# from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
import evaluate
import time


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = f"./logFiles/train_benignOrNOT_partialFT_singleCard.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)
logger.info(f"START for main.py")

'''
***** ATTENTION ***
The purpose of the main_trainUnitModelsForEachCategory.py is to finetune the category wise unit modules. 
It uses pre-trained weights of mm-cot-base-rationale.  
Train the models using the commands as per the mode.

CUDA_VISIBLE_DEVICES=1 python3 main_trainBenignOrHateClassification_L0.py    --category GENERAL     --output_len 10     --execution_mode train_benignOrNOT_partialFT_singleCard    --img_type vit       --user_msg rationale     --epoch 50 --lr 5e-5 --use_caption --use_generate --prompt_format QCM-E     --output_dir experiments     --final_eval

'''
import pandas as pd


category2save_dict = {

    'ISLAM': 'ISALM',
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
    'THE IMMIGRANT PEOPLE': 'IMMIGRANT',
    'CHRISTIANITY': 'CHRISTIANITY',
    'HINDUISM': 'HINDUISM',
    'NO BODY': 'NO_BODY',
    'NOBODY': 'NO_BODY',
    

}


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # List of valid modes
    mode_list = [        
        'train_benignOrNOT_fullFT_noCards', # ol - 10 # no_cardmodel.py # model_noCards - GENERAL
        'train_benignOrNOT_partialFT_singleCard', # 10 # single-card # model_RMMHS - GENERAL
    ]

    # parser.add_argument('--data_root', type=str, default='data')
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
        








def T5Trainer(args):
    logger.info(f'args: {args}')


    SAVE_EXT = category2save_dict[args.category.strip().upper()]    
    execution_mode = args.execution_mode.strip()
    
    logger.info(f'*** execution_mode: {execution_mode}')
    
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
            
        if args.img_type == "vit":
            image_features = torch.load("vision_features/vit_trainC.pth")
            name_maps = json.load(open('vision_features/name_map_vit_trainC.json'))
            
            logger.info(f'image_model: vision_features/vit_trainC.pth')
            
        

    # import inspect
    # methods = inspect.getmembers(T5ForMultimodalGeneration, predicate=inspect.isfunction)

    # for name, method in methods:
    #     logger.info(f"In T5ForMultimodalGeneration: Method name: {name}")

    if args.execution_mode in ['train_benignOrNOT_partialFT_singleCard']:

        # ***** CARD_PATH: ./trained_cards/card_train_benignOrNOT_partialFT_singleCard_GENERAL
        # ***** save_dir: unit_models/VIT_T5Base_partialFT_train_benignOrNOT_partialFT_singleCard_category_GENERAL

        
        from model_singleCard import T5ForMultimodalGeneration
        
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)
        logger.info(f'*****   model loaded: {args.model}: VIT')

        CARD_PATH = f'./trained_cards/card_{args.execution_mode}_{SAVE_EXT}'
        logger.info(f'***** CARD_PATH: {CARD_PATH}')
        
        save_dir = f'unit_models/{args.img_type.upper()}_T5Base_partialFT_{execution_mode}_category_{args.category}'
        logger.info(f'***** save_dir: {save_dir}')
        
    
        for name, param in model.named_parameters():
            if 'category_card' in name:
                param.requires_grad = True
                print(f'*** {name}: TRUE')
            else:
                param.requires_grad = False
            
            logger.info(f'{name}: {param.requires_grad}')


    if args.execution_mode in ['train_benignOrNOT_fullFT_noCards']:
        from model_noCardTrain import T5ForMultimodalGeneration
        model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)
        logger.info(f'*****   model loaded: {args.model}: VIT')
    
        save_dir = f'unit_models/{args.img_type.upper()}_T5Base_fullFT_{execution_mode}_category_{args.category}'
        logger.info(f'***** save_dir: {save_dir}')

        for name, param in model.named_parameters():
            param.requires_grad = True
            print(f'****** {name}: TRUE')
            logger.info(f'{name}: {param.requires_grad}')



    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    logger.info(f'args.evaluate_dir: {args.evaluate_dir}')
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    logger.info(f'*****   model: {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    exp_trn_list = [1,2,3,4,5,6,7,8]; exp_val_list = [9]; exp_test_list = [11, 12, 13]
    imp_trn_list = [14,15,16,17,18,19, 20]; imp_val_list =[21]; imp_test_list = [28, 29, 30, 31]
    ben_trn_list = [33, 34, 35, 36, 37, 38, 39]; ben_val_list =  [40]; ben_test_list= [43]
    
    
    

    ###

    logger.info(f'*****   save_dir: {save_dir}')
    time.sleep(2)    
    # 1/0
    
            
        
    #
    train_df = get_df( exp_trn_list + imp_trn_list + ben_trn_list + ben_trn_list)
    # train_df['TARGET'] = train_df['TARGET'].str.upper()
    
    logger.info(f'**** All categories: {set( list(train_df["LABEL"])) }')
    logger.info(f'**** train_df: { train_df.shape }')
    logger.info(f'**** train_df: { train_df.head()}')
    # raise Exception()

    if execution_mode in [ 'train_helperModule1Q1ACategorySpecific', 'train_helperModule1Q1ACategorySpecific_noCards' ]:
        train_df = reform_df(train_df)
        logger.info(f'****** After reform_df: train_df: { train_df.shape }')
    

    args.category = args.category.replace(f'_', ' ').strip().upper()
    logger.info(f'args.category: {args.category}')
    train_df = get_filtered_dataframe(train_df, args.category.upper().strip())

    # logger.info(f'train_df: {train_df.shape[0]}')
    # logger.info(f'train_df\n: {train_df.shape}')
    # logger.info(f'train_df\n: { set(train_df["TARGET"])}')
    
    eval_df = get_df( exp_val_list + imp_val_list + ben_val_list )
    # eval_df['TARGET'] = eval_df['TARGET'].str.upper()


    if execution_mode in [ 'train_helperModule1Q1ACategorySpecific', 'train_helperModule1Q1ACategorySpecific_noCards' ]:
        eval_df = reform_df( eval_df )
        logger.info(f'**** After reform_df: eval_df: { eval_df.shape }')

    eval_df = get_filtered_dataframe(eval_df, args.category.upper().strip())

    logger.info(f'*** eval_df: {eval_df.shape}')
    logger.info(f'*** eval_df: { eval_df.head()}')
    logger.info(f'eval_df: { set(eval_df["LABEL"])}')

    # raise Exception()
    #
    train_set = CustomDatasetImg(
        train_df,
        tokenizer,
        name_maps,
        args.input_len,
        args.output_len,
        image_features,
        execution_mode,
        args,
        args.category.strip().upper(),
        'TRAIN',
        args.eval_le
    )

    logger.info(f'*** Training set is created.')
    
    # raise Exception()


    eval_set = CustomDatasetImg(
        eval_df,
        tokenizer,
        name_maps,
        args.input_len,
        args.output_len,
        image_features,
        execution_mode,
        args,
        args.category.strip().upper(),
        'VAL',
        args.eval_le
    )
    logger.info(f'*** Evaluation set is created.')
    
    # logger.info(f'*** card_path: {CARD_PATH}')
    # logger.info(f'*** save_dir: {save_dir}')
    
    # raise Exception()

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    # logger.info("model parameters: ", model.num_parameters())
    
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
    logger.info(f'*** Flag: rouge')
    
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
    logger.info(f'args.final_eval: {args.final_eval}')
    
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

    logger.info(f'*** Initializing Seq2SeqTrainer.')
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )

    logger.info(f'*** Training: START: execution_mode: {execution_mode}')
    trainer.train()
    logger.info(f'*** Training: END: execution_mode: {execution_mode}')
    
    
    if args.execution_mode in ['train_benignOrNOT_partialFT_singleCard']:
        trainer.save_model(save_dir)
        model.save_categoryCard(CARD_PATH)
        logger.info(f'SAVED: CARD_PATH: {CARD_PATH}: {args.execution_mode}')
        logger.info(f'SAVED: save_dir: {save_dir}: {args.execution_mode}')


    if args.execution_mode in ['train_benignOrNOT_fullFT_noCards']:
        trainer.save_model(save_dir)
        logger.info(f'SAVED: save_dir: {save_dir}: {args.execution_mode}')


    # model.save_categoryCard(CARD_PATH)


if __name__ == '__main__':

    # training logger to log training progress

    logger.info(f'Time: { time.strftime("%H:%M:%S", time.localtime()) }')
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    
    

    args = parse_args()
    
    # logger.info("args",args)
    # logger.info('====Input Arguments====')
    # print(json.dumps(vars(args), indent=2, sort_keys=False))
    logger.info(f'In main_trainUnitModelsForEachCategory.py')
    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)


    T5Trainer(args = args)

