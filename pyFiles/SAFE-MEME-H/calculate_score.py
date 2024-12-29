from collections import Counter
from sklearn.metrics import hamming_loss
from sklearn.metrics import matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def get_stat( class_id, labels, preds ):
    
    print(f'class_id: {class_id}')
    # print(f'labels: {len(labels)}')
    # print(f'preds:  {len(preds)}')

    mod_labels = []
    mod_preds  = []
    
    for idx in range(len(labels)):
        
        if labels[idx] == class_id:
            mod_labels.append(1)
        else:
            mod_labels.append(0)
        
        if preds[idx] == class_id:
            mod_preds.append(1)
        else:
            mod_preds.append(0)
    
    # print(f'mod_preds:  {mod_preds}')
    # print(f'mod_labels: {mod_labels}')
    
    print(f'\tprecision_score: {precision_score( mod_labels, mod_preds)}')
    print(f'\trecall_score: {recall_score( mod_labels, mod_preds)}')
    print(f'\tf1: {f1_score( mod_labels, mod_preds)}')
    print(f'\taccuracy_score: {accuracy_score( mod_labels, mod_preds)}')

# preds = [ 1, 1, 1, 2, 2]
# labels = [ 0, 0, 1, 2, 1]

# print(Counter([ i[2] for i in triplet_list]))
# print(Counter(test_use['labels']))


def map_label2class2(text):

    if 'EXP' in text.strip().upper():
        return 2
    
    if 'IMP' in text.strip().upper():
        return 1

    return 0



import json
'''
    "rmmhs_image3308.jpg": {
        "seq_id": 8,
        "pred_QA": "Does the provided context contain any hate? # Yes [SEP] Who is the target? # Indians, Indian Socialism [SEP] What is the reason for being targeted? # For being an Indian [SEP] Is there any explicit hate is visible in the context? # Yes. The context has terms like 'Indian, a socialist','smellier, a mistress, a homosexual, and a terrorist lover' [SEP] What kind of hate is expressed here? # Explicit hate[END]",
        "pred_gdesc": "The image shows two men, one of them wearing a white suit and a white jacket, standing on a stage. They are both dressed in black and appear to be engaged in a conversation or observing something. [END]",
        "OCR": "there's a joke in here somewhere? \na indian, a socialist, a smeller, a mistress, a homosexual, and a terrorist lover walk into a bar..",
        "LABEL": "EXPLICIT HATE",
        "PATH": "rmmhs_v2_images/image3308.jpg"
    },
'''

def when_1Q1A(json_dict):
    
    exp_list = []
    pred_list = []
    for key, val in json_dict.items():
        
        exp = map_label2class2( val['label'])
        exp_list.append(exp)
        
        pred_answer = val['pred'].strip()
        for line in pred_answer.split('[SEP]'):
            try:
                query, reply = line.split('#')

                # print(f'For: {reply}')
                pred_class = 0
                if 'What kind of'.upper() in query.upper():
                    # print(f'***')
                    pred_class = map_label2class2(reply)
                    break


            except Exception as e:
                print(f'{e}: {line}')
                pred_class = 0
                print()
                
        pred_list.append(pred_class)
    
    return exp_list, pred_list

def when_allQAGen(json_dict):
    
    exp_list = []
    pred_list = []

    for key, _dict in json_dict.items():

        last_ans = _dict['pred_QA'].strip().split(f'#')[-1]
        pred_class = map_label2class2(last_ans)
        exp_class = map_label2class2(_dict['LABEL'].strip())

        exp_list.append(exp_class)
        pred_list.append(pred_class)
    
    return exp_list, pred_list

def when_benignOrNOT(path):
    
    exp_list = []
    pred_list = []
    total_correct = 0

    json_dict = json.load( open(path))
    print(f'\nBenign: ')

    for key, _dict in json_dict.items():

        exp_class  = _dict['LABEL'].strip().upper()
        
        if 'IMP' in exp_class or 'EXP' in exp_class:
            exp_class = 1
        else:
            exp_class = 0

        pred_class = 0
        if 'HATE' in _dict['pred_label'].strip().upper():
            pred_class = 1
        
        exp_list.append( exp_class )
        pred_list.append( pred_class )

        if exp_list == pred_list:
            total_correct += 1

    print(f'TOtal correct: {total_correct}/{len(exp_list)}')
    
    # print(f'Hamming loss: {hamming_loss(pred_list, exp_list)}')
    print(f'Macro precision: {precision_score(exp_list, pred_list)}')
    print(f'Macro recall: {recall_score(exp_list, pred_list)}')
    print(f'Macro f1: {f1_score(exp_list, pred_list)}')
    print( )

def when_expHate(path_L0, path_L1):

    '''
    "rmmhs_image3238.jpg": {
        "OCR": "feed the greed. hello i'm sir trevor chinn. i control tom watson by money. it's what zionists do",
        "PRED_LABEL": "Explicit [END]",
        "LABEL": "EXPLICIT HATE"
    },
    '''
    
    exp_list = []
    pred_list = []
    total_correct = 0
    

    hateOrBen_dict = json.load( open(path_L0))
    json_dict = json.load( open(path_L1))

    for key, _dict in hateOrBen_dict.items():

        if 'EXP' in _dict['LABEL'].strip().upper() and 'HATE' not in _dict['pred_label'].strip().upper():
            exp_list.append(1)
            pred_list.append(0)

    print(f'In when_expHate: len(exp_list): {len(exp_list)} and len(pred_list): { len(pred_list)}')
    for key, _dict in json_dict.items():

        exp_class  = _dict['LABEL'].strip().upper()
        
        if 'BENIGN' in exp_class:# or 'IMPLICIT' in exp_class:
            continue

        # print(f'Exp: {_dict["LABEL"]} and P: {_dict["PRED_LABEL"]}')
        # if 'IMP' in exp_class: 
        if 'EXP' in exp_class: 
        
            exp_class = 1
        else:
            exp_class = 0

        pred_class = 0
        
        # if 'IMP' in _dict['PRED_LABEL'].strip().upper():
        if 'EXP' in _dict['PRED_LABEL'].strip().upper():
        
            pred_class = 1
        
        exp_list.append( exp_class )
        pred_list.append( pred_class )

        if exp_list == pred_list:
            total_correct += 1

    print(f'TOtal correct: {total_correct}/{len(exp_list)}')
    
    # print(f'Hamming loss: {hamming_loss(pred_list, exp_list)}')
    print(f'Macro precision: {precision_score(exp_list, pred_list)}')
    print(f'Macro recall: {recall_score(exp_list, pred_list)}')
    print(f'Macro f1: {f1_score(exp_list, pred_list)}')
    print( )

def when_impHate(path_L0, path_L1):

    '''
    "rmmhs_image3238.jpg": {
        "OCR": "feed the greed. hello i'm sir trevor chinn. i control tom watson by money. it's what zionists do",
        "PRED_LABEL": "Explicit [END]",
        "LABEL": "EXPLICIT HATE"
    },
    '''
    
    exp_list = []
    pred_list = []
    total_correct = 0
    
    hateOrBen_dict = json.load( open(path_L0))
    json_dict = json.load( open(path_L1))

    for key, _dict in hateOrBen_dict.items():

        if 'IMP' in _dict['LABEL'].strip().upper() and 'HATE' not in _dict['pred_label'].strip().upper():
            exp_list.append(1)
            pred_list.append(0)

    print(f'In when_impHate: len(exp_list): {len(exp_list)} and len(pred_list): {len(pred_list)}')

    for key, _dict in json_dict.items():

        exp_class  = _dict['LABEL'].strip().upper()
        
        if 'BENIGN' in exp_class:# or 'IMPLICIT' in exp_class:
            continue

        # print(f'Exp: {_dict["LABEL"]} and P: {_dict["PRED_LABEL"]}')
        if 'IMP' in exp_class: 
        # if 'EXP' in exp_class: 
        
            exp_class = 1
        else:
            exp_class = 0

        pred_class = 0
        
        if 'IMP' in _dict['PRED_LABEL'].strip().upper():
        # if 'EXP' in _dict['PRED_LABEL'].strip().upper():
        
            pred_class = 1
        
        exp_list.append( exp_class )
        pred_list.append( pred_class )

        if exp_list == pred_list:
            total_correct += 1

    print(f'TOtal correct: {total_correct}/{len(exp_list)}')
    
    # print(f'Hamming loss: {hamming_loss(pred_list, exp_list)}')
    print(f'Macro precision: {precision_score(exp_list, pred_list)}')
    print(f'Macro recall: {recall_score(exp_list, pred_list)}')
    print(f'Macro f1: {f1_score(exp_list, pred_list)}')
    print( )
    


if __name__ == '__main__':


    # path = f'./test_withGDescL1forImpOrExpDetectionInference_usingBenignOrNOT_fullFT_noCard_gDescGeneration.json'
    # pathforBenignOrHate = f'./test_benignOrNOT_partialFT_singleCard_basedOn_benignOrNOT_partialFT_singleCard_basedOn_fullFT_noCard_gDescGeneration.json' # Gdesc based on full FT

    # path = f'./test_withGDescL1forImpOrExpDetectionInference_usingBenignOrNOT_partialFT_singleCard_gDescGeneration.json'
    # pathforBenignOrHate = f'./test_benignOrNOT_partialFT_singleCard_basedOn_partialFT_singleCard_gDescGeneration.json'

    # path_L0 = f'./L0/test_L0_usingfullFTCategoryCardbasedGDesc_v0.json'
    # path_L1 = f'./L1/test_L1_usingfullFTCategoryCardbasedGDesc_v0.json'

    path_L0 = f'./L0/test_L0_usingpartialFTCategoryCardbasedGDesc_v0.json'
    path_L1 = f'./L1/test_L1_usingpartialFTCategoryCardbasedGDesc_v0.json'

    # path_L0 = f'./L0/test_L0_usingfullFTCategoryCardbasedGDesc_v1.json'
    # path_L1 = f'./L1/test_L1_usingfullFTCategoryCardbasedGDesc_v1.json'

    # path_L0 = f'./L0/test_L0_usingpartialFTCategoryCardbasedGDesc_v1.json'
    # path_L1 = f'./L1/test_L1_usingpartialFTCategoryCardbasedGDesc_v1.json'

    print()
    print(path_L0)
    print(path_L1)

    # json_dict = json.load( open(path))

    # when_benignOrNOT(json_dict) # Just run this
    print(f'on Exp\n')
    try:
        when_expHate(path_L0, path_L1)
    except:
        pass

    print(f'on Imp\n')
    try:
        when_impHate(path_L0, path_L1)
    except:
        pass

    try:
        when_benignOrNOT(path_L0)
    except:
        pass

   