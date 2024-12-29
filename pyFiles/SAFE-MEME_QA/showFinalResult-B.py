import json
from pprint import pprint

def get_key2informations(all_data):

    _dict = {}
    for key, data in all_data.items():

        img_id = '_'.join( key.split('_')[1:] )
        information = f"Question: {data['PRED_QUERY']} \nAnswer: {data['pred_ANSWER']}"
        try:
            _list = _dict[img_id]
            _list.append( information )
            _dict[img_id] = _list
        except:
            _dict[img_id] = [ f'\n# {img_id}\nInput text: ' + data['OCR'], f"Original label:  {data['LABEL']}\n",information]

    # pprint(_dict)
    return _dict




if __name__ == '__main__':
    
    path = f'./results/testConfounder_memeAllQueryGen_1Q1A_output.json'
    all_data = json.load( open(path))

    all_data = get_key2informations(all_data)
    
    for key, data in all_data.items():
        data = '\n'.join(data) + '\n'
        with open('./results/formatted_outputsB.txt', 'a') as f:
            f.write(data)

            print(data)
