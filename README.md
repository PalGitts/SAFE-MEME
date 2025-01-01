# SAFE-MEME
SAFE-MEME is a novel framework designed for the detection of fine-grained hate speech in memes. The framework consists of two distinct variants: (a) SAFE-MEME-QA, which employs a Q&amp;A approach, and (b) SAFE-MEME-H, which utilizes hierarchical classification to categorize memes into one of three classes: explicit, implicit, or benign.



General Instruction:

* Create a conda environment:  conda create --name env_X
* Activate the environment:	conda activate env_X
*	Please install amazon-science/mm-cot: https://github.com/amazon-science/mm-cot
* Run pip install -r requirements.txt

*	Chnage directory to mm-cot

* Add or replace the following file (folder) in mm-cot folder,
		** timm, vision_features, unit_models (can be trained from scratch too)
		** use: https://drive.google.com/drive/folders/1PWESNhZIDa1YL6aipVyrl_Pwo4LYo1Zu?usp=sharing

* Please put the all the .py files from pyFiles folder in mm-cot folder.
* Create a folder named, 'results' and 'resultsConf'
