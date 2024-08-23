# create venv, Python: 3.10.12 (recommend) / 3.12.3 have been tested
virtualenv -p python3.10.12 venv

# activate
source venv/bin/activate

# install
pip install transformers vllm accelerate # specific packages  
pip install -r requirements.txt # requirements file  

# useful tutorials
https://github.com/THUDM/GLM-4/blob/main/basic_demo/README_en.md
https://github.com/THUDM/GLM-4/blob/main/finetune_demo/README_en.md