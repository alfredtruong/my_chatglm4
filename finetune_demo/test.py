# https://github.com/THUDM/GLM-4/blob/main/finetune_demo/README_en.md

# https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/README_en.md#dataset-format-example
# https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/README_en.md
'''
## Dataset format example

Here we take the AdvertiseGen data set as an example,
You can download it
from [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing)
Or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) download the AdvertiseGen data set.
Place the decompressed AdvertiseGen directory in the `data` directory and convert it into the following format data set
yourself.

> Please note that the verification set is added to the current fine-tuning code. Therefore, for a complete set of
> fine-tuning data sets, the training data set and the verification data set must be included, and the test data set
> does
> not need to be filled in. Or directly use the validation data set instead.

```
{"messages": [{"role": "user", "content": "Type#skirt*skirt length#skirt"}, {"role": "assistant", "content": "This is versatile Fashionable fairy skirt, the overall design is very elegant and casual. Every girl can instantly turn into a fairy after wearing it. The material is very light and breathable, making it very comfortable to wear in summer."} ]}
```
'''
#%%
##########################################
# build train.json and dev.json
##########################################
from datasets import load_dataset

ds = load_dataset("shibing624/AdvertiseGen")
'''
DatasetDict({
    train: Dataset({
        features: ['content', 'summary'],
        num_rows: 114599
    })
    validation: Dataset({
        features: ['content', 'summary'],
        num_rows: 1070
    })
})
'''

import json

def convert_ds(ds,
    partition: str = 'train',
    outfile: str = 'train.json'
    ):
    with open(outfile, 'w', encoding='utf-8') as f:
        '''
        {'content': '类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤', 'summary': '宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。毕竟好穿时尚，谁都能穿出腿长2米的效果宽松的裤腿，当然是遮肉小能手啊。上身随性自然不拘束，面料亲肤舒适贴身体验感棒棒哒。系带部分增加设计看点，还让单品的设计感更强。腿部线条若隐若现的，性感撩人。颜色敲温柔的，与裤子本身所呈现的风格有点反差萌。'}
        '''
        for data in ds[partition]:
            content = data['content']
            summary = data['summary']

            conversation = {
                "messages": [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": summary}
                ]
            }

            json.dump(conversation, f, ensure_ascii=False)
            f.write('\n')

convert_ds(ds,'train','train.jsonl')
convert_ds(ds,'validation','dev.jsonl')

##########################################
# apply model
##########################################
'''
python inference.py finetune_demo/output/checkpoint-1500
THUDM/GLM-4
python inference.py THUDM/glm-4-9b-chat


CUDA_VISIBLE_DEVICES=0 python inference.py output/checkpoint-1500/ --prompt "你是谁？"

'''

#%%
##########################################
# force compute onto particular gpu
##########################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#%%
##########################################
# Prepare a model for training with a PEFT method such as LoRA by wrapping the base model and PEFT configuration with get_peft_model. 
# For the bigscience/mt0-large model, you're only training 0.19% of the parameters!
##########################################
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282"

#%%
##########################################
# To load a PEFT model for inference:
##########################################
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
