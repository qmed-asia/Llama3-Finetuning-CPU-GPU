The training and validation datasets are loaded as csv files- train.csv and val.csv.

A standard preprocessing pipeline is applied to it. 
1. Ensure that the instruction fine-tuning dataset follows this template.

<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{Insert your instruction prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{Insert your input}<|eot_id|><|start_header_id|>system<|end_header_id|>\n\n{Summary}<|eot_id|>

2. Rename this section as _llama_prompt_ as one of the columns in train.csv and val.csv.
