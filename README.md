# cs3244project
Classifying elderly falls with a CNN -> RNN (LRCN model) using depth video data.

## Setup
1. `pip install -r requirements.txt`
2. `cat data/data_urls.txt | parallel --gnu "wget -P data/sequences/ {}"`
