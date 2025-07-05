```sh
git clone https://github.com/allenai/olmo-cookbook
git checkout mayeec/core-2-test-2
pip install -e ".[all]"
pip install ipykernel ipywidgets transformers torch

# List data
ai2 ls /oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/adult_content/dolma2-tokenizer

# Pull example data
scp ai2:/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/art_and_design/dolma2-tokenizer/part-000-00000.npy .
```

```sh
python src/similarity.py
```

```sh
# Example output
Number of train/val tokens: 484,881,764 / 921,314
(science_math_and_technology) Mutual information: -24,329 bytes
Number of train/val tokens: 120,473,039 / 921,314
(software) Mutual information: 31,412 bytes
Number of train/val tokens: 500,000,000 / 921,314
(entertainment) Mutual information: 4,456 bytes
Number of train/val tokens: 120,083,330 / 921,314
(transportation) Mutual information: -11,772 bytes
Number of train/val tokens: 78,241,252 / 921,314
(art_and_design) Mutual information: -3,170 bytes
Number of train/val tokens: 277,421,748 / 921,314
(games) Mutual information: -12,835 bytes
Number of train/val tokens: 500,000,000 / 921,314
(politics) Mutual information: -7,469 bytes
Number of train/val tokens: 330,081,134 / 921,314
(religion) Mutual information: -37,547 bytes
Number of train/val tokens: 428,305,430 / 921,314
```