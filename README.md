### setup

```sh
# install mixing branch of cookbook
git clone https://github.com/allenai/olmo-cookbook
git checkout mayeec/core-2-test-2
pip install -e ".[all]"

# install requirements
pip install -r requirements.txt

# List data
ai2 ls /oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/adult_content/dolma2-tokenizer

# Pull example data
scp ai2:/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/art_and_design/dolma2-tokenizer/part-000-00000.npy .
```

### run analysis

```sh
# Basic version of MMI
python src/similarity.py

# Fast version of MMI (fast, string-based compressor)
python src/similarity_fast.py
```

### example output

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Subset                      ┃ Train MB ┃ Val MB ┃ Mutual Information ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ science_math_and_technology │      4.2 │   13.9 │        2,654 bytes │
│ finance_and_business        │      4.6 │   13.9 │        2,549 bytes │
│ industrial                  │      4.6 │   13.9 │        2,515 bytes │
│ sports_and_fitness          │      4.2 │   13.9 │        2,459 bytes │
│ food_and_dining             │      4.1 │   13.9 │        2,405 bytes │
│ travel_and_tourism          │      4.3 │   13.9 │        2,377 bytes │
│ literature                  │      4.1 │   13.9 │        2,304 bytes │
│ home_and_hobbies            │      4.3 │   13.9 │        2,240 bytes │
│ politics                    │      4.5 │   13.9 │        2,212 bytes │
│ transportation              │      4.3 │   13.9 │        2,124 bytes │
│ crime_and_law               │      4.5 │   13.9 │        2,124 bytes │
│ social_life                 │      4.2 │   13.9 │        2,096 bytes │
│ electronics_and_hardware    │      4.2 │   13.9 │        2,078 bytes │
│ entertainment               │      4.1 │   13.9 │        2,048 bytes │
│ software_development        │      4.2 │   13.9 │        1,920 bytes │
│ health                      │      4.4 │   13.9 │        1,818 bytes │
│ software                    │      4.3 │   13.9 │        1,805 bytes │
│ fashion_and_beauty          │      4.2 │   13.9 │        1,659 bytes │
│ art_and_design              │      4.3 │   13.9 │        1,560 bytes │
│ games                       │      4.1 │   13.9 │        1,560 bytes │
│ history_and_geography       │      4.2 │   13.9 │        1,541 bytes │
│ religion                    │      4.3 │   13.9 │        1,539 bytes │
│ education_and_jobs          │      4.5 │   13.9 │        1,516 bytes │
│ adult_content               │      4.3 │   13.9 │        1,295 bytes │
└─────────────────────────────┴──────────┴────────┴────────────────────┘
```