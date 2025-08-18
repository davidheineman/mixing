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

# Fast version of MMI with fancy compression
python src/similarity_fast.py
```

### example output

```sh
          mutual information via consistent Zstd streaming (symmetric)          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Subset                      ┃ Train Tokens ┃ Val Tokens ┃ Mutual Information ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ software                    │   50,000,000 │  3,616,639 │       38,809 bytes │
│ science_math_and_technology │   50,000,000 │  3,616,639 │       44,083 bytes │
│ art_and_design              │   50,000,000 │  3,616,639 │       45,386 bytes │
│ entertainment               │   50,000,000 │  3,616,639 │       37,990 bytes │
│ transportation              │   50,000,000 │  3,616,639 │       34,371 bytes │
│ games                       │   50,000,000 │  3,616,639 │       43,045 bytes │
│ politics                    │   50,000,000 │  3,616,639 │       35,352 bytes │
│ literature                  │   50,000,000 │  3,616,639 │       34,649 bytes │
│ religion                    │   50,000,000 │  3,616,639 │       34,438 bytes │
│ social_life                 │   50,000,000 │  3,616,639 │       36,078 bytes │
│ software_development        │   50,000,000 │  3,616,639 │       46,038 bytes │
│ sports_and_fitness          │   50,000,000 │  3,616,639 │       38,065 bytes │
│ finance_and_business        │   50,000,000 │  3,616,639 │       38,434 bytes │
│ fashion_and_beauty          │   50,000,000 │  3,616,639 │       28,918 bytes │
│ crime_and_law               │   50,000,000 │  3,616,639 │       36,154 bytes │
│ industrial                  │   50,000,000 │  3,616,639 │       38,365 bytes │
│ education_and_jobs          │   50,000,000 │  3,616,639 │       39,716 bytes │
│ health                      │   50,000,000 │  3,616,639 │       32,564 bytes │
│ history_and_geography       │   50,000,000 │  3,616,639 │       46,624 bytes │
│ food_and_dining             │   50,000,000 │  3,616,639 │       33,240 bytes │
│ travel_and_tourism          │   50,000,000 │  3,616,639 │       35,289 bytes │
│ home_and_hobbies            │   50,000,000 │  3,616,639 │       34,887 bytes │
│ electronics_and_hardware    │   50,000,000 │  3,616,639 │       37,471 bytes │
│ adult_content               │   50,000,000 │  3,616,639 │       23,626 bytes │
└─────────────────────────────┴──────────────┴────────────┴────────────────────┘
```