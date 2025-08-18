### setup

```sh
git clone https://github.com/allenai/olmo-cookbook
git checkout mayeec/core-2-test-2
pip install -e ".[all]"
pip install ipykernel ipywidgets transformers torch blosc datasets rich

# List data
ai2 ls /oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/adult_content/dolma2-tokenizer

# Pull example data
scp ai2:/oe-training-default/ai2-llm/preprocessed/dclm/baseline_topic_classified_sample/art_and_design/dolma2-tokenizer/part-000-00000.npy .
```

### run analysis

```sh
python src/similarity.py
```

### example output

```sh
                   mutual information using gzip compression                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Subset                      ┃ Train Tokens ┃ Val Tokens ┃ Mutual Information ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ science_math_and_technology │   50,000,000 │  3,618,369 │        1,869 bytes │
│ software                    │   50,000,000 │  3,618,369 │         -999 bytes │
│ entertainment               │   50,000,000 │  3,618,369 │       -1,819 bytes │
│ transportation              │   50,000,000 │  3,618,369 │       -1,124 bytes │
│ art_and_design              │   50,000,000 │  3,618,369 │       -3,520 bytes │
│ games                       │   50,000,000 │  3,618,369 │       -2,905 bytes │
│ politics                    │   50,000,000 │  3,618,369 │       -4,972 bytes │
│ religion                    │   50,000,000 │  3,618,369 │        2,038 bytes │
│ literature                  │   50,000,000 │  3,618,369 │         -332 bytes │
│ social_life                 │   50,000,000 │  3,618,369 │          733 bytes │
│ sports_and_fitness          │   50,000,000 │  3,618,369 │       -2,082 bytes │
│ finance_and_business        │   50,000,000 │  3,618,369 │       -1,687 bytes │
│ fashion_and_beauty          │   50,000,000 │  3,618,369 │          980 bytes │
│ software_development        │   50,000,000 │  3,618,369 │       -3,608 bytes │
│ crime_and_law               │   50,000,000 │  3,618,369 │           33 bytes │
│ industrial                  │   50,000,000 │  3,618,369 │         -477 bytes │
│ education_and_jobs          │   50,000,000 │  3,618,369 │        1,350 bytes │
│ food_and_dining             │   50,000,000 │  3,618,369 │       -3,793 bytes │
│ health                      │   50,000,000 │  3,618,369 │        2,712 bytes │
│ travel_and_tourism          │   50,000,000 │  3,618,369 │       -1,124 bytes │
│ home_and_hobbies            │   50,000,000 │  3,618,369 │         -767 bytes │
│ history_and_geography       │   50,000,000 │  3,618,369 │        1,662 bytes │
│ electronics_and_hardware    │   50,000,000 │  3,618,369 │        1,324 bytes │
│ adult_content               │   50,000,000 │  3,618,369 │       -1,059 bytes │
└─────────────────────────────┴──────────────┴────────────┴────────────────────┘
```