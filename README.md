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

Idk... (1) MI shouldn't be negative (2) this domain overlap doesnt make sense...

```sh
        mutual information via Zstd compression of raw text (symmetric)         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Subset                      ┃ Train Bytes ┃   Val Bytes ┃ Mutual Information ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ travel_and_tourism          │ 222,913,859 │ 603,088,059 │        7,610 bytes │
│ social_life                 │ 218,662,670 │ 603,088,059 │        4,179 bytes │
│ industrial                  │ 241,199,404 │ 603,088,059 │        3,621 bytes │
│ education_and_jobs          │ 236,920,952 │ 603,088,059 │        3,445 bytes │
│ sports_and_fitness          │ 218,989,535 │ 603,088,059 │        2,999 bytes │
│ health                      │ 231,743,763 │ 603,088,059 │        2,027 bytes │
│ software_development        │ 219,866,444 │ 603,088,059 │        1,682 bytes │
│ games                       │ 217,588,162 │ 603,088,059 │        1,528 bytes │
│ science_math_and_technology │ 228,297,372 │ 603,088,059 │       -1,329 bytes │
│ religion                    │ 222,342,935 │ 603,088,059 │       -1,356 bytes │
│ software                    │ 229,289,903 │ 603,088,059 │       -1,641 bytes │
│ politics                    │ 236,777,490 │ 603,088,059 │       -2,092 bytes │
│ fashion_and_beauty          │ 220,674,926 │ 603,088,059 │       -2,463 bytes │
│ finance_and_business        │ 239,689,390 │ 603,088,059 │       -2,544 bytes │
│ electronics_and_hardware    │ 221,618,107 │ 603,088,059 │       -3,021 bytes │
│ literature                  │ 219,214,944 │ 603,088,059 │       -4,185 bytes │
│ history_and_geography       │ 222,219,903 │ 603,088,059 │       -5,701 bytes │
│ entertainment               │ 217,478,640 │ 603,088,059 │       -5,979 bytes │
│ art_and_design              │ 226,565,500 │ 603,088,059 │       -6,327 bytes │
│ adult_content               │ 220,576,577 │ 603,088,059 │       -6,341 bytes │
│ home_and_hobbies            │ 224,875,119 │ 603,088,059 │       -9,642 bytes │
│ transportation              │ 227,112,073 │ 603,088,059 │      -11,811 bytes │
│ crime_and_law               │ 236,116,286 │ 603,088,059 │      -13,083 bytes │
│ food_and_dining             │ 217,289,397 │ 603,088,059 │      -18,434 bytes │
└─────────────────────────────┴─────────────┴─────────────┴────────────────────┘
```