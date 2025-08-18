### setup

```sh
pip install -r requirements.txt
```

<details>
<summary>More features</summary>

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

</details>

### run analysis

```sh
# Basic version of MMI
python src/similarity.py

# Fast version of MMI (fast, string-based compressor)
python src/similarity_fast.py
```

### example output

*Currently, I'm observing negative MIs for >20M tokens.*

```sh
+-----------------------------+----------+--------+--------------------+
| Subset                      | Train MB | Val MB | Mutual Information |
+-----------------------------+----------+--------+--------------------+
| software                    |  109.307 | 62.143 |        5,198 bytes |
| entertainment               |  103.543 | 62.143 |        4,874 bytes |
| religion                    |  105.842 | 62.143 |        4,426 bytes |
| crime_and_law               |  112.435 | 62.143 |        4,275 bytes |
| finance_and_business        |  114.269 | 62.143 |        4,217 bytes |
| education_and_jobs          |  113.190 | 62.143 |        3,937 bytes |
| health                      |  110.642 | 62.143 |        3,920 bytes |
| games                       |  103.444 | 62.143 |        3,860 bytes |
| transportation              |  108.465 | 62.143 |        3,766 bytes |
| history_and_geography       |  105.861 | 62.143 |        3,760 bytes |
| sports_and_fitness          |  104.302 | 62.143 |        3,709 bytes |
| food_and_dining             |  103.497 | 62.143 |        3,678 bytes |
| literature                  |  104.664 | 62.143 |        3,306 bytes |
| home_and_hobbies            |  107.395 | 62.143 |        3,268 bytes |
| travel_and_tourism          |  106.273 | 62.143 |        3,175 bytes |
| software_development        |  104.737 | 62.143 |        3,123 bytes |
| adult_content               |  105.328 | 62.143 |        3,089 bytes |
| politics                    |  113.064 | 62.143 |        2,982 bytes |
| fashion_and_beauty          |  105.339 | 62.143 |        2,486 bytes |
| science_math_and_technology |  108.772 | 62.143 |        2,391 bytes |
| social_life                 |  103.932 | 62.143 |        2,069 bytes |
| art_and_design              |  108.548 | 62.143 |        2,035 bytes |
| industrial                  |  115.161 | 62.143 |        1,605 bytes |
| electronics_and_hardware    |  105.637 | 62.143 |        1,012 bytes |
+-----------------------------+----------+--------+--------------------+
```
