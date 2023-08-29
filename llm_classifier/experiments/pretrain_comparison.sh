
declare -a models=("TheBloke/stable-vicuna-13B-HF" "NousResearch/Nous-Hermes-13b" "TheBloke/Wizard-Vicuna-13B-Uncensored-HF" "nomic-ai/gpt4all-13b-snoozy") # "mosaicml/mpt-7b-instruct"


for model in "${models[@]}"
do

    ## Download
    #download -m $model
    python utils/download.py $model
    downloaded_model="${model////_}"

    ## Run experiments
    ### Topic
    run -f "data/topic/topic_sample.parquet" -t topic -e zero -m "models/$downloaded_model" -n 10
    run -f "data/topic/topic_sample.parquet" -t topic -e few -m "models/$downloaded_model" -n 10

    ### Partisanship
    run -f "data/partisanship/partisanship_sample.parquet" -t partisanship -e zero -m "models/$downloaded_model" -n 10
    run -f "data/partisanship/partisanship_sample.parquet" -t partisanship -e few -m "models/$downloaded_model" -n 10

    ## Trust
    run -f "data/trustworthy/trustworthy_sample.parquet" -t trustworthy -e zero -m "models/$downloaded_model" -n 10
    run -f "data/trustworthy/trustworthy_sample.parquet" -t trustworthy -e few -m "models/$downloaded_model" -n 10

    ### Time Benchmark
    benchmark -m models/$downloaded_model

    ## Delete
    #rm -rf ~/.cache/huggingface/hub
    rm -rf "models/$downloaded_model"

done