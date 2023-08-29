
run -f "data/trustworthy_account/trustworthy_accounts_sample.parquet" -t trustworthy_account -e few -n 10 -m "gpt-3.5-turbo" -k OPEN_AI_KEY
run -f "data/partisanship_account/partisanship_accounts_sample.parquet" -t partisanship_account -e few -n 10 -m "gpt-3.5-turbo" -k OPEN_AI_KEY
