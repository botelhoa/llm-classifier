
import pandas as pd

from torch.utils.data import Dataset


task_maps = {
    "topic":  {
        "government/politics": 0, 
        "sports/fitness": 1, 
        "business/economics": 2, 
        "arts/culture/entertainment": 3, 
        "crime/public safety": 4, 
        "school/education": 5,
        "miscellaneous": 6,
    },
    "partisanship": {
        'liberal': 0, 
        'moderate': 1, 
        'conservative': 2,
    },
    "partisanship_account": {
        'liberal': 0, 
        'centrist': 1, 
        'conservative': 2,
    },
    "trustworthy": {
        'true': 1,
        'false': 0, 
    },
}


def create_text_field(row: pd.Series) -> str:

    if row["message"]:
        return row["message"]
    elif row["description"]:
        return row["description"]
    elif "title" in row:
        return row["title"]
    elif row["image_text"]:
        return row["image_text"]
    else:
        return ""



class PostDataset(Dataset):
    def __init__(self, file_path: str, task: str,) -> None:

        self.task = task
        self.df = pd.read_parquet(file_path).reset_index(drop=True)

            
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:

        if self.task == "topic":
            text = create_text_field(self.df.loc[idx, :])
            id = self.df.loc[idx, "id"]
            #text = self.df.loc[idx, "final_text"]
        elif self.task in ["partisanship_account", "trustworthy_account"]:
            text = self.df.loc[idx, "account_name"]
            id = self.df.loc[idx, "account_id"]

        else:
            id = self.df.loc[idx, "account_id"]
            text = self.df.loc[idx, "text"]

        return {"id": str(id), "text": text}
