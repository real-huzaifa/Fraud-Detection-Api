import pandas as pd
from src import config


def load_merged(save: bool = True) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction + identity files."""
    txn = pd.read_csv(config.TRANSACTION_FILE)
    idn = pd.read_csv(config.IDENTITY_FILE)
    df = txn.merge(idn, on=config.ID_COL, how="left")

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.MERGED_FILE, index=False)

    return df


def load_processed() -> pd.DataFrame:
    """Reload the already-merged parquet (fast)."""
    return pd.read_parquet(config.MERGED_FILE)