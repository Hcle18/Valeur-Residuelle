import pandas as pd
from datetime import datetime, date

def parse_date(date_str):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
        return pd.NaT