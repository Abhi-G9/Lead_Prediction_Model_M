"""
Data Loader Module
Responsible for loading the raw X Education Leads dataset.
Guarantees the original dataset remains untouched.
"""

from pathlib import Path
import pandas as pd


def get_raw_data_path() -> Path:
    """Returns the standardized path to the raw dataset."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "original" / "Leads.csv"


def load_raw_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Loads the raw dataset from CSV.

    Args:
        filepath: Optional path to the CSV file. If None, uses default path.

    Returns:
        pd.DataFrame: A copy of the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    if filepath is None:
        filepath = get_raw_data_path()
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}. Please ensure Leads.csv is present in data/original/."
        )

    # Load with standard pandas engine
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset at {filepath} is empty.")

    return df.copy()


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Successfully loaded dataset with shape: {df.shape}")
