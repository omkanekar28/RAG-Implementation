import pandas as pd


class DatasetCleaner:
    """
    """
    
    def __init__(self, input_dataset_path: str) -> None:
        """
        """
        self.input_df = pd.read_csv(input_dataset_path)
        self.output_df = self.input_df.copy()
        self.columns_to_remove = ["homepage", "id", "original_language", "original_title", "status", "vote_count", "crew"]

    def remove_unreleased(self) -> None:
        """
        """
        pass
        
    def remove_unimportant_columns(self) -> None:
        """
        """
        pass

    def remove_duplicate_rows(self) -> None:
        """
        """
        pass


if __name__ == "__main__":
    cleaner = DatasetCleaner(input_dataset_path="/home/omkanekar28/code/RAG-Implementation/data/movie_dataset.csv")