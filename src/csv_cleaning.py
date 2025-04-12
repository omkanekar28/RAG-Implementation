import pandas as pd


class DatasetCleaner:
    """
    A class to clean the dataset for the recommendation system.
    """
    
    def __init__(self, input_dataset_path: str, output_store_path: str) -> None:
        """
        Initializes the DatasetCleaner object.
        """
        self.input_df = pd.read_csv(input_dataset_path)
        print("Dataset loaded")
        self.cleaned_df = self.input_df.copy()
        self.columns_to_remove = ["index", "homepage", "id", "original_language", "original_title", "status", 
                                  "vote_count", "crew", "budget", "revenue", "vote_average", "popularity"]
        self.output_store_path = output_store_path

    def remove_unreleased(self) -> None:
        """
        Removes all the movies that are not released.
        """
        self.cleaned_df = self.cleaned_df[self.cleaned_df["status"] == "Released"]
        print("Unreleased movies removed")
        
    def remove_unimportant_columns(self) -> None:
        """
        Removes all the columns that are not important for the recommendation system.
        """
        self.cleaned_df.drop(self.columns_to_remove, axis=1, inplace=True)
        print("Unimportant columns removed")

    def remove_duplicate_rows(self) -> None:
        """
        Removes all the duplicate rows from the dataset.
        """
        self.cleaned_df.drop_duplicates(inplace=True)
        print("Duplicate rows removed")

    def remove_nan_values(self) -> None:
        """
        Removes all the rows that have NaN values in them.
        """
        self.cleaned_df.dropna(inplace=True)
        print("NaN values removed")

    def start_process(self) -> None:
        """
        Starts the dataset cleaning process.
        """
        print("Starting dataset cleaning process")
        print(f"Dataset shape before cleaning: {self.cleaned_df.shape}")
        self.remove_unreleased()
        self.remove_unimportant_columns()
        self.remove_duplicate_rows()
        self.cleaned_df.to_csv(self.output_store_path, index=False)
        print("Dataset cleaning completed")
        print(f"Dataset shape after cleaning: {self.cleaned_df.shape}")


if __name__ == "__main__":
    cleaner = DatasetCleaner(
        input_dataset_path="/home/omkanekar28/code/RAG-Implementation/data/movie_dataset.csv",
        output_store_path="/home/omkanekar28/code/RAG-Implementation/data/cleaned_movie_dataset.csv"
    )
    cleaner.start_process()