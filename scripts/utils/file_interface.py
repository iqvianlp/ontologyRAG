from pathlib import Path
import pandas as pd


class InputDoc:
    """
    Class for storing indivisual document in the input file.
    """
    def __init__(self, doc_type: str = "text", 
                 doc_id: str = None, 
                 doc_index: list[str] = None, 
                 doc_row: pd.Series = None, 
                 doc_text: str = None):
        """
        Args:
            doc_type (str): Type of the document. It shoud be either a "file" or a "text".
            doc_id (str): The value in the first column of each row is the default document ID. Assigned when the doc_type is "file".
            doc_index (list[str]): A list of column headers of the document. Assigned when the doc_type is "file".
            doc_row: (pd.Series): Content of each document (row) of the input file. Assigned when the doc_type is "file".
            doc_text (str): Raw text if the doc_type is "text".
        """
        self.doc_type = doc_type
        self.doc_id = doc_id
        self.doc_index = doc_index
        self.doc_row = doc_row
        self.doc_text = doc_text
        self.answer = None
