import pandas as pd

from scripts.utils.file_interface import InputDoc


class Query:
    """
    Query to document prompt details and examples.
    """
    
    def __init__(self,
                 query_id: int,
                 label: str,
                 task: str,
                 region: str,
                 instruction: str = None,
                 note: str = None,
                 example: list[str] = None,
                 ) -> None:
        """
        Args:
            query_id (int): Unique number of the task.
            label (str): Label of the task.
            task (str): Main purpose of the task.
            region (str): Column header of the source data to be processed.
            indtruction (str): Detailed description of the task.
            note (str): specify the output format.
            example: (list[str]): A list of examples for the task.
        """

        self.query_id = query_id
        self.label = label
        self.task = task
        self.instruction = instruction
        self.region = region
        self.note = note
        self.example = example

    
    def format_message(self, input_doc: InputDoc) -> list[list[dict[str:str]]]:
        """
        Formats the prompt in a format that is comprehensible by llms to generate answers.
        """
        input_text = input_doc.doc_text
        
        message = [
            {"role": "system", "content": f"Task: {self.task}"},
            {"role": "system", "content": f"Instructions: {self.instruction}"},
            {"role": "system", "content": f"Note: {self.note}"},
            {"role": "system", "content": f"Examples: {self.example}"},
            {"role": "user", "content": f"{input_text}"},
        ]

        return message


def format_example(query_id: int, example_file_path: str) -> list[str]:
    """
    Concats examples belong to the same query based on the query ID.
    """
    df_example = pd.read_excel(example_file_path)
    if query_id in list(df_example["Task ID"].values):
        examples = []
        for i, row in df_example.iterrows():
            if row["Task ID"] == query_id:
                examples.append(
                    f"""# {row["Input"]}
                    {row["Output"]}
                    """
                )
        return examples
    

def read_prompts(prompt_file_path: str,
                 example_file_path: str = None) -> list[Query]:
    """
    Reads prompts and examples and generate a list of Query objects.
    
    Args:
        prompt_file_path (str): Path of the prompt file.
        example_file_path (str): Path of the file with examples for few shot learning.

    Returns:
        list[Query]: A list of Query objects.
    """
    queries = []

    # Read prompt file
    df_prompt = pd.read_excel(prompt_file_path)
    for i, row in df_prompt.iterrows():
        query = Query(
             query_id = int(row["Task ID"]),
             label = row["Task Label"],
             task = row["Task Summary"], 
             instruction = row["Instruction"],
             region = row["Region"],
             note=row["Note"])
    
        # Format examples
        if example_file_path:
            query.example = format_example(query.query_id, example_file_path)

        queries.append(query)

    return queries
