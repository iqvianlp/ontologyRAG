import argparse
import os
import subprocess
from pathlib import Path

from scripts.eval import eval_code_mapping, eval_mapping_level
from scripts.run_retrieval import run_retrieval


def parse_arguments():
    """
    Parse intput arguments
    :return:
    """
    parser = argparse.ArgumentParser(description='OntologyRAG pipeline CLI',
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='Commands to select the OntologyRAG task to perform',
                                       description='run_indexing: indexes and serves ontology graphs with oxigraph;\n'
                                                   'run_retrieval: accepts input code mapping question(s), allows users to select models and prompting strategies, retrieves sub-graphs from graph database, generates reasoning over retrieved results, and generates summary based on the input question, retrieved results and reasoning results;\n'
                                                   'eval_code_mapping: calculates code mapping accuracy based on given gold dataset;\n'
                                                   'eval_mapping_level: calculates mapping level assessment accuracy based on given gold dataset;\n',
                                       help='EXAMPLE:\n'
                                            'main.py run_indexing -gr <path_to_graph_data_folder>\n'
                                            'main.py run_retrieval -i <one input question or list of input questions> -m <one of gpt-35, gpt-4, flan-xxl, llama-3> -p <one of the following: zero-shot, few-shot, few-shot-enhanced, cot>\n'
                                            'To print the help-text describing the parameters to specify to run a specific task (i.e. "run_indexing", "run_retrieval", "eval_code_mapping", "eval_mapping_level"):\n'
                                            'main.py <task_name> -h',
                                       dest="task")

    # --> Sub-parser run_indexing
    index_sp = subparsers.add_parser('run_indexing')
    index_sp.add_argument('-gr', '--graph_data',
                          help='Required if TASK is equal to "run_indexing".'
                               'Full local path to the parent of the folder storing graph source ttl data.')

    # --> Sub-parser run_retrieval
    retrieve_sp = subparsers.add_parser('run_retrieval')
    retrieve_sp.add_argument('-i', '--input_question',
                             help='Required if TASK is equal to "run_retrieval".'
                                  'A text question or path to a list of questions in a file should be specified.')
    retrieve_sp.add_argument('-gs', '--graph_store',
                             help='Required if TASK is equal to "run_retrieval".'
                                  'Full local path to the folder where graph store should be loaded from.')
    retrieve_sp.add_argument('-m', '--model_name',
                             type=str,
                             choices=['gpt-35', 'gpt-4', 'flan-xxl', 'llama-3'],
                             help='Required if TASK is equal to "run_retrieval".'
                                  'One of the model keywords in the list should be provided: '
                                  '[gpt-35, gpt-4, flan-xxl, llama-3].')
    retrieve_sp.add_argument('-p', '--prompt_mode',
                             type=str,
                             choices=['zero-shot', 'few-shot', 'few-shot-enhanced', 'cot'],
                             default='few-shot-enhanced',
                             required=False,
                             help='Optional if TASK is equal to "run_retrieval".'
                                  'One of the prompt mode keywords in the list should be provided: '
                                  '[zero-shot, few-shot, few-shot-enhanced, cot].'
                                  'Defaults to "few-shot-enhanced" if no keyword is specified.')
    retrieve_sp.add_argument('-t', '--temperature',
                             type=float,
                             default=0.2,
                             required=False,
                             help='Optional if TASK is equal to "run_retrieval".'
                                  'Temperature to control the randomness of the generated text. Suggested range: 0 to 1.'
                                  'Defaults to "0.2" if no value is specified.')

    # --> Sub-parser eval_code_mapping
    eval_cm_sp = subparsers.add_parser('eval_code_mapping')
    eval_cm_sp.add_argument('-gc', '--gold_cm',
                            help='Required if TASK is "eval_code_mapping".'
                                 'Full local path to the code mapping gold dataset.')
    eval_cm_sp.add_argument('-pc', '--pred_cm',
                            help='Required if TASK is "eval_code_mapping".'
                                 'Full local path to file or folder containing the code mapping prediction result.')
    eval_cm_sp.add_argument('-mc', '--model_cm',
                            choices=['gpt-35', 'gpt-4', 'flan-xxl', 'llama-3'],
                            help='Required if TASK is "eval_code_mapping".'
                                 'Name of the model that generated the result.'
                                 'One of the model keywords in the list should be provided: '
                                 '[gpt-35, gpt-4, flan-xxl, llama-3].')

    # --> Sub-parser eval_mapping_level
    eval_ml_sp = subparsers.add_parser('eval_mapping_level')
    eval_ml_sp.add_argument('-gm', '--gold_ml',
                            help='Required if TASK is "eval_mapping_level".'
                                 'Full local path to the mapping level gold dataset.')
    eval_ml_sp.add_argument('-rpm', '--raw_pred_ml',
                            help='Required if TASK is "eval_mapping_level".'
                                 'Full local path to folder containing the mapping level raw prediction result.')
    eval_ml_sp.add_argument('-ppm', '--processed_pred_ml',
                            help='Required if TASK is "eval_mapping_level".'
                                 'Full local path of the folder to store the processed prediction results.')

    int_args = parser.parse_args()

    return int_args


if __name__ == '__main__':
    args = parse_arguments()
    if args.task == "run_indexing":

        graph_store = Path(args.graph_data, "graph_store")
        Path.mkdir(graph_store, exist_ok=True, parents=True)

        graph_source = Path(args.graph_data, "source_ttl")
        assert Path.is_dir(graph_source), f"Graph source TTL files cannot be found under {graph_source}!"

        for filename in os.listdir(graph_source):
            if filename[-3:] == "ttl":
                print(filename)
                filepath = Path(graph_source, filename)
                subprocess.run(["oxigraph", "load", "--location", graph_store,
                                "--graph", f"http://iqvia.com/ontologies/{filename[:-4]}",
                                "--file", filepath])

    elif args.task == "run_retrieval":
        summary = run_retrieval(input_question=args.input_question,
                                graph_data=args.graph_store,
                                model_name=args.model_name,
                                prompt_mode=args.prompt_mode,
                                temperature=args.temperature)
        print(summary)

    elif args.task == "eval_code_mapping":
        eval_result = eval_code_mapping.evaluate(gold_path=args.gold_cm,
                                                 pred_path=args.pred_cm,
                                                 model_name=args.model_cm)
        print(eval_result)

    elif args.task == "eval_mapping_level":
        eval_result = eval_mapping_level.evaluate(gold_path=args.gold_ml,
                                                  raw_pred_path=args.raw_pred_ml,
                                                  processed_pred_output_dir=args.processed_pred_ml)
        print(eval_result)

    else:
        raise ValueError("Please specify the name of the task to perform. "
                         "Choices are [run_indexing, run_retrieval, eval_code_mapping, eval_mapping_level]")
