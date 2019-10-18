from pyrouge import Rouge155

def obtain_pyrouge_scores():
    r = Rouge155()

    # Setting the directory paths
    r.system_dir = "log/golden/"
    r.model_dir = "log/generated/"

    # Defining the file patterns in the directories
    r.system_filename_pattern = '(\d+)_golden.txt'
    r.model_filename_pattern = '(\d+)_generated.txt'

    # Obtain the metrics
    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)

    # Returning the output dictionary
    return output_dict

output_dict = obtain_pyrouge_scores()