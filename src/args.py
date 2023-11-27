import argparse

def get_args():
    """
    This function gets the arguments of the program, that is, the architecture, the optimizer and the learning rate.

    Returns: a dictionary with the values of the arguments, in which the keys are the names defined for each argument in the second argument of each of the functions below.
    """

    # Instantiate the argument parser.
    arg_parser = argparse.ArgumentParser()

    # Parse the architecture.
    arg_parser.add_argument("-a", "--architecture", required=False, default='deeplabv3_resnet101', type=str)
    
    # Parse the optimizer.
    arg_parser.add_argument("-o", "--optimizer", required=False, default='adam', type=str)

    # Parse the number of the run.
    arg_parser.add_argument("-r", "--run", required=False, default=1, type=int)
    
    # Parse the learning rate.
    arg_parser.add_argument("-l", "--learning_rate", required=False, default=0.001, type=float)

    # Parse the arguments and return them as a dictionary.
    return vars(arg_parser.parse_args())
