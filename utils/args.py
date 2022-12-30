import argparse

def get_args(additional_parser=None):
    parser = argparse.ArgumentParser()

    # cuda and seed
    parser.add_argument('--no-cuda', action='store_true', default=False,            
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=77, 
                        help='Random seed.')

    # dataset and model architecture
    parser.add_argument('--dataset', type=str, default="CIFAR10", 
                        help='dataset name')
    parser.add_argument('--model', type=str, default="resnet18",
                        help='model architecture name')
    
    # hyperparameters 
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--optim', type=str, default="Momentum",
                        choices=["SGD", "ADAM", "Momentum"], 
                        help='model optimizer name')
    parser.add_argument('--scheduler', type=str, default="Cosine",
                        help='scheduler name')
    parser.add_argument("--regularizer", type=float, default=0.0005,
                        help='How much strongly weights regularize')
    parser.add_argument("--batch_size", type=int, default=128,
                        help='The batch size of one step')

    parser.add_argument("--model_save", action='store_true', default=False,
                        help="set True if you are testing your code.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="set exp_name if you want to put some special message in log_dir.")
    parser.add_argument("--test_exp", action='store_true', default=False,
                        help="set True if you are testing your code.")

    if additional_parser is not None:
        parser = additional_parser(parser)

    args = parser.parse_args()
    return args