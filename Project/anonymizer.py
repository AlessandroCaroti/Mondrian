
from Utility.utility import *
from Utility.data import Data
from mandorian import main

if __name__ == "__main__":

    # load parameters
    parser = get_parser()
    args = parser.parse_args()

    # print algorithm parameters
    print_args(args)

    # Load the database to anonymize
    data = Data(args.dataset_name, args.columns_type, args.result_name)

    main(args, data)
