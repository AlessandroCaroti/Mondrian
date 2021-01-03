from Utility.utility import *
import Utility.data as data
from mandorian import main

if __name__ == "__main__":

    # load parameters
    parser = get_parser()
    args = parser.parse_args()

    # print algorithm parameters
    print_args(args)

    # Load the database to anonymize
    print("LOAD DATASET")
    data = data.Data(args.dataset_name, args.columns_type, args.result_name)

    # execute Mondrian
    main(args, data)
