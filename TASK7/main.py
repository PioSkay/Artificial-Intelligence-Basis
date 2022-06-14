from ast import While
from bayesian_network import BayesianNetwork
import argparse
import json

# Input

CONFIG_FILE_CONTENT = [
    'file_path',
    'queries'
]

QUERY_TYPES = [
    "markov_blanket",
    "mcmc",
    "debug"
]

MB_TYPE_CONTENT = [
    "type",
    "node"
]

MCMC_TYPE_CONTENT = [
    "type",
    "steps",
    "evidence",
    "query"
]

FILE_MAP = {
    QUERY_TYPES[0]: MB_TYPE_CONTENT,
    QUERY_TYPES[1]: MCMC_TYPE_CONTENT,
    QUERY_TYPES[2]: {}
}

def help():
    print("Formating the input file:\n\
     - python main.py -f <file_name>")

if __name__ == "__main__":
    input_args = argparse.ArgumentParser()
    input_args.add_argument('-f', '--file')
    args = input_args.parse_args()
    if args.file == None or args.file.endswith('json') == False:
        help()
        exit(0)
    else:
        input_file = args.file

    try:
        with open(input_file, 'r') as config_file:
            data = json.load(config_file)
        for i in CONFIG_FILE_CONTENT:
            if data.get(i) == None:
                raise Exception(f'Missing param {i}')

        with open(data[CONFIG_FILE_CONTENT[0]], 'r') as config_file:
            network_data = json.load(config_file)
        network = BayesianNetwork(network_data)
    except KeyboardInterrupt:
        print("Interrupt")
        exit(0)
    except Exception as e:
        print(f"Exception occured: {e}")
        exit(0)
    except:
        print("Unknown error")
        exit(1)

    query_id = 0
    for query in data[CONFIG_FILE_CONTENT[1]]:
        print()
        query_id += 1
        print(f"-------Evaluating query: [QueryId_{query_id}]-------")
        try:
            print("Query content:\n", query)
            if query.get(MCMC_TYPE_CONTENT[0]) == None:
                raise Exception(f"[QueryId_{query_id}] Missing type!")
            if query.get(MCMC_TYPE_CONTENT[0]) not in QUERY_TYPES:
                raise Exception(f"[QueryId_{query_id}] Invalid type!")
            q_type = query.get(MCMC_TYPE_CONTENT[0])

            for i in FILE_MAP[q_type]:
                if query.get(i) == None:
                    raise Exception(f'Missing param {i}')
            print(f"Query answer:")
            if q_type == QUERY_TYPES[0]:
                print(network.get_markov_blanket_for_node(query.get(MB_TYPE_CONTENT[1])))
            elif q_type == QUERY_TYPES[1]:
                output_probab_dict = network.mcmc(query["evidence"], query["query"], query["steps"])
                for element in output_probab_dict:
                    print("Query: " + element)
                    print("Response: ", output_probab_dict[element], sep="")
            elif q_type == QUERY_TYPES[2]:
                print(network.nodes_to_string())
        except Exception as e:
            print(f"Exception occured: {e}")
        except:
            print("Unknown error")

