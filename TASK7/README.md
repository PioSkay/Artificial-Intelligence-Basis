
### Configuration file

#### Avalible paramaters
* file_path - file path to the input file
* queries - queries that are going to be evaluated during execution

There are three types of queries avalible:
In the case of mcmc query following parameters should be added:

{
    "type": "mcmc"
    "steps": 1000 -> integer with the numer of steps,
    "evidence": {
        "HighFever": "T" -> evidence dictionary
    },
    "query": ["Flu", "HighFever"] -> query
}

On the other hand in the case of markov blanket query following parameters should be added:

{
    "type": "markov_blanket"
    "node": "Flu" -> node
}

Finally, to get a debug output:

{
    "type":"debug"
}

#### Runing the code

python main.py -f <config_file>

#### Example:

python main.py -f config.json