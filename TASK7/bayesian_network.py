import numpy as np

from pytest import mark


MAIN_STRUCTURE = [
    "nodes",
    "relations"
]

PARENTS = "parents"
CHILDREN = "children"
PROBABILITIES = "probabilities"

class _node:
    def __init__(self, name, parents, probab, probab_values) -> None:
        self.name = name
        self.parents = parents
        self.probab = probab
        self.probab_values = probab_values
        self.children = []
        self.value = ''

    def __str__(self):
        output = f'Node: {self.name}\n' + \
               f'Parents: [{self.parents}]\n' + \
               f'Probabilities: [{self.probab}]\n'
        output += f'Children: [{self.children}]\n' if self.children else ''
        return output

    def connect(self, node):
        self.children.append(node)

    def has_cycle(self):
        return self.name in self.parents or self.name in self.children

class BayesianNetwork:
    def __init__(self, json_data) -> None:
        for i in MAIN_STRUCTURE:
            if json_data.get(i) == False:
                raise Exception(f'Missing {i}')
        self.nodes = []
        self._init_nodes(json_data)
        self._validate_network()

    '''Initializes network nodes'''
    def _init_nodes(self, json_data):
        for node_name in json_data[MAIN_STRUCTURE[0]]:
            probabilities = json_data[MAIN_STRUCTURE[1]][node_name][PROBABILITIES]
            
            prob_struct = []
            for ele in list(probabilities.keys()):
                prob_struct.append(ele.split(',')[-1])
            self.nodes.append(_node(node_name,
                                    json_data[MAIN_STRUCTURE[1]][node_name][PARENTS],\
                                    json_data[MAIN_STRUCTURE[1]][node_name][PROBABILITIES],\
                                    prob_struct))
        self._init_children()

    '''Initializes network children'''
    def _init_children(self):
        for node in self.nodes:
            for parent in node.parents:
                parent_node = self._get_node(parent)
                parent_node.connect(node.name)

    '''Verifies the propabilities'''
    def _validate_network(self):
        for node in self.nodes:
            probab_vals = list(node.probab.values())
            probab_keys = list(node.probab.keys())
            for ele in probab_keys:
                if len(ele.split(',')) != len(node.parents)+1:
                    raise Exception(f'[{node.name}] Number of parents does not match')
            for value in probab_vals:
                if value > 1 or value < 0:
                    raise Exception(f'[{node.name}] Invalid probability')
            if len(node.parents) == 0:
                sum = 0
                for value in probab_vals:
                    sum += value
                if float(sum) != 1.0:
                    raise Exception(f'[{node.name}] Probability does not sum to 1.0')
            if node.has_cycle() == True:
                raise Exception(f'[{node.name}] Node contains cycle')

    '''Returns a node with a given name'''
    def _get_node(self, name) -> _node:
        for node in self.nodes:
            if name == node.name:
                return node
        return None

    '''Returns a markov blanket.
       It stores its parents, children and parents of children.
    '''
    def get_markov_blanket_for_node(self, name):
        markov_dict = self._generate_markov_blanket_dict(name)
        markov_string = set()
        for child in markov_dict[CHILDREN]:
            markov_string.add(child)
        for parent in markov_dict[PARENTS]:
            markov_string.add(parent)
        for cp in markov_dict[CHILDREN + PARENTS]:
            for child in markov_dict[CHILDREN + PARENTS][cp]:
                markov_string.add(child)
        markov_string.remove(name)
        return list(markov_string)

    def _generate_markov_blanket_dict(self, name):
        node = self._get_node(name)
        if node == None:
            raise Exception(f"Could not find node {name}")
        markov_dict = {}
        markov_dict[PARENTS] = node.parents
        markov_dict[CHILDREN] = node.children
        children_parent = {}
        for child in node.children:
            c_node = self._get_node(child)
            children_parent[child] = c_node.parents
        markov_dict[CHILDREN + PARENTS] = children_parent
        return markov_dict

    def nodes_to_string(self) -> str:
        output = ''
        for node in self.nodes:
            output += str(node)
        return output

    def mcmc(self, evidence, queries, step):
        evidence_except = self._get_nodes_not_in_evidence(evidence)
        for node in evidence_except:
            node.value = np.random.choice(node.probab_values)
        probab_couter = self._get_query_probab_dict(queries)

        for _ in range(step):
            Xi = evidence_except[np.random.randint(0, len(evidence_except))]
            
            Xi.value = self._sample(Xi)

            for query in queries:
                probab_couter[query][self._get_node(query).value] += 1.0
        
        local_sum = 0
        for res in probab_couter:
            local_sum = sum(list(probab_couter[res].values()))

        for res in probab_couter:
            for node in probab_couter[res].keys():
                probab_couter[res][node] /= local_sum

        return probab_couter

    def _get_nodes_not_in_evidence(self, evidence) -> list:
        evidence_output = []
        for node in self.nodes:
            if node.name in evidence.keys():
                node.value = evidence[node.name]
            else:
                evidence_output.append(node)
        return evidence_output

    def _get_query_probab_dict(self, query) -> dict:
        probab_dict = {}
        for node in query:
            node_ptr = self._get_node(node)
            if node_ptr == None:
                raise Exception(f"Could not find node {node}")
            values = {}
            for probab in node_ptr.probab_values:
                values[probab] = 0
            probab_dict[node] = values
        return probab_dict

    def _sample(self, node: _node):
        markov_blanket = self._generate_markov_blanket_dict(node.name)
        probab_dict = {}

        for xj in node.probab_values:
            node.value = xj

            node_parents = ""
            for parent in markov_blanket[PARENTS]:
                parent_ptr = self._get_node(parent)
                node_parents += str(parent_ptr.value) + ','
            node_parents += node.value

            probab_parent = node.probab[node_parents]

            node_children = ""
            local_probab = 1
            for children in markov_blanket[CHILDREN]:
                children_ptr = self._get_node(children)
                parent_vals = []
                for parent in markov_blanket[CHILDREN + PARENTS][children]:
                    parent_ptr = self._get_node(parent)
                    parent_vals.append(str(parent_ptr.value))
                parent_vals.append(str(children_ptr.value))

                node_children = ','.join(parent_vals)
                local_probab *= children_ptr.probab[node_children]
            
            probab_dict[xj] = probab_parent * local_probab
        return self._roulette_select(probab_dict)

    def _roulette_select(self, probab_dict):
        probab_sum = sum(list(probab_dict.values()))
        normalized = [element / probab_sum for element in list(probab_dict.values())]

        prev = 0
        wheel = []
        for element in normalized:
            curr = prev + element
            wheel.append(curr)
            prev = curr
        
        i = 0
        roulette_pick = np.random.uniform(0, 1)
        while i in range(len(wheel)) and wheel[i] < roulette_pick:
            i += 1
        
        if i > len(wheel) - 1:
            i = len(wheel) - 1

        return (list(probab_dict.items())[i])[0]