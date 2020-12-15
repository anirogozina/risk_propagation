from typing import NamedTuple, Any, List, Optional
import datetime


class Elem(NamedTuple):
    target: Optional[int]
    features: List[Any]
    time: str
    user_id: str

class Pipeline:
    def __init__(self, graph, model, print_every=100):
        self.graph = graph
        self.model = model
        self.losses = []
        self.cum_loss = 0
        self.avg_losses = []
        self.iterations = 0
        self.print_every = print_every
    
    def __call__(self, batch):
        for elem in batch:
            self.iterations += 1
            prev_p = self.graph.get_prev_p(elem)
            loss = self.model.calc_loss(elem, prev_p)
            self.cum_loss += loss
            self.losses.append(loss)
            if self.iterations % self.print_every == 0:
                self.avg_losses.append(self.cum_loss/self.print_every)
                self.cum_loss = 0
            self.model.backprop(elem, prev_p)
            self.graph.update(elem, self.model)
            
from dataclasses import dataclass
@dataclass
class GraphElem:
    elem: Elem
    prediction: float


class Graph:
    def __init__(self):
        self.storage = dict()
        self.unique_users = set()

    def get_prev_time(self, time):
        dttm = datetime.datetime.strptime(time, '%Y-%m-%d')
        return (dttm - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    def get_prev_p(self, elem):
        prev_time = self.get_prev_time(elem.time)
        prev_elem = self.storage.get(prev_time, {}).get(elem.user_id)
        if prev_elem is not None:
            return prev_elem.prediction
        return 1.0/100
    def get_default_elem(self, time, user_id):
        raise NotImplementedError
    
    def update(self, elem, model):
        # If it is new element of time, update all previous probas
        if self.storage.get(elem.time) is None:
            self.storage[elem.time] = {}
            prev_time = self.get_prev_time(elem.time)
            if self.storage.get(prev_time) is not None:
                prev_users = self.storage[prev_time]
                for user_id in self.unique_users:
                    graph_elem = prev_users.get(user_id)
                    if graph_elem is not None:
                        prediction = model.predict(graph_elem.elem.features, self.get_prev_p(graph_elem.elem))
                        prev_users[user_id].prediction = prediction
                    else:
                        default_elem = self.get_default_elem(prev_time, user_id)
                        prediction = model.predict(default_elem.features, self.get_prev_p(default_elem))
                        prev_users[user_id] = GraphElem(default_elem, prediction)
        # Add Element and prediction to storage  
        self.unique_users.add(elem.user_id)
        prediction = model.predict(elem.features, self.get_prev_p(elem))
        self.storage[elem.time][elem.user_id] = GraphElem(elem, prediction)
       
    