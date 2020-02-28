from src.bugge.augmented_pq import *

class RulePQ(AugmentedPQ):
    
    def __init__(self, priority_fn=None):
        self.counts_of_nodes_by_prio = {}
        self.unique_prio_queue = AugmentedPQ()
        AugmentedPQ.__init__(self, priority_fn)

    def add_nodes_with_prio(self, nodes, prio):
        if prio not in self.counts_of_nodes_by_prio:
            self.counts_of_nodes_by_prio[prio] = {}
        for node in nodes:
            if node in self.counts_of_nodes_by_prio[prio]:
                self.counts_of_nodes_by_prio[prio][node] += 1
            else:
                self.counts_of_nodes_by_prio[prio][node] = 1

        if not self.unique_prio_queue.contains(prio):
            self.unique_prio_queue.push(prio)
    
    def delete_nodes_with_prio(self, nodes, prio): 
        for node in nodes:
            self.counts_of_nodes_by_prio[prio][node] -= 1
            if self.counts_of_nodes_by_prio[prio][node] == 0:
                del self.counts_of_nodes_by_prio[prio][node]
        if len(self.counts_of_nodes_by_prio[prio]) == 0:
            del self.counts_of_nodes_by_prio[prio]
            self.unique_prio_queue.delete(prio)

    def push(self, x, prio=None):
        prio = AugmentedPQ.push(self, x, prio)
        self.add_nodes_with_prio(x, prio)

    def delete(self, x):
        prio = AugmentedPQ.delete(self, x)
        self.delete_nodes_with_prio(x, prio)

    def update(self, x, new_prio=None):
        old_prio = self._heap[self._dict[x]][0]
        new_prio = AugmentedPQ.update(x, new_prio)
        self.delete_nodes_with_prio(x, old_prio)
        self.add_nodes_with_prio(x, new_prio)

    def number_of_nodes_covered_at_priority(self, prio):
        if prio not in self.counts_of_nodes_by_prio:
            return 0
        return len(self.counts_of_nodes_by_prio[prio])

    def sorted_list_of_prios(self):
        size = self.unique_prio_queue.size()
        the_list = [self.unique_prio_queue.pop() for i in range(0, size)]
        for i in range(0, size):
            self.unique_prio_queue.push(the_list[i])
        return the_list
