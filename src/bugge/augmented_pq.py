class AugmentedPQ:
    """A priority queue with several extra options."""
    
    def __init__(self, priority_fn=None):
        self._priority_fn = priority_fn
        self._size = 0
        self._dict = {}
        self._heap = []
        self._priority_counts = {}

    def _left_child_index(self, index):
        return ((index + 1) * 2) - 1

    def _right_child_index(self, index):
        return (index + 1) * 2

    def _parent_index(self, index):
        return int((int(index + 1) / 2)) - 1

    # Given an index, looks at its two children.
    # If one of those children has greater priority, swaps with that child.
    # The return value is the index of the original element's new position.
    def _update_at_index(self, index):
        left_idx = self._left_child_index(index)
        right_idx = self._right_child_index(index)

        parent_val = self._heap[index][0]
        left_val = parent_val
        right_val = parent_val
        if left_idx < self._size:
            left_val = self._heap[left_idx][0]
            if right_idx < self._size:
                right_val = self._heap[right_idx][0]
        else:
            return index

        min_child_val = left_val
        min_child_idx = left_idx
        if right_val < left_val:
            min_child_val = right_val
            min_child_idx = right_idx

        # If an actual swap is needed
        if min_child_val < parent_val:
            temp = self._heap[min_child_idx]
            self._heap[min_child_idx] = self._heap[index]
            self._heap[index] = temp
            self._dict[self._heap[index][1]].remove(min_child_idx)
            self._dict[self._heap[index][1]].add(index)
            self._dict[self._heap[min_child_idx][1]].remove(index)
            self._dict[self._heap[min_child_idx][1]].add(min_child_idx)
            return min_child_idx

        return index

    def push(self, x, prio=None):
        if prio is not None:
            pass
        elif self._priority_fn is not None:
            prio = self._priority_fn(x)
        else:
            prio = x
        self._heap.append((prio, x))
        if prio in self._priority_counts:
            self._priority_counts[prio] += 1
        else:
            self._priority_counts[prio] = 1

        idx = self._size
        if x in self._dict:
            self._dict[x].add(idx)
        else:
            self._dict[x] = set([idx])
        self._size += 1

        parent_idx = self._parent_index(idx)
        while parent_idx >= 0:
            if self._update_at_index(parent_idx) == parent_idx:
                break
            parent_idx = self._parent_index(parent_idx)

        return prio # Important for rule_pq

    def pop(self):
        item = self._heap[0][1]
        self.delete(item)
        return item

    def delete(self, x):
        index_set = self._dict[x]
        index = index_set.pop()

        prio = self._heap[index][0]
        self._priority_counts[prio] -= 1
        if self._priority_counts[prio] == 0:
            del self._priority_counts[prio]

        if len(index_set) == 0:
            del self._dict[x]
        self._size -= 1
        if self._size == 0:
            self._heap = []
            return prio # Important for rule_pq
        
        if index < self._size:
            self._heap[index] = self._heap.pop()
            self._dict[self._heap[index][1]].remove(self._size)
            self._dict[self._heap[index][1]].add(index)

            child_index = self._update_at_index(index)
            while child_index != index:
                index = child_index
                child_index = self._update_at_index(index)
        else:
            self._heap.pop()

        return prio # Important for rule_pq

    def update(self, x, new_prio=None):
        if new_prio is None:
            new_prio = self._priority_fn(x)
        index_set = self._dict[x]
        single_pop = index_set.pop()
        index_set.add(single_pop)
        old_prio = self._heap[single_pop][0]
        if new_prio == old_prio:
            return old_prio

        num_occurrences_of_x = len(index_set)
        for i in range(0, num_occurrences_of_x):
            index = index_set.pop()
            popped = [index]
            while self._heap[index][0] == new_prio: # TODO: make this more efficient
                index = index_set.pop()
                popped.append(index)
            for p in popped:
                index_set.add(p)

            self._heap[index] = (new_prio, self._heap[index][1])

            if old_prio < new_prio: # If we might need to move down.
                child_index = self._update_at_index(index)
                while child_index != index:
                    index = child_index
                    child_index = self._update_at_index(index)
            else: # If we might need to move up.
                parent_idx = self._parent_index(index)
                while parent_idx >= 0:
                    if self._update_at_index(parent_idx) == parent_idx:
                        break
                    parent_idx = self._parent_index(parent_idx)

        if new_prio in self._priority_counts:
            self._priority_counts[new_prio] += num_occurrences_of_x
        else:
            self._priority_counts[new_prio] = num_occurrences_of_x
        self._priority_counts[old_prio] -= num_occurrences_of_x
        if self._priority_counts[old_prio] == 0:
            del self._priority_counts[old_prio]

        return new_prio # Important for rule_pq

    def contains(self, x):
        return x in self._dict

    def top_item(self):
        return self._heap[0][1]

    def top_priority(self):
        return self._heap[0][0]

    def empty(self):
        return self._size == 0

    def size(self):
        return self._size

    def num_with_priority(self, prio):
        if prio not in self._priority_counts:
            return 0
        return self._priority_counts[prio]
