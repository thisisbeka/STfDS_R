import pandas as pd

df = pd.read_csv('data.csv')

class MRL98_algo():
  """
  Args:
    num_quantiles: Number of quantiles to produce. It is the size of the final  output list, including the mininum and maximum value items.
    buffer_size: The size of the buffers, corresponding to k in the referenced paper.
    num_buffers: The number of buffers, corresponding to b in the referenced paper.
    key: (optional) Key is a mapping of elements to a comparable key, similar to the key argument of Python's sorting methods.
    reverse: (optional) whether to order things smallest to largest, rather than largest to smallest
  """

  _offset_jitter = 0

  _MAX_NUM_ELEMENTS = 1e9
  _qs = None  # Refers to the _QuantileState

  def __init__(self, num_quantiles, buffer_size, num_buffers, key=None,
               reverse=False):
    if key:
      self._comparator = lambda a, b: (key(a) < key(b)) - (key(a) > key(b)) \
        if reverse else (key(a) > key(b)) - (key(a) < key(b))
    else:
      self._comparator = lambda a, b: (a < b) - (a > b) if reverse \
        else (a > b) - (a < b)

    self._num_quantiles = num_quantiles
    self._buffer_size = buffer_size
    self._num_buffers = num_buffers
    self._key = key
    self._reverse = reverse

  @classmethod
  def create(cls, num_quantiles, epsilon=None, max_num_elements=None, key=None,
             reverse=False):
 
    max_num_elements = max_num_elements or cls._MAX_NUM_ELEMENTS
    if not epsilon:
      epsilon = 1.0 / num_quantiles
    b = 2
    while (b - 2) * (1 << (b - 2)) < epsilon * max_num_elements:
      b = b + 1
    b = b - 1
    k = max(2, math.ceil(max_num_elements / float(1 << (b - 1))))
    return cls(num_quantiles=num_quantiles, buffer_size=k, num_buffers=b,
               key=key, reverse=reverse)

  def _add_unbuffered(self, qs, elem):
    """
    Add a new buffer to the unbuffered list, creating a new buffer and
    collapsing if needed.
    """
    qs.unbuffered_elements.append(elem)
    if len(qs.unbuffered_elements) == qs.buffer_size:
      qs.unbuffered_elements.sort(key=self._key, reverse=self._reverse)
      heapq.heappush(qs.buffers,
                     _QuantileBuffer(elements=qs.unbuffered_elements))
      qs.unbuffered_elements = []
      self._collapse_if_needed(qs)

  def _offset(self, newWeight):
    """
    If the weight is even, we must round up or down. Alternate between these
    two options to avoid a bias.
    """
    if newWeight % 2 == 1:
      return (newWeight + 1) / 2
    else:
      self._offset_jitter = 2 - self._offset_jitter
      return (newWeight + self._offset_jitter) / 2

  def _collapse(self, buffers):
    new_level = 0
    new_weight = 0
    for buffer_elem in buffers:
      new_level = max([new_level, buffer_elem.level + 1])
      new_weight = new_weight + buffer_elem.weight
    new_elements = self._interpolate(buffers, self._buffer_size, new_weight,
                                     self._offset(new_weight))
    return _QuantileBuffer(new_elements, new_level, new_weight)

  def _collapse_if_needed(self, qs):
    while len(qs.buffers) > self._num_buffers:
      toCollapse = []
      toCollapse.append(heapq.heappop(qs.buffers))
      toCollapse.append(heapq.heappop(qs.buffers))
      minLevel = toCollapse[1].level

      while len(qs.buffers) > 0 and qs.buffers[0].level == minLevel:
        toCollapse.append(heapq.heappop(qs.buffers))

      heapq.heappush(qs.buffers, self._collapse(toCollapse))

  def _interpolate(self, i_buffers, count, step, offset):
    """
    Emulates taking the ordered union of all elements in buffers, repeated
    according to their weight, and picking out the (k * step + offset)-th
    elements of this list for `0 <= k < count`.
    """

    iterators = []
    new_elements = []
    compare_key = None
    if self._key:
      compare_key = lambda x: self._key(x[0])
    for buffer_elem in i_buffers:
      iterators.append(buffer_elem.sized_iterator())

    if sys.version_info[0] < 3:
      sorted_elem = iter(
          sorted(itertools.chain.from_iterable(iterators), key=compare_key,
                 reverse=self._reverse))
    else:
      sorted_elem = heapq.merge(*iterators, key=compare_key,
                                reverse=self._reverse)

    weighted_element = next(sorted_elem)
    current = weighted_element[1]
    j = 0
    while j < count:
      target = j * step + offset
      j = j + 1
      try:
        while current <= target:
          weighted_element = next(sorted_elem)
          current = current + weighted_element[1]
      except StopIteration:
        pass
      new_elements.append(weighted_element[0])
    return new_elements

  def create_accumulator(self):
    self._qs = _QuantileState(buffer_size=self._buffer_size,
                              num_buffers=self._num_buffers,
                              unbuffered_elements=[], buffers=[])
    return self._qs

  def add_input(self, quantile_state, element):
    """
    Add a new element to the collection being summarized by quantile state.
    """
    if quantile_state.is_empty():
      quantile_state.min_val = quantile_state.max_val = element
    elif self._comparator(element, quantile_state.min_val) < 0:
      quantile_state.min_val = element
    elif self._comparator(element, quantile_state.max_val) > 0:
      quantile_state.max_val = element
    self._add_unbuffered(quantile_state, elem=element)
    return quantile_state

  def merge_accumulators(self, accumulators):
    """Merges all the accumulators (quantile state) as one."""
    qs = self.create_accumulator()
    for accumulator in accumulators:
      if accumulator.is_empty():
        continue
      if not qs.min_val or self._comparator(accumulator.min_val,
                                            qs.min_val) < 0:
        qs.min_val = accumulator.min_val
      if not qs.max_val or self._comparator(accumulator.max_val,
                                            qs.max_val) > 0:
        qs.max_val = accumulator.max_val

      for unbuffered_element in accumulator.unbuffered_elements:
        self._add_unbuffered(qs, unbuffered_element)

      qs.buffers.extend(accumulator.buffers)
    self._collapse_if_needed(qs)
    return qs

  def extract_output(self, accumulator):
    """
    Outputs num_quantiles elements consisting of the minimum, maximum and
    num_quantiles - 2 evenly spaced intermediate elements. Returns the empty
    list if no elements have been added.
    """
    if accumulator.is_empty():
      return []

    all_elems = accumulator.buffers
    total_count = len(accumulator.unbuffered_elements)
    for buffer_elem in all_elems:
      total_count = total_count + accumulator.buffer_size * buffer_elem.weight

    if accumulator.unbuffered_elements:
      accumulator.unbuffered_elements.sort(key=self._key, reverse=self._reverse)
      all_elems.append(_QuantileBuffer(accumulator.unbuffered_elements))

    step = 1.0 * total_count / (self._num_quantiles - 1)
    offset = (1.0 * total_count - 1) / (self._num_quantiles - 1)
    quantiles = [accumulator.min_val]
    quantiles.extend(
        self._interpolate(all_elems, self._num_quantiles - 2, step, offset))
    quantiles.append(accumulator.max_val)
    return quantiles
