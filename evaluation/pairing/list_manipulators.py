from typing import List, Tuple, Any


def nest_tup_to_nest_list(tuples: Tuple[Tuple[Any]]) -> List[List[Any]]:
    return list(map(list, tuples))


def slice_items(lst: list, indices: list) -> list:
    return [lst[i] for i in indices]


def missing_indices(indices: List[int], N: int) -> List[int]:
    all_ind_set = set(range(N))
    miss_ind_set = all_ind_set.difference(set(indices))
    return list(miss_ind_set)


def concat_none(lst: List[Any], side: str = "right") -> List[List[Any]]:
    none_lst = [None] * len(lst)
    if side == "right":
        output = nest_tup_to_nest_list(zip(lst, none_lst))
    elif side == "left":
        output = nest_tup_to_nest_list(zip(none_lst, lst))
    return output


def get_miss_items(
    lst: List[Any],
    indices: List[int],
    none_side: str = "right"
) -> List[List[Any]]:
    miss_indices = missing_indices(indices, len(lst))
    miss_items = slice_items(lst, miss_indices)
    return concat_none(miss_items, side=none_side)
