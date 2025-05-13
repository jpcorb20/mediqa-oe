from typing import List, Union, Dict, Callable, Generator


def identity(x: any) -> any:
    return x


def slice_gen(lst: List[Union[Dict[str, any], None]], field: str, operator: Callable = identity) -> Generator:
    for r in lst:
        yield operator(r.get(field)) if r and r.get(field) else ""
