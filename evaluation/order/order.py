from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional


@dataclass
class Order:
    """Generic Order class"""
    description: str
    order_type: str
    reason: str
    provenance: list[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Order object to a dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create an Order object from a dictionary."""
        valid = {k: data.get(k, None) for k in cls.__annotations__.keys()}
        return cls(**valid)
