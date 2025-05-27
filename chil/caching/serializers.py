"""
Serializers for Redis cache data.
"""

import json
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import logging

logger = logging.getLogger(__name__)


class Serializer(ABC):
    """Base serializer interface."""
    
    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        pass


class JSONSerializer(Serializer):
    """JSON serializer for Redis data."""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to JSON bytes."""
        return json.dumps(value, separators=(',', ':')).encode(self.encoding)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to value."""
        return json.loads(data.decode(self.encoding))


class PickleSerializer(Serializer):
    """Pickle serializer for Redis data."""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value using pickle."""
        return pickle.dumps(value, protocol=self.protocol)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes to value."""
        return pickle.loads(data)


class MsgPackSerializer(Serializer):
    """MessagePack serializer for Redis data."""
    
    def __init__(self):
        try:
            import msgpack
            self.msgpack = msgpack
        except ImportError:
            raise ImportError("msgpack-python is required for MsgPackSerializer")
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value using msgpack."""
        return self.msgpack.packb(value, use_bin_type=True)
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize msgpack bytes to value."""
        return self.msgpack.unpackb(data, raw=False)


# Serializer registry
_serializers: Dict[str, Type[Serializer]] = {
    "json": JSONSerializer,
    "pickle": PickleSerializer,
    "msgpack": MsgPackSerializer,
}


def get_serializer(name: str, **kwargs) -> Serializer:
    """
    Get serializer by name.
    
    Args:
        name: Serializer name (json, pickle, msgpack)
        **kwargs: Additional arguments for serializer
    
    Returns:
        Serializer instance
    """
    if name not in _serializers:
        raise ValueError(f"Unknown serializer: {name}")
    
    serializer_class = _serializers[name]
    return serializer_class(**kwargs)


def register_serializer(name: str, serializer_class: Type[Serializer]) -> None:
    """
    Register a custom serializer.
    
    Args:
        name: Serializer name
        serializer_class: Serializer class
    """
    if not issubclass(serializer_class, Serializer):
        raise TypeError("Serializer must inherit from Serializer base class")
    
    _serializers[name] = serializer_class
    logger.info(f"Registered serializer: {name}")


class CompressedSerializer(Serializer):
    """Wrapper to add compression to any serializer."""
    
    def __init__(self, base_serializer: Serializer, level: int = 6):
        self.base_serializer = base_serializer
        self.level = level
        
        import zlib
        self.zlib = zlib
    
    def serialize(self, value: Any) -> bytes:
        """Serialize and compress value."""
        data = self.base_serializer.serialize(value)
        return self.zlib.compress(data, level=self.level)
    
    def deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize value."""
        decompressed = self.zlib.decompress(data)
        return self.base_serializer.deserialize(decompressed)