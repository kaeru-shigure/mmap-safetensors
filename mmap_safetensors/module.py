import numpy as np
import json

type_map = {
    "BOOL": np.bool,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
}

class MmapSafetensorsDict(dict):
    def __init__(self, file_path, header_size, header, mode):
        self.file_path = file_path
        self.header_size = header_size
        self.header = header
        self.mode = mode
    @classmethod
    def load(cls, file_path, mode="r"):
        with open(file_path, 'rb') as f:
            header_size = int.from_bytes(f.read(8), byteorder='little')
            header = json.loads(f.read(header_size).decode('utf-8'))
        metadata = header.get("__metadata__", {}) or {}
        del header["__metadata__"]
        return cls(file_path, header_size, header, mode), metadata
    def __getitem__(self, key):
        info = self.header.get(key, None)
        if info is None:
            raise KeyError(key)
        return np.memmap(
            self.file_path,
            dtype = type_map[info["dtype"]],
            mode = 'r',
            offset = 8 + self.header_size + info["data_offsets"][0],
            shape = info["shape"],
            order = 'C',
        )
    def __setitem__(self, key, value):
        raise NotImplementedError()
    def __delitem__(self, key):
        raise NotImplementedError()
    def __iter__(self):
        return iter(self.keys())
    def __len__(self):
        return len(self.keys())
    def __repr__(self):
        return repr(self.header)
    def __str__(self):
        return str(self.header)
    def __contains__(self, key):
        return key in self.header
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    def clear(self):
        return self.header.clear()
    def keys(self):
        return self.header.keys()
    def values(self):
        return list(self[key] for key in self.keys())
    def items(self):
        return list((key, self[key]) for key in self.keys())
    def remove(self, key):
        del self.header[key]

def load_safetensors(*args, **kwargs):
    return MmapSafetensorsDict.load(*args, **kwargs)
