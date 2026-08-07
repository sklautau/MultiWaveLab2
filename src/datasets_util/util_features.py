'''Useful code for dealing with features'''

from typing import Dict, Any


@staticmethod
def append_identification(dic: Dict[str, Any], modality: str, group: str) -> Dict[str, Any]:
    """
    For all keys (originalkey) in the dictionary dic, prepend
    modality and group as modality_group_originalkey.

    Rules:
    - If key starts with modality_, insert group after modality
    - Otherwise prepend modality_group_
    """

    out = {}
    prefix = f"{modality}_"
    new_prefix = f"{modality}_{group}_"

    for key, value in dic.items():

        if key.startswith(prefix):
            # remove modality_ prefix
            rest = key[len(prefix):]
            new_key = f"{modality}_{group}_{rest}"
        else:
            new_key = f"{modality}_{group}_{key}"

        out[new_key] = value

    return out
