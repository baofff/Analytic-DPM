
__all__ = ['merge_dict_', 'merge_dict', 'get_val', 'parse_self_ref_dict', 'single_chain_dict']

import copy
import re
import os
import datetime


################################################################################
# Recursively merge dict
################################################################################

def merge_dict_(dest: dict, src: dict):
    r""" Merge src to dest, inplace
    """
    for k, val in src.items():
        if isinstance(val, dict) and k in dest.keys() and isinstance(dest[k], dict):
            merge_dict_(dest[k], src[k])
        else:
            dest[k] = val


def merge_dict(dest: dict, src: dict):
    r""" Merge src to dest
    """
    dest = copy.deepcopy(dest)
    merge_dict_(dest, src)
    return dest


################################################################################
# Recursively get value in a dict
################################################################################

def get_val(dct: dict, *fields, **kwargs):
    r"""
    Args:
        dct: a dict
        fields: the keys
    """
    try:
        cur = dct
        for field in fields:
            cur = cur[field]
        return cur
    except Exception as e:
        if "default" in kwargs.keys():
            return kwargs["default"]
        else:
            raise e


################################################################################
# Parse a self-reference dict
################################################################################

def _is_reference(s: str):
    return isinstance(s, str) and '$' in s


def _parse_term(term: str, dct_name: str):
    r""" '$(a.b.c.d)' -> "dct_name['a']['b']['c']['d']"
    Args:
        term: a term, e.g., '$(a)', '$(a.b.c.d)'
        dct_name: the name of the dict
    """
    assert isinstance(term, str) and isinstance(dct_name, str)
    keys = term[2:-1].split('.')
    res = ""
    for key in keys:
        res += "['%s']" % key
    return dct_name + res


def _parse_reference(ref: str, dct_name: str):
    r""" '$(a.b) // 10' -> "dct_name['a']['b'] // 10"
    Args:
        ref: a reference, e.g., '$(a.b) // 10', '$(a) + $(b.c)'
        dct_name: the name of the dict
    """
    assert isinstance(ref, str) and isinstance(dct_name, str)
    matches = list(re.finditer("\$\([^$()]*\)", ref))
    for match in matches[::-1]:
        span = match.span()
        term = match.group()
        parsed_term = _parse_term(term, dct_name)
        ref = ref[:span[0]] + parsed_term + ref[span[1]:]
    return ref


def _parse_self_ref_dict(local_dct: dict, global_dct: dict):
    for k, val in local_dct.items():
        if _is_reference(val):
            local_dct[k] = eval(_parse_reference(val, 'global_dct'))
        elif isinstance(val, dict):
            _parse_self_ref_dict(val, global_dct)


def parse_self_ref_dict(dct: dict):
    r""" Parse the auto-reference in a dict
    Only allow one-level reference

    Args:
        dct: a dict which might have self-reference
    """
    dct = copy.deepcopy(dct)
    _parse_self_ref_dict(dct, dct)
    return dct


################################################################################
# Create dict
################################################################################

def single_chain_dict(key: str, val):
    sub_keys = key.split('.')
    for sub_key in sub_keys[::-1]:
        dct = {sub_key: val}
        val = dct
    return dct
