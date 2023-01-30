#!/usr/bin/env python3
import bz2
import os
import sys
import urllib.parse
import capnp

from callogg.util.filereader import FileReader

default_capnp_log = capnp.load(
    os.path.join((os.path.dirname(os.path.realpath(__file__))), "../../msgs/log.capnp")
)
# this is an iterator itself, and uses private variables from LogReader
class MultiLogIterator:
    def __init__(
        self,
        log_paths,
        fs = None,
        sort_by_time=False,
        schema=(
            os.path.join(
                (os.path.dirname(os.path.realpath(__file__))), "../../msgs/log.capnp"
            )
        ),
    ):
        self.capnp_log = capnp.load(schema)
        self._log_paths = log_paths
        self.sort_by_time = sort_by_time
        self.fs = fs
        self._first_log_idx = next(
            i for i in range(len(log_paths)) if log_paths[i] is not None
        )
        self._current_log = self._first_log_idx
        self._idx = 0
        self._log_readers = [None] * len(log_paths)
        self.start_time = self._log_reader(self._first_log_idx)._ts[0]

    def _log_reader(self, i):
        if self._log_readers[i] is None and self._log_paths[i] is not None:
            log_path = self._log_paths[i]
            self._log_readers[i] = LogReader(log_path, fs = self.fs, sort_by_time=self.sort_by_time)
        if i:
            self._log_readers[i-1] = None
        return self._log_readers[i]

    def __iter__(self):
        return self

    def _inc(self):
        lr = self._log_reader(self._current_log)
        if self._idx < len(lr._ents) - 1:
            self._idx += 1
        else:
            self._idx = 0
            self._current_log = next(
                i
                for i in range(self._current_log + 1, len(self._log_readers) + 1)
                if i == len(self._log_readers) or self._log_paths[i] is not None
            )
            if self._current_log == len(self._log_readers):
                self._idx = 0
                raise StopIteration

    def __next__(self):
        while 1:
            lr = self._log_reader(self._current_log)
            ret = lr._ents[self._idx]
            self._inc()
            return ret

    def tell(self):
        # returns seconds from start of log
        return (
            self._log_reader(self._current_log)._ts[self._idx] - self.start_time
        ) * 1e-9

    def seek(self, ts):
        # seek to nearest minute
        minute = int(ts / 60)
        if minute >= len(self._log_paths) or self._log_paths[minute] is None:
            return False

        self._current_log = minute

        # HACK: O(n) seek afterward
        self._idx = 0
        while self.tell() < ts:
            self._inc()
        return True


class LogReader:
    def __init__(
        self,
        fn,
        fs = None,
        canonicalize=True,
        only_union_types=False,
        sort_by_time=False,
        schema=(
            os.path.join(
                (os.path.dirname(os.path.realpath(__file__))), "../../msgs/log.capnp"
            )
        ),
    ):
        self.capnp_log = capnp.load(schema)
        data_version = None
        _, ext = os.path.splitext(urllib.parse.urlparse(fn).path)
        with FileReader(fn, fs=fs) as f:
            dat = f.read()
            # print(ext)
        if ext == "":
            # old rlogs weren't bz2 compressed
            ents = self.capnp_log.Event.read_multiple_bytes(dat)
        elif ext == ".bz2":
            dat = bz2.decompress(dat)
            ents = self.capnp_log.Event.read_multiple_bytes(dat)
        else:
            raise Exception(f"unknown extension {ext}")

        self._ents = list(
            sorted(ents, key=lambda x: x.logMonoTime) if sort_by_time else ents
        )
        self.count = len(self._ents)
        self._ts = [x.logMonoTime for x in self._ents]
        self.data_version = data_version
        self._only_union_types = only_union_types

    def __iter__(self):
        for ent in self._ents:
            if self._only_union_types:
                try:
                    ent.which()
                    yield ent
                except capnp.lib.capnp.KjException:
                    pass
            else:
                yield ent


def flatten(structure, key="", path="", flattened=None, ignore_key=""):
    if flattened is None:
        flattened = {}
    if type(structure) not in (dict, list):
        flattened[((path + ".") if path else "") + key] = structure
    elif isinstance(structure, list):
        if not flattened and all(isinstance(i, dict) for i in structure):
            if not ((path + ".") if path else "") + key:
                flattened["data"] = structure
            else:
                flattened[((path + ".") if path else "") + key] = structure
        elif not any(isinstance(i, (list, dict)) for i in structure):
            flattened[((path + ".") if path else "") + key] = structure
        else:
            for i, item in enumerate(structure):
                flatten(item, "%d" % i, ".".join(filter(None, [path, key])), flattened)
    else:
        for new_key, value in structure.items():
            if new_key == ignore_key and not path:
                flatten(value, "", "", flattened)
            else:
                flatten(value, new_key, ".".join(filter(None, [path, key])), flattened)
    return flattened


if __name__ == "__main__":
    import codecs

    # capnproto <= 0.8.0 throws errors converting byte data to string
    # below line catches those errors and replaces the bytes with \x__
    codecs.register_error("strict", codecs.backslashreplace_errors)
    log_path = sys.argv[1]
    lr = LogReader(log_path, sort_by_time=True)
    import pdb

    import pandas as pd
    from tqdm import tqdm

    print([field for field in dir(default_capnp_log)])
    # df = pd.DataFrame()
    # a = 0
    # capnp_df_dict = dict.fromkeys(default_capnp_log.Event().schema.union_fields)
    # capnp_df_count = dict.fromkeys(default_capnp_log.Event().schema.union_fields)
    # for key in capnp_df_dict.keys():
    #     capnp_df_count[key] = 0

    # k = capnp_df_dict.keys()
    for msg in tqdm(lr):
        try:
            _tmpkey = msg.which()
            print(_tmpkey)
            if _tmpkey == 'can':
                print(msg.to_dict())
        except:
            pass
        # pdb.set_trace()
#         if _tmpkey in k:
# 
#             # _tmp = pd.DataFrame.from_dict(pd.json_normalize(flatten(msg.to_dict()))).set_index('logMonoTime')
#             # capnp_df_dict[msg.which()] = pd.concat([capnp_df_dict[msg.which()], _tmp])
#             # print(capnp_df_dict[msg.which()])
#             capnp_df_dict[_tmpkey].append((flatten(msg.to_dict(), ignore_key=_tmpkey)))
#             capnp_df_count[_tmpkey] += 1
#     for key in tqdm(capnp_df_dict.keys()):
#         if not len(capnp_df_dict[key]):
#             continue
#         capnp_df_dict[key] = pd.json_normalize(capnp_df_dict[key])
#         # print(capnp_df_dict[key])
#         capnp_df_dict[key] = pd.DataFrame.from_dict(capnp_df_dict[key]).set_index(
#             "logMonoTime"
#         )
#     print(capnp_df_dict["clocks"].index)
    # pdb.set_trace()
