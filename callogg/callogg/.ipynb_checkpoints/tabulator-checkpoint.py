from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from tzlocal import get_localzone
import pdb

class Tabulator:
    @property
    def starttime(self):
        return self._starttime

    @property
    def endtime(self):
        return self._endtime

    @property
    def duration(self):
        return self._duration

    @property
    def date(self):
        return datetime.fromtimestamp(self._starttime)

    @property
    def localdate(self):
        return self.date.astimezone(get_localzone())

    def __init__(self, capnp_reader, mdf_reader, capnp_fields=None):

        self.capnp_reader = capnp_reader
        self._capnp_schema = capnp_reader.capnp_log.Event().schema
        self.mdf_reader = mdf_reader
        self._capnp_fields = (
            capnp_fields if capnp_fields else self._capnp_schema.union_fields
        )
        if not "clocks" in self._capnp_fields:
            self._capnp_fields.append("clocks")
        self._capnp_header = self._capnp_schema.non_union_fields

        self.capnp_df = {}
        self.mdf_df = None

        self._starttime = None
        self._endtime = None

    @staticmethod
    def capnp_to_pandas_ext(capnp_reader, fields):
        if not isinstance(fields, (str, list)):
            raise TypeError(f"Fileds must be list of string")
        elif isinstance(fields, str):
            fields = [fields]
        elif isinstance(fields, list):
            if not all(isinstance(field, str) for field in fields):
                raise TypeError(f"Fileds must be list of string")
        if not "clocks" in fields:
            fields.append("clocks")

        for field in fields:
            if not field in capnp_reader.capnp_log.Event().schema.union_fields:
                fields.remove(field)
                print(f"{field} is not valid field name")

        _tmpdict = {k: [] for k in fields}
        for msg in capnp_reader:
            if msg.which() in fields:
                _tmpkey = msg.which()
            else:
                continue
            try:
                _tmpdict[_tmpkey].append((Tabulator.flatten(msg.to_dict())))
            except Exception as e:
                print(f"Error occurred during capnp_to_pandas with msg {_tmpkey}", e)
        try:
            capnp_df = {
                k: pd.DataFrame.from_dict(pd.json_normalize(_tmpdict[k]))
                for k in fields
                if _tmpdict[k]
            }
        except Exception as e:
            print(f"No valid messages in {capnp_reader._log_paths}", e)

        return capnp_df

    def capnp_to_pandas(self, fields=None):
        if fields:
            if not isinstance(fields, (str, list)):
                raise TypeError(f"Fileds must be list of string")
            elif isinstance(fields, str):
                fields = [fields]
            elif isinstance(fields, list):
                if not all(isinstance(field, str) for field in fields):
                    raise TypeError(f"Fileds must be list of string")
            if not "clocks" in fields:
                fields.append("clocks")

        _tmpdict = (
            {k: [] for k in fields} if fields else {k: [] for k in self._capnp_fields}
        )
        # for msg in tqdm(self.capnp_reader):
        for msg in self.capnp_reader:
            if msg.which() in self._capnp_fields:
                _tmpkey = msg.which()
            else:
                continue
            try:
                _tmpdict[_tmpkey].append((Tabulator.flatten(msg.to_dict())))
            except Exception as e:
                print(f"Error occurred during capnp_to_pandas with msg {_tmpkey}", e)
        try:
            self.capnp_df.update(
                {
                    k: pd.DataFrame.from_dict(pd.json_normalize(_tmpdict[k])).set_index(
                        "logMonoTime"
                    )
                    for k in self._capnp_fields
                    if _tmpdict[k]
                }
            )
            for k, v in self.capnp_df.items():
                if isinstance(v, list):
                    continue
                else:
                    v.sort_index(inplace=True)
        except Exception as e:
            print(f"No valid messages in {self.capnp_reader._log_paths}", e)

        self._capnp_unixtime()

        return self.capnp_df

    def capnp_to_pandas_can(self):
        fields = ['can']
        if fields:
            if not isinstance(fields, (str, list)):
                raise TypeError(f"Fileds must be list of string")
            elif isinstance(fields, str):
                fields = [fields]
            elif isinstance(fields, list):
                if not all(isinstance(field, str) for field in fields):
                    raise TypeError(f"Fileds must be list of string")
            if not "clocks" in fields:
                fields.append("clocks")

        _tmpdict = (
            {k: [] for k in fields} if fields else {k: [] for k in fields}
        )
        for msg in tqdm(self.capnp_reader):
        # for msg in self.capnp_reader:
            if msg.which() in fields:
                _tmpkey = msg.which()
            else:
                continue

            try:
                _tmpdict[_tmpkey].append((Tabulator.flatten(msg.to_dict())))
            except Exception as e:
                print(f"Error occurred during capnp_to_pandas with msg {_tmpkey}", e)
        # try:
        self.capnp_df.update(
            {
                k: pd.DataFrame.from_dict(pd.json_normalize(_tmpdict[k])).set_index(
                    "logMonoTime"
                )
                for k in fields
                if _tmpdict[k]
            }
        )
        for k, v in self.capnp_df.items():
            if isinstance(v, list):
                continue
            else:
                v.sort_index(inplace=True)
        # except Exception as e:
        #     print(e)
        #     print(f"No valid messages in {self.capnp_reader._log_paths}", e)

        self._capnp_unixtime()

        return self.capnp_df

    def _capnp_unixtime(self):
        if not [v for k, v in self.capnp_df.items() if v["valid"].any()]:
            raise ValueError(f"Empty capnp dataframe")
        _walltime = self.capnp_df["clocks"]["clocks.wallTimeNanos"].values
        _logtime = self.capnp_df["clocks"].index.values

        _gaptimelist = _walltime - _logtime
        _meandelta = np.mean(_gaptimelist)
        _jumpindex = np.where(np.abs(_gaptimelist) > 2 * _meandelta)[0].tolist()
        if _jumpindex and max(_jumpindex) + 1 < len(_walltime):
            _walltime = _walltime[max(_jumpindex) + 1 :]
            _logtime = _logtime[max(_jumpindex) + 1 :]

        _gaptime = int(_walltime[-1]) - int(_logtime[-1])

        for v in [v for k, v in self.capnp_df.items() if not v.empty]:
            v.reset_index(inplace=True)
            v.index = 1e-9 * (_gaptime + v["logMonoTime"])
            v.index.name = "Timestamp"
            if not self._starttime and not self._endtime:
                self._starttime = v.index[0]
                self._endtime = v.index[-1]
            else:
                self._starttime = min(self._starttime, v.index[0])
                self._endtime = min(self._endtime, v.index[0])

        self._duration = self._endtime - self._starttime

    def mdf_to_pandas(self):
        self.mdf_df = self.mdf_reader.columnify.to_dataframe(
            use_interpolation=False, time_as_datetime=True, time_from_zero=False
        )

    @staticmethod
    def flatten(structure, key="", path="", flattened=None, ignore_key=""):
        if flattened is None:
            flattened = {}
        if type(structure) not in (dict, list):
            flattened[((path + ".") if path else "") + key] = structure
        elif isinstance(structure, list):
            if not flattened and all(isinstance(i, dict) for i in structure):
                flattened[((path + ".") if path else "") + key] = structure
            elif not any(isinstance(i, (list, dict)) for i in structure):
                flattened[((path + ".") if path else "") + key] = structure
            else:
                for i, item in enumerate(structure):
                    Tabulator.flatten(
                        item, "%d" % i, ".".join(filter(None, [path, key])), flattened
                    )
        else:
            for new_key, value in structure.items():
                if new_key == ignore_key:
                    Tabulator.flatten(value, "", "", flattened)
                else:
                    Tabulator.flatten(
                        value, new_key, ".".join(filter(None, [path, key])), flattened
                    )
        return flattened


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(
        description="Parsing capnp_log into (.csv | .h5 | .parquet) formats"
    )
    parser.add_argument("capnp_logs", help="path to capnp log file or directory")
    parser.add_argument(
        "--capnp_schema",
        help="path to capnp message schema *.capnp",
        default=os.path.join(os.path.real(__file__), "./msgs/log.capnp"),
    )
    parser.add_argument("mf4_logs", help="path to mf4 log file or directory")
    parser.add_argument("--mf4_dbc", help="path to dbc file for can frames", nargs="+")
    parser.add_argument(
        "--dst",
        help="output format ['csv', 'hdf5', 'parquet']",
        default="csv",
        choices=["csv", "hdf5", "parquet"],
    )
    parser.add_argument("--verbose", action="store_true", default=False)
