import logging
import os
import sys
from glob import glob

# Custom Packages Start
# Refer to github.com/3secondz-lab
import capnp
from mdfcolumnify import MdfColumnify

from .tabulator import Tabulator
from .util.logreader import LogReader, MultiLogIterator
from .util.mapreader import MapHelper

# Custom Packages End


logger = logging.getLogger("callogg")
formatter = logging.Formatter(
    fmt="{asctime} - {name:10s} [{levelname:^7s}] {message}",
    style="{",
    datefmt="%m/%d/%Y %H:%M:%S",
)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)


class Callogg:
    def __init__(
        self,
        capnp_log: str | list | None = None,
        capnp_schema: str | None = None,
        mdf_log: str | list | None = None,
        mdf_dbc: str | list | None = None,
    ):

        if capnp_log and capnp_schema:
            self.__capnp_reader = self.__capnp_load(capnp_log, capnp_schema)

        if mdf_log:
            self.__mdf_reader = self.__mdf_load(mdf_log, mdf_dbc)

        if not self.__capnp_reader == None:
            self.__tabulator = Tabulator(self.__capnp_reader, self.__mdf_reader))

    def __capnp_load(
        self, capnp_log: str | list | None = None, capnp_schema: str | None = None
    ):
        if isinstance(capnp_log, str):
            if os.path.isabs(capnp_log):
                _capnp_log = capnp_log
            else:
                _capnp_log = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), capnp_log
                )

            if os.path.isfile(_capnp_log):
                _capnp_log_list = [_capnp_log]
            elif os.path.isdir(_capnp_log):
                _capnp_log_list = glob(os.path.join(_capnp_log, "*.bz2"))
                if not _capnp_log_list:
                    raise FileNotFoundError(
                        f"No capnp_log found in directory {_capnp_log}"
                    )
            else:
                raise FileNotFoundError(f"{capnp_log} is not a valid File or Directory")

        elif isinstance(capnp_log, list):
            _capnp_log_list = []
            for log in capnp_log:
                if not os.path.isabs(log):
                    log = os.path.join(os.path.dirname(os.path.realpath(__file__)), log)
                if not os.path.isfile(log):
                    raise FileNotFoundError(f"{log} is not a valid File or Directory")
                _capnp_log_list.append(log)

        if isinstance(capnp_schema, str):
            if os.path.isabs(capnp_schema):
                _capnp_schema = capnp_schema
            else:
                _capnp_schema = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), capnp_schema
                )

        if _capnp_log_list and os.path.isfile(_capnp_schema):
            try:
                _capnp_reader = MultiLogIterator(
                    _capnp_log_list, sort_by_time=True, schema=_capnp_schema
                )
                return _capnp_reader
            except Exception as e:
                print(
                    f"Cannot open capnp_log {_capnp_log_list} with schema {_capnp_schema}",
                    e
                )
                return None

    def __mdf_load(
        self, mdf_log: str | list | None = None, mdf_dbc: str | list | None = None
    ):
        if isinstance(mdf_log, str):
            if os.path.isabs(mdf_log):
                _mdf_log = mdf_log
            else:
                _mdf_log = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), mdf_log
                )

        _mdf_dbc = []
        if isinstance(mdf_dbc, str):
            if os.path.isabs(mdf_dbc):
                _mdf_dbc.append(mdf_dbc)
            else:
                _mdf_dbc.append(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), mdf_dbc)
                )
        if isinstance(mdf_dbc, list):
            for dbc in mdf_dbc:
                if os.path.isabs(mdf_dbc):
                    _mdf_dbc.append(dbc)
                else:
                    _mdf_dbc.append(
                        os.path.join(os.path.dirname(os.path.realpath(__file__)), dbc)
                    )

        try:
            if mdf_dbc:
                _mdf_reader = MdfColumnify(mdf_log, dbc_list=_mdf_dbc)
            else:
                _mdf_reader = MdfColumnify(mdf_log)

            return _mdf_reader
        except Exception as e:
            print("Cannot open mdf_log {mdf_log} with dbc_file {mdf_dbc}", e)
            return None

    def export(self, dst, fmt):
        
        dst = os.path.join(os.path.getcwd(),'output') if not dst else None
        fmt = "all" if not fmt else None

        if not hasattr(self, '__tabulator'):
            logger.error(f"Log file not imported")
            return False

        if not self.__tabulator.capnp_reader == None:
            self.__tabulator.capnp_to_pandas
        else:
            logger.info("Capnp log not imported")
        if not self.__tabulator.mdf_reader == None:
            self.__tabulator.mdf_to_pandas
        else:
            logger.info("MDF log not imported")

        
