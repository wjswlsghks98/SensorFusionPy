import logging
import os

import numpy as np
import pandas as pd
import pyproj

logger = logging.getLogger("callogg.util.mapreader")
formatter = logging.Formatter(
    fmt="{asctime} - {name:10s} [{levelname:^7s}] {message}",
    style="{",
    datefmt="%m/%d/%Y %H:%M:%S",
)
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

if os.path.isfile(os.path.join(os.getcwd(), ".token")):
    with open(os.path.join(os.getcwd(), ".token")) as f:
        api_key = f.read()
else:
    api_key = None


class MapHelper:

    __COORD_ECEF = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
    __COORD_LLA = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
    __api_key = None

    def __init__(self, **kwargs):
        """
        Road Snap Options
        - annotations = ["distance", "duration", "speed", "congestion", "congestion_numeric", "maxspeed"]
        - approaches = ["unrestricted"(default), "curb"]
        - geometries = ["geojson", "polyline"(defualt), "polyline6"]
        - language = ["en"(defualt), "es", "fr", "de", "it", "pt", "ru", "ja", "ko"]
        - overview = ["full"(default), "simplified", "false"]
        - radiuses = between 0.0-50.0 (default 5.0)
        - steps = [True, False(default)]
        - tidy = [True, False(default)]
        """
        if "api_key" in kwargs.keys():
            with open(os.path.join(os.getcwd(), ".token"), "w") as f:
                f.write(kwargs["api_key"])
            self.__api_key = kwargs["api_key"]
        elif api_key:
            self.__api_key = api_key
        else:
            logger.warning(f"API key not specified")

        self.__snap = Snap()

    @property
    def api_key(self):
        return self.__api_key

    @api_key.setter
    def api_key(self, argv):
        with open(os.path.join(os.getcwd(), ".token"), "w") as f:
            f.write(argv)
        self.__api_key = argv

    @property
    def snap(self):
        return self.__snap

    @staticmethod
    def ecef_to_lla(
        ecef_x: pd.core.series.Series,
        ecef_y: pd.core.series.Series,
        ecef_z: pd.core.series.Series,
    ) -> tuple[pd.core.series.Series]:
        __transform = pyproj.Transformer.from_proj(
            MapHelper.__COORD_ECEF, MapHelper.__COORD_LLA
        )
        lon, lat, alt = __transform.transform(ecef_x, ecef_y, ecef_z)
        return (lon, lat, alt)

    @staticmethod
    def lla_to_ecef(
        lon: pd.core.series.Series,
        lat: pd.core.series.Series,
        alt: pd.core.series.Series,
    ) -> tuple[pd.core.series.Series]:
        __transform = pyproj.Transformer.from_proj(
            MapHelper.__COORD_LLA, MapHelper.__COORD_ECEF
        )
        ecef_x, ecef_y, ecef_z = __transform.transform(lon, lat, alt)
        return (ecef_x, ecef_y, ecef_z)


class Snap:
    __MATCHING_OPTIONS = [
        "annotations",
        "approaches",
        "geometries",
        "language",
        "overview",
        "radiuses",
        "steps",
        "tidy",
    ]
    __MATCHING_ANNOATATIONS = [
        "distance",
        "duration",
        "speed",
        "congestion",
        "congestion_numeric",
        "maxspeed",
    ]
    __MATCHING_APPROACHES = ["unrestricted", "curb"]
    __MATCHING_GEOMETRIES = ["geojson", "polyline", "polyline6"]
    __MATCHING_LANGUAGE = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko"]
    __MATCHING_OVERVIEW = ["full", "simplified", "false"]
    __MATCHING_STEPS = bool
    __MATCHING_RADIUSES = float
    __MATCHING_TIDY = bool
    __query_params = ""
    __query_url = "https://api.mapbox.com/matching/v5/mapbox/driving?"
    __options = dict.fromkeys(__MATCHING_OPTIONS)

    def __init__(self, **kwargs):
        for k in [key for key in kwargs.keys() if key in Snap.__MATCHING_OPTIONS]:
            self.__options[k] = kwargs.pop(k)
        if kwargs:
            raise NameError(f"Unknown option argument {kwargs.keys()}")

    @property
    def options(self):
        return self.__options

    @property
    def annotations(self):
        return self.__options["annotations"]

    @property
    def approaches(self):
        return self.__options["approaches"]

    @property
    def geometries(self):
        return self.__options["geometries"]

    @property
    def language(self):
        return self.__options["language"]

    @property
    def overview(self):
        return self.__options["overview"]

    @property
    def steps(self):
        return self.__options["steps"]

    @property
    def tidy(self):
        return self.__options["tidy"]

    @property
    def radiuses(self):
        return self.__options["radiuses"]

    def set_options(self, **kwargs):
        for (k, v) in [
            (k, v) for (k, v) in kwargs.items() if k in Snap.__MATCHING_OPTIONS
        ]:
            self.__options[k] = v

    @annotations.setter
    def annotations(self, argv):
        self.__options["annotations"] = []
        if isinstance(argv, list):
            for arg in argv:
                if arg in Snap.__MATCHING_ANNOATATIONS:
                    self.__options["annotations"].append(arg)
                else:
                    raise ValueError(
                        f"Invalid option: {argv} must be members of {Snap.__MATCHING_ANNOATATIONS}"
                    )

        if isinstance(argv, str):
            if argv in Snap.__MATCHING_ANNOATATIONS:
                self.__options["annotations"] = [argv]
        else:
            raise ValueError(
                f"Invalid option: {argv} must be members of {Snap.__MATCHING_ANNOATATIONS}"
            )

    @approaches.setter
    def approaches(self, argv):
        if argv in Snap.__MATCHING_APPROACHES:
            self.__options["approaches"] = argv
        else:
            raise ValueError(
                f"Invalid option: {argv} must be members of {Snap.__MATCHING_APPROACHES}"
            )

    @geometries.setter
    def geometries(self, argv):
        if argv in Snap.__MATCHING_GEOMETRIES:
            self.__options["geometries"] = argv
        else:
            raise ValueError(
                f"Invalid option: {argv} must be members of {Snap.__MATCHING_GEOMETRIES}"
            )

    @language.setter
    def language(self, argv):
        if argv in Snap.__MATCHING_LANGUAGE:
            self.__options["language"] = argv
        else:
            raise ValueError(
                f"Invalid option: {argv} must be members of {Snap.__MATCHING_LANGUAGE}"
            )

    @overview.setter
    def overview(self, argv):
        if argv in Snap.__MATCHING_OVERVIEW:
            self.__options["overview"] = argv
        else:
            raise ValueError(
                f"Invalid option: {argv} must be members of {Snap.__MATCHING_OVERVIEW}"
            )

    @steps.setter
    def steps(self, argv):
        if isinstance(argv, Snap.__MATCHING_STEPS):
            self.__options["steps"] = argv
        else:
            raise TypeError(
                f"Invalid option: {argv} must be a type {Snap.__MATCHING_STEPS}"
            )

    @tidy.setter
    def tidy(self, argv):
        if isinstance(argv, Snap.__MATCHING_TIDY):
            self.__options["tidy"] = argv
        else:
            raise TypeError(
                f"Invalid option: {argv} must be a type {Snap.__MATCHING_TIDY}"
            )

    @radiuses.setter
    def radiuses(self, argv):
        if isinstance(argv, Snap.__MATCHING_RADIUSES):
            self.__options["radiuses"] = argv
        else:
            raise TypeError(
                f"Invalid option: {argv} must be a type {Snap.__MATCHING_RADIUSES}"
            )

    def __add_annotation_option(self, query_param=""):
        if self.__options["annotations"]:
            query_param += "&annotations="
            if isinstance(self.__options["annotations"], list):
                for annotation in self.__options["annotations"]:
                    if not isinstance(annotation, str):
                        raise ValueError(f"Invalid option: annotation={annotation}")
                    query_param += annotation
                    query_param += ","
                query_param = query_param[:-1]
            elif isinstance(self.__options["annotations"], str):
                query_param += f"&annotations={self.__options['annotations']}"
            else:
                raise NameError(
                    f"Invalid option: annotation={self.__options['annotations']}"
                )

    def __add_single_option(self, key, query_param=""):
        if key in self.__options.keys():
            if key == "annotations":
                query_param += self.__add_annotation_option(self)
            if isinstance(self.__options["key"], str):
                query_param = f"&{key}={self.__options[key]}"
            else:
                raise ValueError(f"Invalid option: {key}={self.__options[key]}")
        else:
            raise NameError(f"Unknown option: {key}")
        return query_param

    def __add_list_option(self, entry, key, query_param=""):
        if key in self.__options.keys():
            if isinstance(self.__options[key], str):
                query_param += f"&{key}="
                for i in range(len(entry)):
                    query_param += f"{self.__options[key]};"
                query_param = query_param[:-1]
            else:
                raise ValueError(f"Invalid option: {key}={self.__options[key]}")
        else:
            raise NameError(f"Unknown option: {key}")
        return query_param

    def __add_entry(self, entry):
        if not ("longitude" in entry.keys() and "latitude" in entry.keys()):
            raise KeyError(
                f"Invalid entry: Could not find latitude and longitude in {entry.keys()}"
            )
        if len(entry["longitude"]) != len(entry["latitude"]):
            raise ValueError(
                f"Shape mismatch: longitude={len(entry['longitude'])} latitude={len(entry['latitude'])}"
            )
        if not isinstance(entry["longitude"], list) and isinstance(
            entry["latitude"], list
        ):
            entry["longitude"] = [entry["longitude"]]
            entry["latitude"] = [entry["latitude"]]
        if len(entry["longitude"] == 1):
            entry["longitude"].append(entry["longitude"])
        if len(entry["latitude"] == 1):
            entry["latitude"].append(entry["latitude"])
        query_entry = "coordinates="
        for lon, lat in zip(entry["longitude"], entry["latitude"]):
            query_entry += f"{lon},{lat},"
        if "timestamp" in entry.keys():
            if len(entry["timestamp"]) != len(entry["longitude"]):
                raise ValueError(
                    f"Shape mismatch: timestamp={len(entry)} latitude/longitude={len(entry['latitude'])}"
                )
            query_entry += "&timestamps="
            for timestamp in entry["timestamps"]:
                query_entry += f"{timestamp};"
            query_entry = query_entry[:-1]
        return query_entry[:-1]

    def snap_to_road(
        self,
        latitude: pd.core.series.Series,
        logitude: pd.core.series.Series,
        timestamp: pd.core.series.Series | None = None,
        **kwargs,
    ) -> tuple[pd.core.series.Series]:
        # TODO
        return
