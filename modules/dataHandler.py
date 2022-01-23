#!/usr/bin/env python
"""
Module for loading different datafiles.
"""
from __future__ import annotations
import warnings
from itertools import compress
from typing import Tuple, List, Any, Iterable, overload
import numpy as np
import datetime
from dataclasses import dataclass, field
import tarfile
from zipfile import ZipFile
import netCDF4
import re
from shapely.geometry import Point      # type: ignore
import pickle
import geopandas as gp
from hydra import initialize, compose
from . import errors

# Util functions
@overload
def spherical_to_cartesian(lon: np.ndarray, lat: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def spherical_to_cartesian(lon: float, lat: float, r: float) -> Tuple[float, float, float]: ...
def spherical_to_cartesian(lon: np.ndarray|float, lat: np.ndarray|float, r: np.ndarray|float) -> Tuple[np.ndarray|float, np.ndarray|float, np.ndarray|float]:
    """Converts spherical (lat, lon, r) to cartesian (x,y,z)"""
    lat = lat*(2*np.pi)/360
    lon = lon*(2*np.pi)/360
    rcos_theta = r * np.cos(lon)
    x = rcos_theta * np.cos(lat)
    y = rcos_theta * np.sin(lat)
    z = r * np.sin(lon)
    return x, y, z

@overload
def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]: ...
@overload
def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def cartesian_to_spherical(x: np.ndarray|float, y: np.ndarray|float, z: np.ndarray|float) -> Tuple[np.ndarray|float, np.ndarray|float, np.ndarray|float]:
    """Converts cartesian (x,y,z) to spherical (lat, lon, r)"""
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    lat = np.arctan2(z, hxy) * 360/(2*np.pi)
    lon = np.arctan2(y, x) * 360/(2*np.pi)
    return lat, lon, r

@overload
def decyear(year: int, month: int, day: int) -> np.ndarray: ...
@overload
def decyear(year: int, month: int, day: int, h: int=0, m:int=0, s: int=0) -> np.ndarray: ...
@overload
def decyear(year: np.ndarray, month: np.ndarray, day: np.ndarray, h: np.ndarray, m:np.ndarray, s: np.ndarray) -> np.ndarray: ...
def decyear(year: int|np.ndarray, month: int|np.ndarray, day: int|np.ndarray, h: int|np.ndarray=0, m:int|np.ndarray =0, s: int|np.ndarray=0) -> np.ndarray:
    refyear = 2015
    leapyr = ((np.r_[year]%4==0) * (np.r_[year]%100!=0) + (np.r_[year]%400==0)).astype(int)
    day_of_year = np.r_[0,31,28,31,30,31,30,31,31,30,31,30,31].cumsum()
    extraday = np.r_[month>2].astype(int)*leapyr 
    year_seconds = 31557600.0 # Weighted average of leap and non-leap years
    seconds_from_ref = ((year-refyear)*year_seconds + (((( day_of_year[month-1]+extraday + day-1)*24+h)*60 + m)*60 +s))
    return refyear + seconds_from_ref/year_seconds

def to_int(*args:str) -> Tuple[int, ...]:
    """ Convert inputs to int"""
    return tuple((int(arg) for arg in args))

# ts_profile from gz
def import_tsprofiles(filename: str, print_exception: bool = False, verbose: bool = False) -> TSProfiles:
    """ Imports tar file and returns a TSProfiles object containing the different profiles"""
    profiles = TSProfiles()
    with tarfile.open(filename, "r:gz") as files:
        # Get each file
        for idx, file in enumerate(files):
            if verbose and idx % 10000 == 0:
                print(idx, end='\r')
            # Checks if the file is a dat file or not
            if '.dat' not in file.name:
                continue
            # Get data from file
            if (file_content := files.extractfile(file)) is not None:
                try:
                    data = file_content.read()
                    content = [
                        [
                            float(cell) for cell in line.replace(' ','').split('\t') if cell != ''
                        ] for line in data.decode("utf-8").split('\n') if line != ''
                    ]
                    profiles.append(
                        TSProfile(
                            time=content[0][0],
                            file_name=file.name,
                            lat=content[0][1],
                            lon=content[0][2],
                            profile=np.array(content[1:], dtype=np.float64)
                        )
                    )
                except ValueError:
                    if print_exception:
                        print(f"\nError in file: '{file.name}'\n")
    return profiles


# Tsprofile classes
@dataclass
class TSProfile:
    """Class for keeping track of a ts profile"""
    time: float
    file_name: str
    lat: float
    lon: float
    profile: np.ndarray
    point: Point = field(init=False)
    profile_length: int = field(init=False)
    depth: np.ndarray = field(init=False)
    temperatur: np.ndarray = field(init=False)
    salinity: np.ndarray = field(init=False)
    contains_data: bool = field(init=False)

    def __post_init__(self):
        # Init
        self.profile_length = len(self.profile)
        self.point = Point(self.lon, self.lat)
        if self.profile_length == 0:
            self.depth = np.array([], dtype=np.float64)
            self.temperatur = np.array([])
            self.salinity = np.array([])
            self.contains_data = False
        else:
            self.depth = self.profile[:,0]
            self.temperatur = self.profile[:,1]
            self.salinity = self.profile[:,2]
            self.contains_data = not np.isnan(self.profile).any()

    def cartesian(self) -> List[float]:
        """ Returns x, y, z togethert in a list"""
        return list(spherical_to_cartesian(self.lat, self.lon, 6356.75))
        
    def latlon(self) -> List[float]:
        """ Returns lat lon together in a list"""
        return [self.lat, self.lon]

    def __len__(self) -> int:
        return self.profile_length

    def __eq__(self, other: TSProfile) -> bool:
        """ Checks if self is equal to another TSProfile"""
        if not isinstance(other, TSProfile):
            raise errors.InvalidType(str(type(other)), str(TSProfile))
        for key in list(self.__annotations__):
            self_var = getattr(self, key)
            other_var = getattr(other, key)
            if isinstance(self_var, np.ndarray) and isinstance(other_var, np.ndarray):
                comparision = (self_var == other_var).all()
            else:
                comparision = self_var == other_var
            if not comparision:
                return False
        return True
    
    def __repr__(self) -> str:
        out_str = f"TSProfile(time={self.time}, file_name={self.file_name}, point=({self.lat}, {self.lon}),"
        out_str += f" profile_length={self.profile_length},"
        if self.contains_data:
            out_str += f" depth=({self.depth[0]:.1e} {self.depth[-1]:.1e}), temperatur=({self.temperatur[0]:.1e} {self.temperatur[-1]:.1e}),"
            out_str += f" salinity=({self.salinity[0]:.1e} {self.salinity[-1]:.1e})"
        out_str += ')'
        return out_str        

@dataclass
class TSProfiles:
    """ Keeps track of all ts profiles"""
    profiles: List[TSProfile] = field(default_factory=list, init=True)
    __check_profiles: bool = True

    def __post_init__(self) -> None:
        if self.__check_profiles:
            self.profiles = [profile for profile in self.profiles if profile.contains_data]

    def append(self, profile: TSProfile) -> None:
        """ Custom append function"""
        if not isinstance(profile, TSProfile):
            raise errors.InvalidType(str(type(profile)), str(TSProfile))
        if profile.contains_data:
            self.profiles.append(profile)

    def __dir__(self) -> Iterable[str]:
        out_dirs = list(super().__dir__())
        if len(self.profiles):
            out_dirs += [x for x in dir(self.profiles[0]) if not x.startswith('__') and not x.startswith('_TSProfile')]
        return out_dirs
    
    def __get_type(self, var: Any) -> str:
        """
        Checks objects type and returns one of the following types:
        - 'int'
        - 'float'
        - 'object'
        """
        if isinstance(var, int):
            return 'int'
        if isinstance(var, float):
            return 'float'
        return 'object'
    
    def __eq__(self, other: TSProfiles) -> bool:
        """ Checks if self is equal to another TSProfiles"""
        if not isinstance(other, TSProfiles):
            raise errors.InvalidType(str(type(other)), str(TSProfiles))
        return all(map(lambda x, y: x == y, self.profiles, other.profiles))
    
    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            pass
        if len(self.profiles) == 0:
            raise errors.NoProfiles("This TSProfiles does not contain any profiles")
        if not hasattr(self.profiles[0], __name):
            raise errors.InvalidAttribute(f"{__name} is not a valid attribute in TSProfile")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            out = getattr(self.profiles[0], __name)
            if callable(out):
                return np.array([getattr(profile, __name)() for profile in self.profiles], dtype=self.__get_type(out()))
            return np.array([getattr(profile, __name) for profile in self.profiles], dtype=self.__get_type(out))

    def __len__(self) -> int:
        return len(self.profiles)
    
    def __getstate__(self) -> dict:
        return self.__dict__
    
    @overload
    def __getitem__(self, items: int) -> TSProfile: ...
    @overload
    def __getitem__(self, items: Tuple[int]) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: Tuple[bool]) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: List[int]) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: List[bool]) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: np.ndarray) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: slice) -> TSProfiles: ...
    @overload
    def __getitem__(self, items: range) -> TSProfiles: ...
    def __getitem__(self, items: int | Tuple[int] | Tuple[bool] | List[int] | List[bool] | np.ndarray | slice | range) -> TSProfile | TSProfiles:
        """ Custom getitem function y[items] -> TSProfiles"""
        # Check for numpy
        if isinstance(items, (np.ndarray, range)):
            items = list(items)
        if isinstance(items, int):
            return self.profiles[items]
        if isinstance(items, slice):
            return TSProfiles(self.profiles[items], False)
        if isinstance(items, (list, tuple)):
            # Check for list of ndarrays
            if isinstance(items[0], (np.ndarray, list)):
                # Takes list of bool array for the different profiles and cuts the profile to the bool array
                if isinstance(items[0][0], (bool, np.bool_)):
                    tsprofile = self.__copy__()
                    keys = [key for key, item in TSProfile.__annotations__.items() if item == 'np.ndarray']
                    for key in keys:
                        for profile, bool_arr in zip(tsprofile, items): # type: ignore
                            setattr(profile, key, getattr(profile, key)[bool_arr])

                    for profile in tsprofile: # type: ignore
                        if isinstance(profile, TSProfiles):
                            continue
                        profile.profile_length = len(profile.profile)
                    return tsprofile

            # Check for bool array
            if all([x==True or x==False for x in set(items)]) and len(items):
                if len(self) != len(items):
                    raise errors.InvalidNumberOfItem(len(items), len(self))
                return TSProfiles(list(compress(self.profiles, items)), False)
            # Check for integer array
            elif len([i for i in items if not isinstance(i,int)]) == 0:
                return TSProfiles([self.profiles[i] for i in items], False)
        raise errors.InvalidItem(f"{type(items)} is not a valid type")
    
    def __repr__(self) -> str:
        max_len = 7
        out_str = "TSProfiles(\n\t"
        out_str += ",\n\t".join([repr(profile) for profile in self.profiles[:max_len]])
        if len(self.profiles) > max_len:
            out_str += ",\n\t..."
            out_str += ",\n\t..."
            out_str += ",\n\t..."
        out_str += "\n)"
        return out_str
    
    def __copy__(self) -> TSProfiles:
        """ Return a copy of the object"""
        return TSProfiles(self.profiles[:], False)


# Load datasets
def load_tsprofiles(ts_profiles_path: str) -> TSProfiles:
    """ Loads pickle file containing TSProfiles."""
    with open(ts_profiles_path, 'rb') as handle:
        ts_profiles = TSProfiles(**pickle.load(handle))
    return ts_profiles

def clean_ts_profiles(ts_profiles: TSProfiles) -> TSProfiles:
    """ Cleans the ts profiles"""
    # Remove data profiles with less then 5 data points
    bool_length: np.ndarray = ts_profiles.profile_length > 5
    ts_profiles = ts_profiles[bool_length]
    # Remove salinity out of range (These values dont make sence)
    ts_profiles = ts_profiles[[((salinity >= 0) & (salinity <= 100)).all() for salinity in ts_profiles.salinity]]

    # Remove temperatur out of range (These values dont make sence)
    ts_profiles = ts_profiles[[((temperatur >= -100) & (temperatur <= 100)).all() for temperatur in ts_profiles.temperatur]]

    # Use only data within 0-100m
    ts_profiles = ts_profiles[[((depth >= 0) & (depth <= 100)) for depth in ts_profiles.depth]]
    bool_length: np.ndarray = ts_profiles.profile_length > 5
    ts_profiles = ts_profiles[bool_length]
    return ts_profiles

def load_bathymetri(bathymetri_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a Network Common Data Form (NetCDF) file (.nc) containing bathymetri data,
    with the following fields:
        lat, lon, bedrock_topography
    """
    bathymetri = netCDF4.Dataset(bathymetri_path) # type: ignore
    bathymetri_lat: np.ndarray = bathymetri['lat'][:].data
    bathymetri_lon: np.ndarray = bathymetri['lon'][:].data
    bathymetri_topography: np.ndarray = bathymetri['bedrock_topography'][:].data[bathymetri_lat > 60, :]
    bathymetri_lat: np.ndarray = bathymetri_lat[bathymetri_lat > 60]
    return bathymetri_lat, bathymetri_lon, bathymetri_topography

def load_ghrsst(ghrsst_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a Network Common Data Form (NetCDF) file (.nc) containing ghrsst data,
    with the following fields:
        lat, lon, sst, time, time_bnds
    """
    ghrsst = netCDF4.Dataset(ghrsst_path) # type: ignore
    ghrsst_lat: np.ndarray = ghrsst['lat'][:].data
    ghrsst_lon: np.ndarray = ghrsst['lon'][:].data
    ghrsst_sst: np.ndarray = ghrsst['sst'][:].data
    ghrsst_times: np.ndarray = ghrsst['time'][:].data
    ghrsst_time_bnds: np.ndarray = ghrsst['time_bnds'][:].data
    ghrsst_sst: np.ndarray = ghrsst_sst[:,ghrsst_lat >= 60,:]
    ghrsst_lat: np.ndarray = ghrsst_lat[ghrsst_lat >= 60]
    ghrsst_fraction_time = []
    for ghrsst_time in ghrsst_times: 
        ghrsst_date = datetime.date(1800,1,1) + datetime.timedelta(ghrsst_time)
        ghrsst_fraction_time.append(decyear(ghrsst_date.year, ghrsst_date.month, ghrsst_date.day)[0])
    ghrsst_fraction_time = np.array(ghrsst_fraction_time)

    return ghrsst_lat, ghrsst_lon, ghrsst_sst, ghrsst_times, ghrsst_fraction_time, ghrsst_time_bnds

def load_smos(smos_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a zip file containing multiple Network Common Data Form (NetCDF) 
    file (.nc) which contain smos data, with the following fields:
        lat, lon, sss, time_coverage_center
    """
    # lat, lon, sss
    smos_data = []
    with ZipFile(smos_path, 'r') as archive:
        files = archive.filelist
        for file in files:
            if not file.filename.endswith('.nc'):
                continue
            smos_data.append(netCDF4.Dataset('inmemory.nc', memory=archive.read(file))) # type: ignore

    smos_lat, smos_lon, smos_sss, smos_time = [], [], [], []
    for d in smos_data:
        time = to_int(*re.split('-|T|:', d.time_coverage_center))
        time_decyear = decyear(*time)[0] # type: ignore
        smos_lat.append(d['lat'][:].data)
        smos_lon.append(d['lon'][:].data)
        smos_sss.append(d['sss'][:].data)
        smos_time.append(time_decyear)

    smos_sss, smos_time = np.array(smos_sss), np.array(smos_time)
    smos_lat, smos_lon = np.array(smos_lat), np.array(smos_lon)
    return smos_sss, smos_time, smos_lat, smos_lon

def load_all(verbose: bool = True) -> Tuple[
    gp.GeoDataFrame, TSProfiles, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """ Loads all the data using hydra"""
    with initialize(config_path="./config"):
        cfg = compose(config_name="setup")
    # Region
    if verbose:
        print("Regions [0/4]" + " "*10, end='\r')
    regions = gp.read_file(cfg.REGIONS_PATH)
    if not isinstance(regions, gp.GeoDataFrame):
        raise errors.InvalidFile(cfg.REGIONS_PATH)

    # TSProfiles
    if verbose:
        print("TSProfiles [1/4]" + " "*10, end='\r')
    if cfg.LOAD_TS_PROFILE_BACKUP:
        with open(cfg.BACKUP, 'rb') as handle:
            ts_profiles = TSProfiles(**pickle.load(handle))
    else:
        ts_profiles = load_tsprofiles(cfg.TS_PROFILE_PATH)
    ts_profiles = clean_ts_profiles(ts_profiles)

    if cfg.SAVE_TS_PROFILE_BACKUP:
        with open(cfg.BACKUP, 'wb') as handle:
            pickle.dump(ts_profiles.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Bathymetri
    if verbose:
        print("Bathymetri [2/4]" + " "*10, end='\r') 
    bathymetri_lat, bathymetri_lon, bathymetri_topography = load_bathymetri(cfg.BATHYMETRI)

    # GHRSST
    if verbose:
        print("GHRSST [3/4]" + " "*10, end='\r')
    ghrsst_lat, ghrsst_lon, ghrsst_sst, ghrsst_times, ghrsst_fraction_time, ghrsst_time_bnds = load_ghrsst(cfg.GHRSST_PATH)

    # SMOS
    if verbose:
        print("SMOS [4/4]" + " "*10, end='\r')
    smos_sss, smos_time, smos_lat, smos_lon = load_smos(cfg.BCN_SMOS_PATH)

    return (
        regions,
        ts_profiles,
        bathymetri_lat, bathymetri_lon, bathymetri_topography,
        ghrsst_lat, ghrsst_lon, ghrsst_sst, ghrsst_times, ghrsst_fraction_time, ghrsst_time_bnds,
        smos_sss, smos_time, smos_lat, smos_lon
    )



if __name__ == '__main__':
    (
        regions,
        ts_profiles,
        bathymetri_lat, bathymetri_lon, bathymetri_topography,
        ghrsst_lat, ghrsst_lon, ghrsst_sst, ghrsst_times, ghrsst_fraction_time, ghrsst_time_bnds,
        smos_sss, smos_time, smos_lat, smos_lon
    ) = load_all(verbose=True)