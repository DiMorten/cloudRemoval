"""
    Generic data loading routines for the SEN12MS-CR dataset of corresponding Sentinel 1,
    Sentinel 2 and cloudy Sentinel 2 data.

    The SEN12MS-CR class is meant to provide a set of helper routines for loading individual
    image patches as well as triplets of patches from the dataset. These routines can easily
    be wrapped or extended for use with many deep learning frameworks or as standalone helper 
    methods. For an example use case please see the "main" routine at the end of this file.

    NOTE: Some folder/file existence and validity checks are implemented but it is 
          by no means complete.

    Authors: Patrick Ebel (patrick.ebel@tum.de), Lloyd Hughes (lloyd.hughes@tum.de),
    based on the exemplary data loader code of https://mediatum.ub.tum.de/1474000, with minimal modifications applied.
"""

import os
# import rasterio

import numpy as np

from enum import Enum
from glob import glob
import rasterio
import pdb
#from icecream import ic

s1_list = []
s2_list = []
s2_cloudy_list = []

class S1Bands(Enum):
    VV = 1
    VH = 2
    ALL = [VV, VH]
    NONE = []


class S2Bands(Enum):
    B01 = aerosol = 1
    B02 = blue = 2
    B03 = green = 3
    B04 = red = 4
    B05 = re1 = 5
    B06 = re2 = 6
    B07 = re3 = 7
    B08 = nir1 = 8
    B08A = nir2 = 9
    B09 = vapor = 10
    B10 = cirrus = 11
    B11 = swir1 = 12
    B12 = swir2 = 13
    ALL = [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
    RGB = [B04, B03, B02]
    NONE = []


class Seasons(Enum):
    SPRING = "ROIs1158_spring"
    SUMMER = "ROIs1868_summer"
    FALL = "ROIs1970_fall"
    WINTER = "ROIs2017_winter"
    ALL = [SPRING, SUMMER, FALL, WINTER]


class Sensor(Enum):
    s1 = "s1"
    s2 = "s2"
    s2_cloudy = "s2_cloudy"

# Note: The order in which you request the bands is the same order they will be returned in.


class SEN12MSCRDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir

        if not os.path.exists(self.base_dir):
            raise Exception(
                "The specified base_dir for SEN12MS-CR dataset does not exist")

    """
        Returns a list of scene ids for a specific season.
    """

    def get_scene_ids(self, season):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season + "_s1")

        if not os.path.exists(path):
            raise NameError("Could not find season {} in base directory {}".format(
                season, self.base_dir))        

        scene_list = [os.path.basename(s) for s in glob(os.path.join(path, "*"))]

        scene_list = [int(s.split("_")[1]) for s in scene_list]

        return set(scene_list)

    """
        Returns a list of patch ids for a specific scene within a specific season
    """

    def get_patch_ids(self, season, scene_id):
        season = Seasons(season).value
        path = os.path.join(self.base_dir, season+"_s1", f"s1_{scene_id}")

        if not os.path.exists(path):
            raise NameError(
                "Could not find scene {} within season {}".format(scene_id, season))

        patch_ids = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob(os.path.join(path, "*"))]
        patch_ids = [int(p.rsplit("_", 1)[1].split("p")[1]) for p in patch_ids]

        return patch_ids

    """
        Return a dict of scene ids and their corresponding patch ids.
        key => scene_ids, value => list of patch_ids
    """

    def get_season_ids(self, season):
        season = Seasons(season).value

        ids = {}
        scene_ids = self.get_scene_ids(season)

        for sid in scene_ids:
            ids[sid] = self.get_patch_ids(season, sid)

        return ids

    """
        Returns raster data and image bounds for the defined bands of a specific patch
        This method only loads a sinlge patch from a single sensor as defined by the bands specified
    """

    def get_patch(self, season, sensor, scene_id, patch_id, bands):
        season = Seasons(season).value
        sensor = sensor.value

        if isinstance(bands, (list, tuple)):
            bands = [b.value for b in bands]
        else:
            bands = bands.value
        
        filename = "{}_{}_{}_p{}.tif".format(season, sensor, scene_id, patch_id)
        patch_path = os.path.join(self.base_dir, "{}_{}".format(season, sensor), "{}_{}".format(sensor, scene_id), filename)
        
        if sensor == 's1':
            s1_list.append(patch_path)
        elif sensor == 's2':
            s2_list.append(patch_path)
        elif sensor == 's2_cloudy':
            s2_cloudy_list.append(patch_path)

        with rasterio.open(patch_path) as patch:
            data = patch.read(bands)
            bounds = patch.bounds

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)

        return data, bounds


    def get_s1s2(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL):
        
        s1, bounds = self.get_patch(season, Sensor.s1, scene_id, patch_id, s1_bands)
        s2, _ = self.get_patch(season, Sensor.s2, scene_id, patch_id, s2_bands)

        return s1, s2, bounds

    """
        Returns a triplet of patches. S1, S2 and cloudy S2 as well as the geo-bounds of the patch
    """

    def get_s1s2s2cloudy_triplet(self, season, scene_id, patch_id, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        s1, bounds = self.get_patch(season, Sensor.s1, scene_id, patch_id, s1_bands)
        s2, _ = self.get_patch(season, Sensor.s2, scene_id, patch_id, s2_bands)
        s2_cloudy, _ = self.get_patch(season, Sensor.s2_cloudy, scene_id, patch_id, s2cloudy_bands)

        return s1, s2, s2_cloudy, bounds

    """
        Returns a triplet of numpy arrays with dimensions D, B, W, H where D is the number of patches specified
        using scene_ids and patch_ids and B is the number of bands for S1, S2 or cloudy S2
    """

    def get_triplets(self, season, scene_ids=None, patch_ids=None, s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL):
        season = Seasons(season)
        scene_list = []
        patch_list = []
        bounds = []
        s1_data = []
        s2_data = []
        s2cloudy_data = []

        # This is due to the fact that not all patch ids are available in all scenes
        # And not all scenes exist in all seasons
        if isinstance(scene_ids, list) and isinstance(patch_ids, list):
            raise Exception("Only scene_ids or patch_ids can be a list, not both.")

        if scene_ids is None:
            scene_list = self.get_scene_ids(season)
        else:
            try:
                scene_list.extend(scene_ids)
            except TypeError:
                scene_list.append(scene_ids)

        if patch_ids is not None:
            try:
                patch_list.extend(patch_ids)
            except TypeError:
                patch_list.append(patch_ids)

        for sid in scene_list:
            if patch_ids is None:
                patch_list = self.get_patch_ids(season, sid)

            for pid in patch_list:
                s1, s2, s2cloudy, bound = self.get_s1s2s2cloudy_triplet(
                    season, sid, pid, s1_bands, s2_bands, s2cloudy_bands)
                s1_data.append(s1)
                s2_data.append(s2)
                s2cloudy_data.append(s2cloudy)
                bounds.append(bound)

        return np.stack(s1_data, axis=0), np.stack(s2_data, axis=0), np.stack(s2cloudy_data, axis=0), bounds


if __name__ == "__main__":
    import time
    # Load the dataset specifying the base directory
#    sen12mscr = SEN12MSCRDataset("D:\Javier\Repo_Noa\SAR2Optical Project\Datasets\SEN2MS-CR")
    sen12mscr = SEN12MSCRDataset("D:/jorg/phd/fifth_semester/project_forestcare/cloud_removal/dataset")


    data, set_ids = [], []
    
    seasons = [Seasons.SUMMER]
    for season in seasons:

        scene_ids = sen12mscr.get_scene_ids(season)
        scene_ids = list(np.sort(list(scene_ids)))
        
        for i, sid in zip(range(len(scene_ids)), scene_ids):

            if i < len(scene_ids) * .07: d = 1     # Validation ROIs   (10% of the training set)
            elif i < len(scene_ids) * .7: d = 0    # Train ROIs        (70% of the dataset)
            else: d = 2                            # Test ROIs         (30% of the dataset)

            patches = sen12mscr.get_patch_ids(season, sid)

            for patch in patches:
                data.append([season, sid, patch])
                set_ids.append(d) 
        

    set_ids = np.asarray(set_ids)
    train_patches = np.argwhere(set_ids == 0)
    val_patches = np.argwhere(set_ids == 1)
    test_patches = np.argwhere(set_ids == 2)
    print("Training patches: ", train_patches.shape[0])
    print("Validation patches: ", val_patches.shape[0])
    print("Test patches: ", test_patches.shape[0])
    print(set_ids.shape[0], len(data))
    
    # Co-registered images in the triplet
    idx = np.arange(len(data)); np.random.shuffle(idx)
    for i in idx:
        s1, s2, s2_cloudy, bounds = sen12mscr.get_s1s2s2cloudy_triplet(data[i][0], data[i][1], data[i][2], s1_bands=S1Bands.ALL, s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)
        print(data[i])
        print(s1.shape)
        print(s2.shape)
        print(s2_cloudy.shape)
        print(bounds)
        break


    summer_ids = sen12mscr.get_season_ids(Seasons.SUMMER)
    cnt_patches = sum([len(pids) for pids in summer_ids.values()])
    print("Summer: {} scenes with a total of {} patches".format(
        len(summer_ids), cnt_patches))

#    pdb.set_trace()        
    start = time.time()
    '''
    # Load the RGB bands of the first S2 patch in scene 11
    SCENE_ID = 11
    s2_rgb_patch, bounds = sen12mscr.get_patch(Seasons.SUMMER, Sensor.s2, SCENE_ID, 
                                            summer_ids[SCENE_ID][0], bands=S2Bands.RGB)
    print("Time Taken {}s".format(time.time() - start))
                                            
    print("S2 RGB: {} Bounds: {}".format(s2_rgb_patch.shape, bounds))

    print("\n")

    # Load a triplet of patches from the first three scenes of Summer - all S1 bands, NDVI S2 bands, and NDVI S2 cloudy bands
    i = 0
    start = time.time()
    for scene_id, patch_ids in summer_ids.items():
        if i >= 3:
            break

        s1, s2, s2cloudy, bounds = sen12mscr.get_s1s2s2cloudy_triplet(Seasons.SUMMER, scene_id, patch_ids[0], s1_bands=S1Bands.ALL,
                                                        s2_bands=[S2Bands.red, S2Bands.nir1], s2cloudy_bands=[S2Bands.red, S2Bands.nir1])
        print(
            f"Scene: {scene_id}, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}, Bounds: {bounds}")
        i += 1

    print("Time Taken {}s".format(time.time() - start))
    print("\n")
    '''

    start = time.time()
    # Load all bands of all patches in a specified scene (scene 15)
    '''
    scene = 17
    s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.SPRING, scene, s1_bands=S1Bands.ALL, 
                                        s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)
    

    scene = 39
    s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.FALL, scene, s1_bands=S1Bands.ALL, 
                                        s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)
    '''

    #scene = 86
    #s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.SUMMER, scene, s1_bands=S1Bands.ALL, 
    #                                    s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)

    scene = 40
    s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.SPRING, scene, s1_bands=S1Bands.ALL, 
                                        s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)
    '''

    scene = 120
    s1, s2, s2cloudy, _ = sen12mscr.get_triplets(Seasons.SUMMER, scene, s1_bands=S1Bands.ALL, 
                                        s2_bands=S2Bands.ALL, s2cloudy_bands=S2Bands.ALL)
    '''

    print(f"Scene: 106, S1: {s1.shape}, S2: {s2.shape}, cloudy S2: {s2cloudy.shape}")
    print("Time Taken {}s".format(time.time() - start))

    print(s1_list)

    print(len(s1_list), len(s2_list), len(s2_cloudy_list))

    print("\\")
    full_list = []
    for idx in range(len(s1_list)):

        s1 = s1_list[idx].split(sep="\\")[1:]


        file_path = s1[0] + '/' + s1[1] + '/' + s1[2]
        file_path  = file_path.replace('s1_', 'sensor_')
        file_path  = file_path.replace('_s1', '_sensor')

        print(file_path)

        full_list.append(file_path)
        
    print(full_list)

    print(len(full_list))
    ## full_list = full_list[:5]

    import pickle

    with open("full_list.txt", "wb") as fp:   #Pickling
        pickle.dump(full_list, fp)

#        pdb.set_trace()
#    s1_list = [x[68:] for x in s1_list]

#    print(s1_list)
    """
    full_list = []
    for idx in len(s1_list):
        full_list.append([])
    import pickle

    with open("s1_list.txt", "wb") as fp:   #Pickling
        pickle.dump(s1_list, fp)

    with open("s2_list.txt", "wb") as fp:   #Pickling
        pickle.dump(s2_list, fp)

    with open("s2_cloudy_list.txt", "wb") as fp:   #Pickling
        pickle.dump(s2_cloudy_list, fp)
    """
