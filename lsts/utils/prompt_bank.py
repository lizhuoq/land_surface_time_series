import numpy as np


def get_ismn_prompt(variable: str, id: int, postfix: bool = True, metadata_path: str = "/data/home/scv7343/run/ismn_metadata.csv") -> str:
    import pandas as pd
    metadata = pd.read_csv(metadata_path, header=[0, 1])
    metadata.set_index(("variable", "key"), inplace=True)
    attribute = metadata.loc[id]
    assert variable == attribute[("variable", "val")]
    latitude = attribute[("latitude", "val")]
    longitude = attribute[("longitude", "val")]
    lc_2000 = attribute[("lc_2000", "val")]
    lc_2005 = attribute[("lc_2005", "val")]
    lc_2010 = attribute[("lc_2010", "val")]
    lc = pd.Series([lc_2000, lc_2005, lc_2010]).mode()
    depth_from = attribute[("variable", "depth_from")]
    depth_to = attribute[("variable", "depth_to")]
    clay_fraction = attribute[("clay_fraction", "val")]
    climate_KG = attribute[("climate_KG", "val")]
    elevation = attribute[("elevation", "val")]
    organic_carbon = attribute[("organic_carbon", "val")]
    sand_fraction = attribute[("sand_fraction", "val")]
    saturation = attribute[("saturation", "val")]
    silt_fraction = attribute[("silt_fraction", "val")]
    if variable == "soil_moisture":
        prefix = "Soil moisture is crucial for plant growth as it affects the plants' ability to absorb water and nutrients, thereby influencing their growth and development. The soil moisture undergoes temporal variations due to factors such as seasons, precipitation, and evaporation, which have significant impacts on soil moisture cycling, agricultural production, and ecosystem functionality."
    elif variable == "soil_temperature":
        prefix = "Soil temperature has significant impacts on plant growth and microbial activity, influencing chemical reaction rates, nutrient transformation, and water movement within the soil. It varies with seasonal changes and the day-night cycle, exhibiting distinct periodicity, while being influenced by factors such as climatic conditions, topographical features, and soil types."
    elif variable == "snow_depth":
        prefix = "The snow depth refers to the thickness of snow covering the ground, which has significant impacts on climate, hydrology, and ecosystems. The temporal variation of snow depth reflects the seasonal climate changes and long-term climate trends in a region."
    elif variable == "snow_water_equivalent":
        prefix = "The snow water equivalent refers to the amount of water contained in a certain volume of snow, and it plays an important role in water resources management and hydrological processes. The temporal variations of snow water equivalent reflect the changes in water content in the snowpack over time, which are of significant importance for predicting floods, water resources management, agricultural irrigation, and so forth."
    elif variable == "soil_suction":
        prefix = "Soil suction is the attraction force of soil particles to water molecules, which affects the movement and distribution of water in soil. The temporal variation of soil suction reflects the change in soil moisture status over time, which is of significant importance for understanding soil moisture movement and soil moisture management."
    elif variable == "surface_temperature":
        prefix = "The surface temperature reflects the distribution of heat on the Earth's surface, which has significant impacts on climate, ecosystems, and human activities. Its temporal variations reflect the influences of factors such as climate change, urbanization, and land use changes on surface temperature, making it one of the key indicators for studying the dynamic changes in the Earth system."
    elif variable == "precipitation":
        prefix = "Precipitation is the process in which moisture in the atmosphere falls to the ground in the form of rain, snow, hail, and other forms, significantly influencing surface water cycles and ecosystems. The temporal variation of precipitation time series describes the changing patterns of precipitation over time, including trends in parameters such as frequency, intensity, and duration of precipitation events."
    elif variable == "air_temperature":
        prefix = "The air temperature has a wide-ranging impact on both biological and natural systems, influencing phenomena such as plant growth, animal behavior, and atmospheric circulation. Changes in air temperature time series reflect the seasonal variations, long-term trends, as well as the frequency and intensity of extreme weather events."
    postfix = [f"The variable of this time series is {" ".join(variable.split("_"))}.", 
               f"The longitude of the observation site is {longitude} decimal degrees, the latitude is {latitude} decimal degrees, with an elevation of {elevation} meters.", 
               "The depth below the surface is represented as positive values, while above the surface it is represented as negative values.", 
               f"The observation depth ranges from {depth_from} to {depth_to} meters."]
    if clay_fraction != np.nan:
        postfix.append(f"The clay fraction is {clay_fraction}% by weight.")
    if organic_carbon != np.nan:
        postfix.append(f"The organic carbon is {organic_carbon}% by weight.")
    if sand_fraction != np.nan:
        postfix.append(f"The sand fraction is {sand_fraction}% by weight.")
    if saturation != np.nan:
        postfix.append(f"The saturation is {saturation}% by volume.")
    if silt_fraction != np.nan:
        postfix.append(f"The silt fraction is {silt_fraction}% by weight.")
    climate_KG_long_name = get_climate_long_name(climate_KG)
    lc_name = get_lc_name(lc)
    postfix.append(f"The land cover type is {lc_name}, and the climate type is {climate_KG_long_name}.")
    if postfix:
        return prefix + "".join(postfix)
    return prefix


def get_lc_name(id: int) -> str:
    lc_map = {
        10: "Cropland, rainfed", 
        11: "Cropland, rainfed / Herbaceous cover", 
        12: "Cropland, rainfed / Tree or shrub cover",
        20: "Cropland, irrigated or post-flooding",
        30: "Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous",
        40: "Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)",
        50: "Tree cover, broadleaved, evergreen, Closed to open (>15%)",
        60: "Tree cover, broadleaved, deciduous, Closed to open (>15%)",
        61: "Tree cover, broadleaved, deciduous, Closed (>40%)",
        62: "Tree cover, broadleaved, deciduous, Open (15-40%)",
        70: "Tree cover, needleleaved, evergreen, closed to open (>15%)",
        71: "Tree cover, needleleaved, evergreen, closed (>40%)",
        72: "Tree cover, needleleaved, evergreen, open (15-40%)",
        80: "Tree cover, needleleaved, deciduous, closed to open (>15%)",
        81: "Tree cover, needleleaved, deciduous, closed (>40%)",
        82: "Tree cover, needleleaved, deciduous, open (15-40%)",
        90: "Tree cover, mixed leaf type (broadleaved and needleleaved)",
        100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
        110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
        120: "Shrubland",
        121: "Shrubland / Evergreen Shrubland",
        122: "Shrubland / Deciduous Shrubland",
        130: "Grassland",
        140: "Lichens and mosses",
        150: "Sparse vegetation (tree, shrub, herbaceous cover) (<15%)",
        152: "Sparse vegetation (tree, shrub, herbaceous cover) (<15%) / Sparse shrub (<15%)",
        153: "Sparse vegetation (tree, shrub, herbaceous cover) (<15%) / Sparse herbaceous cover (<15%)",
        160: "Tree cover, flooded, fresh or brakish water",
        170: "Tree cover, flooded, saline water",
        180: "Shrub or herbaceous cover, flooded, fresh/saline/brakish water",
        190: "Urban areas",
        200: "Bare areas",
        201: "Consolidated bare areas",
        202: "Unconsolidated bare areas",
        210: "Water",
        220: "Permanent snow and ice",
    }
    return lc_map[id]


def get_climate_long_name(short_name: str) -> str:
    climate_map = {
        "Af": "Tropical Rainforest", 
        "Am": "Tropical Monsoon", 
        "As": "Tropical Savanna Dry", 
        "Aw": "Tropical Savanna Wet", 
        "BWk": "Arid Desert Cold", 
        "BWh": "Arid Desert Hot", 
        "BWn": "Arid Desert With Frequent Fog", 
        "BSk": "Arid Steppe Cold", 
        "BSh": "Arid Steppe Hot", 
        "BSn": "Arid Steppe With Frequent Fog", 
        "Csa": "Temperate Dry Hot Summer", 
        "Csb": "Temperate Dry Warm Summer", 
        "Csc": "Temperate Dry Cold Summer", 
        "Cwa": "Temperate Dry Winter, Hot Summer", 
        "Cwb": "Temperate Dry Winter, Warm Summer", 
        "Cwc": "Temperate Dry Winter, Cold Summer", 
        "Cfa": "Temperate Without Dry Season, Hot Summer", 
        "Cfb": "Temperate Without Dry Season, Warm Summer", 
        "Cfc": "Temperate Without Dry Season, Cold Summer", 
        "Dsa": "Cold Dry Summer, Hot Summer",
        "Dsb": "Cold Dry Summer, Warm Summer", 
        "Dsc": "Cold Dry Summer, Cold Summer", 
        "Dsd": "Cold Dry Summer, Very Cold Winter", 
        "Dwa": "Cold Dry Winter, Hot Summer", 
        "Dwb": "Cold Dry Winter, Warm Summer", 
        "Dwc": "Cold Dry Winter, Cold Summer", 
        "Dwd": "Cold Dry Winter, Very Cold Winter", 
        "Dfa": "Cold Dry Without Dry Season, Hot Summer", 
        "Dfb": "Cold Dry Without Dry Season, Warm Summer", 
        "Dfc": "Cold Dry Without Dry Season, Cold Summer", 
        "Dfd": "Cold Dry Without Dry Season, Very Cold Winter", 
        "ET": "Polar Tundra", 
        "EF": "Polar Eternal Winter", 
        "W": "Water"
    }
    return climate_map[short_name]
