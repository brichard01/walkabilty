# ---- super-categories for your label set ----
SUPER_CATS = {
    # natural/green
    "greenery": {
        "Vegetation", "Terrain"  # natural ground; keep simple
    },

    # pedestrian infrastructure
    "ped_infra": {
        "Sidewalk", "Crosswalk - Plain", "Lane Marking - Crosswalk",
        "Curb", "Curb Cut", "Pedestrian Area", "Bike Lane"
    },

    # people, riders, and bicycles
    "people_bike": {
        "Person", "Bicyclist", "Motorcyclist", "Other Rider", "Bicycle",
        # rare but living beings – put here to avoid creating extra buckets
        "Bird", "Ground Animal"
    },

    # motor traffic & vehicles (incl. rail vehicles)
    "motor_traffic": {
        "Car", "Bus", "Truck", "Motorcycle", "Caravan", "Trailer",
        "Other Vehicle", "On Rails", "Wheeled Slow"
    },

    # roadway / drivable or transport surfaces/marks
    "roadway": {
        "Road", "Service Lane", "Lane Marking - General", "Parking",
        "Rail Track", "Pothole"
    },

    # buildings & major structures
    "buildings": {
        "Building", "Wall", "Fence", "Guard Rail", "Barrier",
        "Bridge", "Tunnel"
    },

    # street furniture & signage/utilities
    "furniture_signage": {
        "Street Light", "Pole", "Utility Pole", "Traffic Light",
        "Traffic Sign (Back)", "Traffic Sign (Front)", "Traffic Sign Frame",
        "Banner", "Bench", "Bike Rack", "Billboard",
        "Catch Basin", "CCTV Camera", "Fire Hydrant",
        "Junction Box", "Mailbox", "Manhole", "Phone Booth", "Trash Can"
    },

    # sky & water
    "sky_water": {
        "Sky", "Water"
    },
}

# ---- classes to ignore (do not contribute to features)
# original: Mountain(25), Sand(26), Snow(28), Boat(53)
# + we also ignore "Car Mount"(63) and "Ego Vehicle"(64) because they are artifacts of the recording rig.
IGNORE_IDS = {25, 26, 28, 53, 63, 64}
