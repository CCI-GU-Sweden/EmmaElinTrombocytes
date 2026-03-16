# EmmaElinTrombocytes
Detecting trombocytes with yolo and omero

to run the annotation:
>> python annotate_omero_datasets.py --token TOKEN_FROM_OMERO --config_name annotate_cells_and_granules

Be sure to indicate in the config.yaml, in the annotate part, the dataset to annotate and the different options! Better than arg parsing.
