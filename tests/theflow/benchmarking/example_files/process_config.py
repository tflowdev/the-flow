def process_config(theflow_config: dict, experiment_dict: dict) -> dict:
    """Modify a The Flow config.

    :param theflow_config: a The Flow config.
    :param experiment_dict: a benchmarking config experiment dictionary.

    returns: a modified The Flow config.
    """

    # Only keep input_features and output_features for the ames_housing dataset.
    if experiment_dict["dataset_name"] == "ames_housing":
        main_config_keys = list(theflow_config.keys())
        for key in main_config_keys:
            if key not in ["input_features", "output_features"]:
                del theflow_config[key]

    # Set the early_stop criteria to stop training after 7 epochs of no score improvement.
    theflow_config["trainer"] = {"early_stop": 7}

    # use sparse encoder for categorical features to mimic logreg
    theflow_config["combiner"] = {"type": "concat"}
    for i, feature in enumerate(theflow_config["input_features"]):
        if feature["type"] == "category":
            theflow_config["input_features"][i]["encoder"] = "sparse"
    for i, feature in enumerate(theflow_config["output_features"]):
        if feature["type"] == "category":
            theflow_config["output_features"][i]["encoder"] = "sparse"

    return theflow_config
