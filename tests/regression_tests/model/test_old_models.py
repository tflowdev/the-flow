import os
import zipfile

import pandas as pd
import pytest
import wget

from theflow.api import The FlowModel
from theflow.data.dataset_synthesizer import build_synthetic_dataset_df
from theflow.globals import MODEL_FILE_NAME

NUM_EXAMPLES = 25


def test_model_loaded_from_old_config_prediction_works(tmpdir):
    # Titanic model based on 0.5.3.
    old_model_url = "https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/theflow_unit_tests/old_model.zip"
    old_model_filename = wget.download(old_model_url, tmpdir)
    with zipfile.ZipFile(old_model_filename, "r") as zip_ref:
        zip_ref.extractall(tmpdir)
    example_data = {
        "PassengerId": 892,
        "Pclass": 3,
        "Name": "Kelly, Mr. James",
        "Sex": "male",
        "Age": 34.5,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "330911",
        "Fare": 7.8292,
        "Cabin": None,
        "Embarked": "Q",
    }
    test_set = pd.DataFrame(example_data, index=[0])

    theflow_model = The FlowModel.load(os.path.join(tmpdir, "old_model/model"))
    predictions, _ = theflow_model.predict(dataset=test_set)

    assert predictions.to_dict()["Survived_predictions"] == {0: False}


@pytest.mark.parametrize(
    "model_url",
    [
        "https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/theflow_unit_tests/titanic_v07.zip",
        "https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/theflow_unit_tests/twitter_bots_v05_1.zip",
        "https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/theflow_unit_tests/respiratory_v05.zip",
        # TODO(Arnav): Re-enable once https://github.com/theflow-ai/theflow/issues/3150 is resolved since the GBM
        # model uses the PassthroughDecoder for the category output feature.
        # "https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/theflow_unit_tests/gbm_adult_census_income_v061.zip",  # noqa: E501
    ],
    ids=["titanic", "twitter_bots", "respiratory"],  # , "gbm_adult_census_income"],
)
def test_predict_deprecated_model(model_url, tmpdir):
    model_dir = os.path.join(tmpdir, MODEL_FILE_NAME)
    os.makedirs(model_dir)

    archive_path = wget.download(model_url, tmpdir)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    theflow_model = The FlowModel.load(model_dir)
    df = build_synthetic_dataset_df(NUM_EXAMPLES, theflow_model.config)

    pred_df, _ = theflow_model.predict(df)
    assert len(pred_df) == 25
