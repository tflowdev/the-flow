# Copyright (c) 2022 Predibase, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pandas as pd

from theflow.datasets.loaders.dataset_loader import DatasetLoader


class AdultCensusIncomeLoader(DatasetLoader):
    def load_file_to_dataframe(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith(".test"):
            # The test file contains the line "|1x3 Cross validator" before the CSV content.
            return pd.read_csv(file_path, skiprows=1)
        return super().load_file_to_dataframe(file_path)

    def transform_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        processed_df = super().transform_dataframe(dataframe)
        processed_df["income"] = processed_df["income"].str.rstrip(".")
        processed_df["income"] = processed_df["income"].str.strip()
        return processed_df
