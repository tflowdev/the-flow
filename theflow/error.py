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

from theflow.api_annotations import PublicAPI


@PublicAPI
class The FlowError(Exception):
    """Base class for all custom exceptions raised by the The Flow framework."""

    def __reduce__(self):
        """Docs: https://docs.python.org/3/library/pickle.html#object.__reduce__."""
        raise NotImplementedError(
            "Implement __reduce__ for all subclasses of The FlowError as it's necessary for "
            "serialization by Ray. See https://github.com/theflow-ai/theflow/pull/2695."
        )


@PublicAPI
class InputDataError(The FlowError, ValueError):
    """Exception raised for errors in the input data.

    Appropriate for data which is not convertible to the input feature type, columns with all missing values,
    categorical columns with only one category, etc...

    Attributes:
        column - The name of the input column which caused the error
        feature_type - The The Flow feature type which caused the error (number, binary, category...).
        message - An error message describing the situation.
    """

    def __init__(self, column_name: str, feature_type: str, message: str):
        self.column_name = column_name
        self.feature_type = feature_type
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f'Column "{self.column_name}" as {self.feature_type} feature: {self.message}'

    def __reduce__(self):
        return type(self), (self.column_name, self.feature_type, self.message)


@PublicAPI
class ConfigValidationError(The FlowError, ValueError):
    """Exception raised for errors in the The Flow configuration.

    Appropriate for bad configuration values, missing required configuration values, etc...

    Attributes:
        message - An error message describing the situation.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __reduce__(self):
        return type(self), (self.message,)
