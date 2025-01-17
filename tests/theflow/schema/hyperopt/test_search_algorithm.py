import pytest

from theflow.schema.hyperopt.search_algorithm import BaseSearchAlgorithmConfig
from theflow.schema.hyperopt.utils import register_search_algorithm_config, search_algorithm_config_registry
from theflow.schema.utils import theflow_dataclass, ProtectedString


@pytest.fixture(
    params=[  # Tuples of SA name, dependency list, whether it should raise an exception
        ("no_deps", None, False),
        ("installed", [("theflow", "theflow")], False),
        ("multiple_installed", [("theflow", "theflow"), ("marshmallow", "marshmallow")], False),
        ("not_installed", [("fake_dependency", "fake_dependency")], True),
        ("mixed_installed", [("fake_dependency", "fake_dependency"), ("theflow", "theflow")], True),
    ]
)
def dependency_check_config(request):
    key, deps, raises_exception = request.param

    @register_search_algorithm_config(key, dependencies=deps)
    @theflow_dataclass
    class DependencyCheckConfig(BaseSearchAlgorithmConfig):
        type: str = ProtectedString(key)

    yield DependencyCheckConfig(), raises_exception
    del search_algorithm_config_registry[key]


def test_dependency_check(dependency_check_config):
    """Test that the hyperopt search alg dependency check properly identifies missing dependencies.

    Most search algorithms supported by Ray Tune have additional dependencies that may not be installed. The schema
    records these dependencies and can be used to verify they are installed at run time.
    """
    config, raises_exception = dependency_check_config
    if raises_exception:
        with pytest.raises(ImportError):
            config.dependencies_installed()
    else:
        assert config.dependencies_installed()
