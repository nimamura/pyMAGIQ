# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "scipy", "pandas", "tensorflow", "keras")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError(
        "Missing required dependencies {0}".format(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies

from . import utils
from . import vis
