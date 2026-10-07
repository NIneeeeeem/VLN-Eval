"""InstanceImageNav metric names reuse the Habitat navigation formulas."""
from nav_eval.evaluation.metrics import (
    ObjectNavNE,
    ObjectNavOracleSuccess,
    ObjectNavSoftSPL,
    ObjectNavSPL,
    ObjectNavSuccess,
)


class InstanceImageNavSuccess(ObjectNavSuccess):
    name = "instance_imagenav.success"


class InstanceImageNavSPL(ObjectNavSPL):
    name = "instance_imagenav.spl"


class InstanceImageNavSoftSPL(ObjectNavSoftSPL):
    name = "instance_imagenav.soft_spl"


class InstanceImageNavNE(ObjectNavNE):
    name = "instance_imagenav.ne_m"


class InstanceImageNavOracleSuccess(ObjectNavOracleSuccess):
    name = "instance_imagenav.oracle_success"
