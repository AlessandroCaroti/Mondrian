from enum import Enum

class Type(Enum):
    # type of data: Explicit Identifiers, Sensitive data and Quasi-Identifiers: can be NUMBER, CATEGORICAL or DATE
    EI = "EI"
    SD = "SD"
    NUMERICAL = "NUMERICAL"
    DATE = "DATE"
    CATEGORICAL = "CATEGORICAL"