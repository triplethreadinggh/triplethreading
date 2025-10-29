from triplethreading._core import hello_from_bin
from triplethreading.model import LinearRegression, CauchyRegression
from triplethreading.logit import LogisticRegression

def hello() -> str:
    return hello_from_bin()

__all__ = ['LinearRegression', 'LogisticRegression', 'CauchyRegression']

