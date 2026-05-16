from enum import StrEnum


class ServiceSide(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'


class MatchType(StrEnum):
    SINGLES = "singles"
    DOUBLES = "doubles"