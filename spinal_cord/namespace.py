from enum import Enum


class Afferent(Enum):
    IA = 'Ia'
    II = 'II'
    SENSORY = 'sensory'

class Muscle(Enum):
    FLEX = 'TA'
    EXTENS = 'GM'


class Speed(Enum):
    FIFTEEN = 15
    DEFAULT = ''


class Interval(Enum):
    DEFAULT = 20


class Group(Enum):
    IA = 'Ia'
    II = 'II'
    MOTO = 'Mn'
