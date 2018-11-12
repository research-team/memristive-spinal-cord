from enum import Enum


class Params(Enum):
	MODEL = "model"
	EES_RATE = "ees_rate"
	INH_COEF = "inh_coef"
	SPEED = "speed"
	SIM_TIME = "sim_time"
	C_TIME = "c_time"
	RECORD_FROM = "record_from"
	MULTITEST = "multitest"


class Name(Enum):
	MP_E = "MP_E"

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
