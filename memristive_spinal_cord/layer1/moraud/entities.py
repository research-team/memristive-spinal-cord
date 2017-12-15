from enum import Enum


class Layer1Entities(Enum):
    FLEX_MOTOR = "Flexor Motoneurons"
    EXTENS_MOTOR = "Extensor Motoneurons"
    FLEX_INTER_1A = "Flexor 1A Interneurons"
    EXTENS_INTER_1A = "Extensor 1A Interneurons"
    FLEX_INTER_2 = "Flexor 2 Interneurons"
    EXTENS_INTER_2 = "Extensor 2 Interneurons"
    FLEX_AFFERENT_1A = "Flexor 1A fibers"
    EXTENS_AFFERENT_1A = "Extensor 1a fibers"
    FLEX_AFFERENT_2 = "Flexor 2 fibers"
    EXTENS_AFFERENT_2 = "Extensor 2 fibers"
