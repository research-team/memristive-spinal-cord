import nest
from spinal_cord.afferents.afferent_fiber import AfferentFiber
from spinal_cord.motogroups.motogroup import Motogroup
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.ees.ees import EES


class Level1:

    def __init__(self):

        flex_motogroup = Motogroup()
        extens_motogroup = Motogroup()

        afferent_fiber_ia_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.IA)
        afferent_fiber_ia_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
        afferent_fiber_ii_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.II)
        afferent_fiber_ii_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.II)

        ees = EES()

        ees.connect(
            500,
            afferent_fiber_ia_extens,
            afferent_fiber_ia_flex,
            afferent_fiber_ii_extens,
            afferent_fiber_ii_flex
        )
