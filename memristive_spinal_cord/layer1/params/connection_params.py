from memristive_spinal_cord.layer1.params.entity_params import EntityParams


class ConnectionParams(EntityParams):
    def __init__(self, pre, post, syn_spec, conn_spec):
        self.pre = pre
        self.post = post
        self.syn_spec = syn_spec
        self.conn_spec = conn_spec

    def to_nest_params(self):
        return dict(
            pre=self.pre,
            post=self.post,
            syn_spec=self.syn_spec,
            conn_spec=self.conn_spec,
        )
