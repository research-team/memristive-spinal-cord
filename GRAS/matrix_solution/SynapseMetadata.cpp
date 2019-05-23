struct SynapseMetadata{
	// struct for human-readable initialization of connectomes
	int post_id;
	int synapse_delay;
	float synapse_weight;

	SynapseMetadata() = default;
	SynapseMetadata(int post_id, float synapse_delay, float synapse_weight){
		this->post_id = post_id;
		this->synapse_delay = static_cast<int>(synapse_delay * (1 / 0.25) + 0.5); // round
		this->synapse_weight = synapse_weight;
	}
};