NEURON	{
  POINT_PROCESS IaGenerator
    POINTER fhill, fhill2
  RANGE y
  RANGE interval, number, start
  RANGE noise, freq, mean, vel, invl
}

PARAMETER {
	number	= 10 <0,1e9>	: number of spikes
	start		= 50 (ms)	: start of first spike
	noise		= 0 <0,1>	: amount of randomeaness (0.0 - 1.0)
	freq
  mean = 1
	interval	= 10 (ms) <1e-9,1e9>: time between spikes (msec)
}

ASSIGNED {
	y
	event (ms)
	on
	end (ms)
  t0
  v0
  fhill
  vel
  fhill0
  fhill2
  len2
}

PROCEDURE seed(x) {
	set_seed(x)
}

INITIAL {
	on = 0
	y = 0
  t0 = start
  v0 = 50
  freq = 80
  fhill0 = 0
	interval = 1000/freq
	if (noise < 0) {
		noise = 0
	}
	if (noise > 1) {
		noise = 1
	}
	if (start >= 0 && number > 0) {
		: randomize the first spike so on average it occurs at start+interval
		event = start + invl(t)
		net_send(event, 3)
	}
}

PROCEDURE init_sequence(t(ms)) {
	if (number > 0) {
		on = 1
		event = t
		end = t + 1e-6 + invl(t)*(number-1)
	}
}

FUNCTION invl(t (ms)) (ms) {
  len2 = 1.5*fhill2*(t-t0)
  if (fhill > fhill0){
    vel = v0 + len2 + 0.15*fhill*(t-t0)*(t-t0)
    if (vel < 0){vel = 1}
  }else{
    vel = v0 - len2 - 0.4*fhill*(t-t0)*(t-t0)
    if (vel < 0){vel = 1}
  }
  v0 = vel
  fhill0 = fhill
  mean = 1000/vel
  printf("time: %g, mean: %g \n", t, mean)

  t0 = t
	if (noise == 0) {
		invl = mean
	}else{
		invl = (1. - noise)*mean + noise*mean*exprand(1)
	}
}

PROCEDURE event_time(t (ms)) {
	if (number > 0) {
		event = event + invl(t)
	}
	if (event > end) {
		on = 0
	}
}

NET_RECEIVE (w) {
	if (flag == 0) { : external event
		if (w > 0 && on == 0) { : turn on spike sequence
			init_sequence(t)
			net_send(0, 1)
		}else if (w < 0 && on == 1) { : turn off spiking
			on = 0
		}
	}
	if (flag == 3) { : from INITIAL
		if (on == 0) {
			init_sequence(t)
			net_send(0, 1)
		}
	}
	if (flag == 1 && on == 1) {
		y = 2
		net_event(t)
		event_time(t)
		if (on == 1) {
			net_send(event - t, 1)
		}
		net_send(.1, 2)
	}
	if (flag == 2) {
		y = 0
	}
}
