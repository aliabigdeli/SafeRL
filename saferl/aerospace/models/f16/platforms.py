import abc

import numpy as np
import math
from scipy.spatial.transform import Rotation

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseODESolverDynamics, BaseDynamics

import time
from scipy.integrate import RK45
from .aerobench.highlevel.controlled_f16 import controlled_f16
from .aerobench.util import get_state_names, Euler, StateIndex, print_state, Freezable

from .aerobench.lowlevel.low_level_controller import LowLevelController
from .aerobench.example.wingman.wingman_autopilot import WingmanAutopilot
from numpy import deg2rad

class BaseDubinsPlatform(BasePlatform):

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'heading': self.heading,
            'v': self.v,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def v(self):
        return self.state.v

    @property
    def yaw(self):
        return self.state.yaw

    @property
    def pitch(self):
        return self.state.pitch

    @property
    def roll(self):
        return self.state.roll

    @property
    def heading(self):
        return self.state.heading

    @property
    def gamma(self):
        return self.state.gamma


class BaseDubinsState(BasePlatformStateVectorized):

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError

    @property
    def velocity(self):
        velocity = np.array([
            self.v * math.cos(self.heading) * math.cos(self.gamma),
            self.v * math.sin(self.heading) * math.cos(self.gamma),
            -1 * self.v * math.sin(self.gamma),
        ], dtype=np.float64)
        return velocity

    @property
    def yaw(self):
        return self.heading

    @property
    def pitch(self):
        return self.gamma

    @property
    @abc.abstractmethod
    def roll(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heading(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gamma(self):
        raise NotImplementedError


class F16SimState(Freezable):
    '''object containing simulation state

    With this interface you can run partial simulations, rather than having to simulate for the entire time bound

    if you just want a single run with a fixed time, it may be easier to use the run_f16_sim function
    '''

    def __init__(self, initial_state, ap, step=1/30, extended_states=False,
                integrator_str='rk45', v2_integrators=False, print_errors=True, keep_intermediate_states=True,
                 custom_stop_func=None):

        self.model_str = model_str = ap.llc.model_str
        self.v2_integrators = v2_integrators
        initial_state = np.array(initial_state, dtype=float)

        self.keep_intermediate_states = keep_intermediate_states
        self.custom_stop_func = custom_stop_func

        self.step = step
        self.ap = ap
        self.print_errors = print_errors

        llc = ap.llc

        num_vars = len(get_state_names()) + llc.get_num_integrators()

        if initial_state.size < num_vars:
            # append integral error states to state vector
            x0 = np.zeros(num_vars)
            x0[:initial_state.shape[0]] = initial_state
        else:
            x0 = initial_state

        assert x0.size % num_vars == 0, f"expected initial state ({x0.size} vars) to be multiple of {num_vars} vars"
        self.x0 = x0
        self.ap = ap

        self.times = None
        self.states = None
        self.modes = None

        self.extended_states = extended_states

        if self.extended_states:
            self.xd_list = None
            self.u_list = None
            self.Nz_list = None
            self.ps_list = None
            self.Ny_r_list = None

        self.cur_sim_time = 0
        self.total_sim_time = 0

        self.der_func = make_der_func(ap, model_str, v2_integrators)

        if integrator_str == 'rk45':
            integrator_class = RK45
            self.integrator_kwargs = {}
        else:
            assert integrator_str == 'euler'
            integrator_class = Euler
            self.integrator_kwargs = {'step': step}

        self.integrator_class = integrator_class
        self.integrator = None

        self.freeze_attrs()

    def init_simulation(self):
        'initial simulation (upon first call to simulate_to)'

        assert self.integrator is None

        self.times = [0]
        self.states = [self.x0]

        # mode can change at time 0
        self.ap.advance_discrete_mode(self.times[-1], self.states[-1])

        self.modes = [self.ap.mode]

        if self.extended_states:
            xd, u, Nz, ps, Ny_r = get_extended_states(self.ap, self.times[-1], self.states[-1],
                                                      self.model_str, self.v2_integrators)

            self.xd_list = [xd]
            self.u_list = [u]
            self.Nz_list = [Nz]
            self.ps_list = [ps]
            self.Ny_r_list = [Ny_r]
        
        # note: fixed_step argument is unused by rk45, used with euler
        self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                **self.integrator_kwargs)

    def simulate_to(self, tmax, tol=1e-7, update_mode_at_start=False):
        '''simulate up to the passed in time

        this adds states to self.times, self.states, self.modes, and the other extended state lists if applicable 
        '''

        # underflow errors were occuring if I don't do this
        oldsettings = np.geterr()
        np.seterr(all='raise', under='ignore')

        start = time.perf_counter()

        ap = self.ap
        step = self.step

        if self.integrator is None:
            self.init_simulation()
        elif update_mode_at_start:
            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes[-1] = ap.mode # overwrite last mode

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf, \
                                                        **self.integrator_kwargs)

        assert tmax >= self.cur_sim_time
        self.cur_sim_time = tmax

        assert self.integrator.status == 'running', \
            f"integrator status was {self.integrator.status} in call to simulate_to()"

        assert len(self.modes) == len(self.times), f"modes len was {len(self.modes)}, times len was {len(self.times)}"
        assert len(self.states) == len(self.times)

        while True:
            if not self.keep_intermediate_states and len(self.times) > 1:
                # drop all except last state
                self.times = [self.times[-1]]
                self.states = [self.states[-1]]
                self.modes = [self.modes[-1]]

                if self.extended_states:
                    self.xd_list = [self.xd_list[-1]]
                    self.u_list = [self.u_list[-1]]
                    self.Nz_list = [self.Nz_list[-1]]
                    self.ps_list = [self.ps_list[-1]]
                    self.Ny_r_list = [self.Ny_r_list[-1]]

            next_step_time = self.times[-1] + step

            if abs(self.times[-1] - tmax) > tol and next_step_time > tmax:
                # use a small last step
                next_step_time = tmax

            if next_step_time >= tmax + tol:
                # don't do any more steps
                break

            # goal for rest of the loop: do one more step

            while next_step_time >= self.integrator.t + tol:
                # keep advancing integrator until it goes past the next step time
                assert self.integrator.status == 'running'
                
                self.integrator.step()

                if self.integrator.status != 'running':
                    break

            if self.integrator.status != 'running':
                break

            # get the state at next_step_time
            self.times.append(next_step_time)

            try:
                self.states.append(self.integrator.x)
            except:
                dense_output = self.integrator.dense_output()
                self.states.append(dense_output(next_step_time))
            # if abs(self.integrator.t - next_step_time) < tol:
            #     self.states.append(self.integrator.x)
            # else:
            #     dense_output = self.integrator.dense_output()
            #     self.states.append(dense_output(next_step_time))

            # re-run dynamics function at current state to get non-state variables
            if self.extended_states:
                xd, u, Nz, ps, Ny_r = get_extended_states(ap, self.times[-1], self.states[-1],
                                                          self.model_str, self.v2_integrators)

                self.xd_list.append(xd)
                self.u_list.append(u)

                self.Nz_list.append(Nz)
                self.ps_list.append(ps)
                self.Ny_r_list.append(Ny_r)

            mode_changed = ap.advance_discrete_mode(self.times[-1], self.states[-1])
            self.modes.append(ap.mode)

            stop_func = self.custom_stop_func if self.custom_stop_func is not None else ap.is_finished

            if stop_func(self.times[-1], self.states[-1]):
                # this both causes the outer loop to exit and sets res['status'] appropriately
                self.integrator.status = 'autopilot finished'
                break

            if mode_changed:
                # re-initialize the integration class on discrete mode switches
                self.integrator = self.integrator_class(self.der_func, self.times[-1], self.states[-1], np.inf,
                                                        **self.integrator_kwargs)

        if self.integrator.status == 'failed' and self.print_errors:
            print(f'Warning: integrator status was "{self.integrator.status}"')
        elif self.integrator.status != 'autopilot finished':
            assert abs(self.times[-1] - tmax) < tol, f"tmax was {tmax}, self.times[-1] was {self.times[-1]}"

        self.total_sim_time += time.perf_counter() - start
        np.seterr(**oldsettings)


class SimModelError(RuntimeError):
    'simulation state went outside of what the model is capable of simulating'

def make_der_func(ap, model_str, v2_integrators):
    'make the combined derivative function for integration'

    def der_func(t, full_state):
        'derivative function, generalized for multiple aircraft'

        u_refs = ap.get_checked_u_ref(t, full_state)

        num_aircraft = u_refs.size // 4
        num_vars = len(get_state_names()) + ap.llc.get_num_integrators()
        assert full_state.size // num_vars == num_aircraft

        xds = []

        for i in range(num_aircraft):
            state = full_state[num_vars*i:num_vars*(i+1)]

            #print(f".called der_func(aircraft={i}, t={t}, state={full_state}")

            alpha = state[StateIndex.ALPHA]
            if not -2 < alpha < 2:
                raise SimModelError(f"alpha ({alpha}) out of bounds")

            vel = state[StateIndex.VEL]
            # even going lower than 300 is probably not a good idea
            if not 200 <= vel <= 3000:
                raise SimModelError(f"velocity ({vel}) out of bounds")

            alt = state[StateIndex.ALT]
            if not -10000 < alt < 100000:
                raise SimModelError(f"altitude ({alt}) out of bounds")

            u_ref = u_refs[4*i:4*(i+1)]

            xd = controlled_f16(t, state, u_ref, ap.llc, model_str, v2_integrators)[0]
            xds.append(xd)

        rv = np.hstack(xds)

        return rv

    return der_func

def get_extended_states(ap, t, full_state, model_str, v2_integrators):
    '''get xd, u, Nz, ps, Ny_r at the current time / state

    returns tuples if more than one aircraft
    '''

    llc = ap.llc
    num_vars = len(get_state_names()) + llc.get_num_integrators()
    num_aircraft = full_state.size // num_vars

    xd_tup = []
    u_tup = []
    Nz_tup = []
    ps_tup = []
    Ny_r_tup = []

    u_refs = ap.get_checked_u_ref(t, full_state)

    for i in range(num_aircraft):
        state = full_state[num_vars*i:num_vars*(i+1)]
        u_ref = u_refs[4*i:4*(i+1)]

        xd, u, Nz, ps, Ny_r = controlled_f16(t, state, u_ref, llc, model_str, v2_integrators)

        xd_tup.append(xd)
        u_tup.append(u)
        Nz_tup.append(Nz)
        ps_tup.append(ps)
        Ny_r_tup.append(Ny_r)

    if num_aircraft == 1:
        rv_xd = xd_tup[0]
        rv_u = u_tup[0]
        rv_Nz = Nz_tup[0]
        rv_ps = ps_tup[0]
        rv_Ny_r = Ny_r_tup[0]
    else:
        rv_xd = tuple(xd_tup)
        rv_u = tuple(u_tup)
        rv_Nz = tuple(Nz_tup)
        rv_ps = tuple(ps_tup)
        rv_Ny_r = tuple(Ny_r_tup)

    return rv_xd, rv_u, rv_Nz, rv_ps, rv_Ny_r

class F162dPlatform(BaseDubinsPlatform):

    def __init__(self, name, controller=None, rta=None, v_min=10, v_max=100, integration_method='Euler'):

        dynamics = F162dDynamics(name, v_min=v_min, v_max=v_max, integration_method=integration_method)
        actuator_set = F162dActuatorSet()

        state = F162dState()

        super().__init__(name, dynamics, actuator_set, state, controller, rta=rta)


class F162dState(BaseDubinsState):

    def build_vector(self, x=0, y=0, heading=0, v=50, **kwargs):

        return np.array([x, y, heading, v], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @x.setter
    def x(self, value):
        self._vector[0] = value

    @property
    def y(self):
        return self._vector[1]

    @y.setter
    def y(self, value):
        self._vector[1] = value

    @property
    def z(self):
        return 0

    @property
    def heading(self):
        return self._vector[2]

    @heading.setter
    def heading(self, value):
        self._vector[2] = value

    @property
    def v(self):
        return self._vector[3]

    @v.setter
    def v(self, value):
        self._vector[3] = value

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:2] = self._vector[0:2]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler('z', self.yaw)

    @property
    def gamma(self):
        return 0

    @property
    def roll(self):
        return 0


class F162dActuatorSet(BaseActuatorSet):

    def __init__(self):

        actuators = [
            ContinuousActuator(
                'rudder',
                [np.deg2rad(-6), np.deg2rad(6)],
                0
            ),
            ContinuousActuator(
                'throttle',
                [-10, 10],
                0
            )
        ]

        super().__init__(actuators)


def get_wingman_autopilot_init():
    ### Initial Conditions ###
    power = 4 # engine power level (0-10)

    # Default alpha & beta
    alpha = 0 #deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 3600 #3800        # altitude (ft)
    vt = 550          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = -np.pi/2 #math.pi/8   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    return init

class F162dDynamics(BaseDynamics):

    def __init__(self, name, v_min=10, v_max=100, *args, **kwargs):
        self.v_min = v_min
        self.v_max = v_max
        self.name = name


        init = get_wingman_autopilot_init() # it doesn't matter, because we reinitialize the 'F16SimState' it in the step function
        self.ap = WingmanAutopilot(target_heading=-np.pi/2, target_vel=550, target_alt=3600, stdout=True)
        

        self.inner_step = 1
        extended_states = False # used for plotting in aerobench
        self.fss = F16SimState(init, self.ap, self.inner_step, extended_states,
                integrator_str='rk45', v2_integrators=False, print_errors=True, custom_stop_func=None)

        super().__init__()

    def step(self, step_size, state, control):

        if self.fss.cur_sim_time == 0:
            # we initialize the 'F16SimState' here because in SafeRL platfrom, yaml config file only initializes the F162dPlatform.state ('F162dState')
            self.fss.x0[StateIndex.POSE] = state.x
            self.fss.x0[StateIndex.POSN] = state.y
            self.fss.x0[StateIndex.PSI] = np.pi/2 - state.heading
            self.fss.x0[StateIndex.VT] = state.v
            # ToDo: check if need to assign the target heading and velocity
            self.ap.targets[0] = self.fss.x0[StateIndex.PSI]
            self.ap.targets[1] = self.fss.x0[StateIndex.VT]
        
        
        self.ap.targets[0] -= control[0] # modify target heading by rudder control input
        # modify target velocity by throttle control input
        k_v = 1.0 # need to be tuned
        if self.ap.targets[1] + control[1]*k_v < self.v_min:
            self.ap.targets[1] = self.v_min
        elif self.ap.targets[1] + control[1]*k_v > self.v_max:
            self.ap.targets[1] = self.v_max
        else:
            self.ap.targets[1] += control[1]*k_v

        t_to = self.fss.cur_sim_time + step_size
        self.fss.simulate_to(t_to)
        last_state = self.fss.states[-1]
        rv_state = F162dState(x=last_state[StateIndex.POSE], y=last_state[StateIndex.POSN], heading=last_state[StateIndex.PSI], v=last_state[StateIndex.VT])
        return rv_state


"""
3D Dubins Implementation
"""


class Dubins3dPlatform(BaseDubinsPlatform):

    def __init__(self, name, controller=None, v_min=10, v_max=100, integration_method='Euler'):

        dynamics = Dubins3dDynamics(v_min=v_min, v_max=v_max, integration_method=integration_method)
        actuator_set = Dubins3dActuatorSet()
        state = Dubins3dState()

        super().__init__(name, dynamics, actuator_set, state, controller)

    def generate_info(self):
        info = {
            'gamma': self.gamma,
            'roll': self.roll,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret


class Dubins3dState(BaseDubinsState):

    def build_vector(self, x=0, y=0, z=0, heading=0, gamma=0, roll=0, v=100, **kwargs):
        return np.array([x, y, z, heading, gamma, roll, v], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @x.setter
    def x(self, value):
        self._vector[0] = value

    @property
    def y(self):
        return self._vector[1]

    @y.setter
    def y(self, value):
        self._vector[1] = value

    @property
    def z(self):
        return self._vector[2]

    @z.setter
    def z(self, value):
        self._vector[2] = value

    @property
    def heading(self):
        return self._vector[3]

    @heading.setter
    def heading(self, value):
        self._vector[3] = value

    @property
    def gamma(self):
        return self._vector[4]

    @gamma.setter
    def gamma(self, value):
        self._vector[4] = value

    @property
    def roll(self):
        return self._vector[5]

    @roll.setter
    def roll(self, value):
        self._vector[5] = value

    @property
    def v(self):
        return self._vector[6]

    @v.setter
    def v(self, value):
        self._vector[6] = value

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:3] = self._vector[0:3]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler('ZYX', [self.yaw, self.pitch, self.roll])


class Dubins3dActuatorSet(BaseActuatorSet):

    def __init__(self):

        actuators = [
            ContinuousActuator(
                'ailerons',
                [np.deg2rad(-6), np.deg2rad(6)],
                0
            ),
            ContinuousActuator(
                'elevator',
                [np.deg2rad(-6), np.deg2rad(6)],
                0
            ),
            ContinuousActuator(
                'throttle',
                [-10, 10],
                0
            )
        ]

        super().__init__(actuators)


class Dubins3dDynamics(BaseODESolverDynamics):

    def __init__(
            self, v_min=10, v_max=100,
            roll_min=-math.pi/3, roll_max=math.pi/3, gamma_min=-math.pi/9, gamma_max=math.pi/9,
            g=32.17,
            *args, **kwargs):
        self.v_min = v_min
        self.v_max = v_max
        self.roll_min = roll_min
        self.roll_max = roll_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.g = g

        super().__init__(*args, **kwargs)

    def step(self, step_size, state, control):
        state = super().step(step_size, state, control)

        # enforce velocity limits
        if state.v < self.v_min or state.v > self.v_max:
            state.v = max(min(state.v, self.v_max), self.v_min)

        # enforce roll limits
        if state.roll < self.roll_min or state.roll > self.roll_max:
            state.roll = max(min(state.roll, self.roll_max), self.roll_min)

        # enforce gamma limits
        if state.gamma < self.gamma_min or state.gamma > self.gamma_max:
            state.gamma = max(min(state.gamma, self.gamma_max), self.gamma_min)

        return state

    def dx(self, t, state_vec, control):
        x, y, z, heading, gamma, roll, v = state_vec

        elevator, ailerons, throttle = control

        # enforce velocity limits
        if v <= self.v_min and throttle < 0:
            throttle = 0
        elif v >= self.v_max and throttle > 0:
            throttle = 0

        # enforce roll limits
        if roll <= self.roll_min and ailerons < 0:
            ailerons = 0
        elif roll >= self.roll_max and ailerons > 0:
            ailerons = 0

        if gamma <= self.gamma_min and elevator < 0:
            elevator = 0
        elif gamma >= self.gamma_max and elevator > 0:
            elevator = 0

        x_dot = v * math.cos(heading) * math.cos(gamma)
        y_dot = v * math.sin(heading) * math.cos(gamma)
        z_dot = -1 * v * math.sin(gamma)

        gamma_dot = elevator
        roll_dot = ailerons
        heading_dot = (self.g / v) * math.tan(roll)                      # g = 32.17 ft/s^2
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, z_dot, heading_dot, gamma_dot, roll_dot, v_dot], dtype=np.float64)

        return dx_vec
