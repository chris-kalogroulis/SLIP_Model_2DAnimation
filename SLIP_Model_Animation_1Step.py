# p = cartesian coords = [x,y] = [r cos(th), r sin(th)]
# q = generalised coords = [r,θ] = [r,th]

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------
# Model + helpers
# ----------------------------------------------------------

@dataclass
class SLIPParams:
    m: float = 6.0    # mass
    g: float = 9.81   # gravity
    k: float = 1800.0 # spring constant
    r0: float = 0.5   # rest length

def p_from_q(r, th, xfoot = 0.0):
    # p = [x,y] = [r cos(th), r sin(th)]
    x = xfoot + r * np.cos(th)
    y = r * np.sin(th)
    return x, y

def jacobian_p_wrt_q(r, th):
    # J = dp/dq = [dx/dr  dx/dth]
    #             [dy/dr  dy/dth]
    # p' = Jq'
    return np.array([
        [np.cos(th), -r * np.sin(th)],
        [np.sin(th),  r * np.cos(th)]
    ])

def qdot_from_pdot(r, th, pdot):
    # p' = Jq' 
    # so solving q' using sim. equations method
    J = jacobian_p_wrt_q(r, th)
    return np.linalg.solve(J, pdot)

def pdot_from_qdot(r, th, rdot, thdot):
    # p' = Jq'
    J = jacobian_p_wrt_q(r, th)
    qdot = np.array([rdot, thdot])
    pdot = np.matmul(J,qdot)
    return pdot[0], pdot[1]

# ----------------------------------------------------------
# Dynamics
# ----------------------------------------------------------

def stance_dynamics(t, X, P: SLIPParams):
    # X = state vector during slip dynamics
    # X = [r, th, rdot, thdot]
    r, th, rdot, thdot = X

    rdd  = r*thdot**2 - P.g*np.sin(th) - (P.k/P.m)*(r-P.r0)
    thdd = -(2/r)*rdot*thdot - (P.g/r)*np.cos(th)

    # return X'
    return np.array([rdot, thdot, rdd, thdd])

def liftoff_event(t, X, P: SLIPParams):
    # liftoff when r returns to r0 while increasing (after compression)
    r = X[0]
    return r - P.r0
liftoff_event.terminal = True
liftoff_event.direction = 1 # detect upward crossing only

def flight_dynamics(t, Y, P: SLIPParams):
    # Y = state vector during flight dynamics
    # Y = [x, y, xdot, ydot]
    x, y, xdot, ydot = Y

    xdd = 0
    ydd = -P.g

    # return Y'
    return np.array([xdot, ydot, xdd, ydd])

def touchdown_event(theta0):
    # touchdown when y hits r0*sin(theta0) while descending
    def ev(t, Y, P: SLIPParams):
        y = Y[1]
        return y - P.r0*np.sin(theta0)
    
    ev.terminal = True
    ev.direction = -1
    return ev

# ----------------------------------------------------------
# One step simulation (stance -> flight)
# ----------------------------------------------------------

def simulate_one_step(P: SLIPParams, v=2.0, alpha=0.1, theta0=1.2,
                      tmax_stance=2.0, tmax_flight=2.0):
    """
    v: initial centre-of-mass speed at touchdown
    alpha: direction angle of velocity (radians) of centre of mass at touchdown, 
           pdot @ touchdown = v*[cos(alpha), -sin(alpha)] (downward for +alpha)
    theta0: touchdown leg angle from horizontal (radians)
    """

    # ---- stance initial conditions ----
    r_init = P.r0 * (1 - 1e-9)
    th_init = theta0

    # x, y velocity @ touchdown
    pdot_TD = v * np.array([np.cos(alpha), -np.sin(alpha)])


    rdot_init, thdot_init = qdot_from_pdot(r_init, th_init, pdot_TD)

    X0 = np.array([r_init, th_init, rdot_init, thdot_init])

    def lo_event(t, X):
        return liftoff_event(t, X, P)

    lo_event.terminal = True
    lo_event.direction = 1

    sol_st = solve_ivp(
        fun=lambda t, X: stance_dynamics(t, X, P),
        # is the same as:
        # def f(t,x):
        #     return stance_dynamics(t, X, P)
        t_span=(0.0, tmax_stance),
        y0=X0,
        events= lo_event,
        max_step=1e-2,
        rtol=1e-8, atol=1e-10
    )

    # Convert stance trajectory to global Cartesian
    xfoot = 0.0
    xs_st, ys_st = [], []
    for r, th in sol_st.y[:2].T:
        x, y = p_from_q(r, th, xfoot=xfoot)
        xs_st.append(x)
        ys_st.append(y)
    xs_st = np.array(xs_st)
    ys_st = np.array(ys_st)

    # State at liftoff (end of stance)
    rL, thL, rdotL, thdotL = sol_st.y[:, -1]
    xL, yL = p_from_q(rL, thL, xfoot=xfoot)
    xdotL, ydotL = pdot_from_qdot(rL, thL, rdotL, thdotL)


    # ---- flight initial conditions ----
    Y0 = np.array([xL, yL, xdotL, ydotL])

    def td_event(t, Y):
        return touchdown_event(theta0)(t, Y, P)

    td_event.terminal = True
    td_event.direction = -1

    sol_fl = solve_ivp(
        fun= lambda t, Y: flight_dynamics(t, Y, P),
        t_span= (sol_st.t[-1], sol_st.t[-1] + tmax_flight),
        y0= Y0,
        events= td_event,
        max_step= 1e-2,
        rtol= 1e-8, atol= 1e-10
    )

    xs_fl = sol_fl.y[0]
    ys_fl = sol_fl.y[1]

    # Foot placement at touchdown (new stance foot) based on geometry:
    # x = xfoot_next + r0*cos(theta0) -> xfoot_next = x - r0*cos(theta0)
    xTD = xs_fl[-1]
    xfoot_next = xTD - P.r0 * np.cos(theta0)

    return {
        "stance": sol_st,
        "flight": sol_fl,
        "stance_xy": (xs_st, ys_st),
        "flight_xy": (xs_fl, ys_fl),
        "xfoot0": xfoot,
        "xfoot_next": xfoot_next,
        "theta0": theta0
    }

# ----------------------------------------------------------
# Animation
# ----------------------------------------------------------
def animate_step(sim, P: SLIPParams, fps=60):

    xs_st, ys_st = sim["stance_xy"]
    xs_fl, ys_fl = sim["flight_xy"]
    xfoot0       = sim["xfoot0"]
    xfoot_next   = sim["xfoot_next"]
    theta0       = sim["theta0"]

    # Build a single timeline of frames
    xs = np.concatenate([xs_st, xs_fl])
    ys = np.concatenate([ys_st, ys_fl])

    phase = np.concatenate([np.zeros_like(xs_st, dtype=int),
                            np.ones_like(xs_fl, dtype=int)])
    
    # For stance: foot is xfoot0. For flight: show "planned touchdown foot" xfoot_next.
    xfoot_series = np.where(phase == 0, xfoot0, xfoot_next)

    # Plot setup
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    # ground line
    ax.plot([xs.min()-1, xs.max()+1], [0,0])

    # artists
    mass_pt,  = ax.plot([], [], marker="o", markersize=10)
    leg_ln,   = ax.plot([], [], linewidth=2)
    trail_ln, = ax.plot([], [], linewidth=1)

    # limits
    ax.set_xlim(xs.min()-0.6, xs.max()+0.6)
    ax.set_ylim(-0.1, max(ys.max()+0.6, P.r0+0.4))

    def init():
        mass_pt.set_data([],[])
        leg_ln.set_data([],[])
        trail_ln.set_data([],[])
        return mass_pt, leg_ln, trail_ln
    
    def update(i):
        x, y = xs[i], ys[i]
        xf = xfoot_series[i]

        # leg: during stance it connects to contact
        #      during flight we connect to next foot placement
        leg_ln.set_data([xf, x], [0.0, y])

        mass_pt.set_data([x], [y])
        trail_ln.set_data([xs[:i+1], ys[:i+1]])

        ax.set_title("SLIP: stance + flight")
        return mass_pt, leg_ln, trail_ln
    
    interval_ms = 1000/fps
    ani = FuncAnimation(fig, update, frames=len(xs), init_func=init,
                        interval=interval_ms, blit=True)
    
    plt.show()

# ----------------------------------------------------------
# Run demo
# ----------------------------------------------------------
if __name__ == "__main__":
    P = SLIPParams(m=6.0, g=9.81, k=1800.0, r0=0.5)

    sim = simulate_one_step(P, v=2.5, alpha=0.15, theta0=2.0)
    animate_step(sim, P)