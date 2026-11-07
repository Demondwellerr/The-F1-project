import numpy as np
from scipy.interpolate import splprep, splev


def generate_track(seed=0, n_points=400, n_control=12, radius=300, width=14):
    rng = np.random.default_rng(seed)

    for attempt in range(500):
        # 1. lumpy blob: evenly spaced angles with a little jitter, random radii
        angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)
        angles += rng.uniform(-0.2, 0.2, n_control)
        radii = rng.uniform(0.5, 1.0, n_control) * radius
        pts = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
        pts = np.vstack([pts, pts[:1]])  # close the loop

        # 2. smooth closed curve
        tck, _ = splprep(pts.T, s=0, per=True)

        # 3. resample at evenly spaced distances along the curve
        u = np.linspace(0, 1, 5000)
        x, y = splev(u, tck)
        dist = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        u_even = np.interp(np.linspace(0, dist[-1], n_points, endpoint=False), dist, u)
        x, y = splev(u_even, tck)
        dx, dy = splev(u_even, tck, der=1)
        ddx, ddy = splev(u_even, tck, der=2)

        # curvature at each point (1 / corner radius)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

        # 4. reject tracks with corners tighter than ~2x the track width
        if np.max(np.abs(curvature)) > 1 / (2 * width):
            continue

        # tangent and normal (the sideways direction)
        tangent = np.c_[dx, dy]
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]

        center = np.c_[x, y]
        left = center + normal * width / 2
        right = center - normal * width / 2
        return center, left, right, curvature, width

    raise RuntimeError("Could not generate a valid track, try a different seed")