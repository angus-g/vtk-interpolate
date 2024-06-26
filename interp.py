import sys

import numpy as np
import pyvista as pv
import xarray as xr

cylindrical = True

nx, ny, nz = 60, 45, 30
if cylindrical:
    grid_shape = nz, nx
else:
    grid_shape = nz, nx, ny

def interpolate(filename):
    theta = np.linspace(0, 360, nx+1)[:-1]
    phi = np.linspace(0, 180, ny)
    r = np.linspace(1.208, 2.208, nz)

    x, y, z = np.meshgrid(np.radians(theta), 0 if cylindrical else np.radians(phi), r)
    phi_factor = 1 if cylindrical else np.sin(y)

    x_cart = z * np.cos(x) * phi_factor
    y_cart = z * np.sin(x) * phi_factor

    if cylindrical:
        z_cart = y
    else:
        z_cart = z * np.cos(y)

    cylinder_grid = pv.StructuredGrid(x_cart, y_cart, z_cart)

    g = pv.read(filename)
    # either interpolate for weighted, or sample for direct
    sampled = cylinder_grid.sample(g)

    coords = {"lon": theta, "depth": r}
    if cylindrical:
        coord_list = ("depth", "lon")
    else:
        coord_list = ("depth", "lon", "lat")
        coords["lat"] = phi - 90

    ds = xr.Dataset(coords=coords)

    for name, data in sampled.point_data.items():
        # skip info arrays
        if name.startswith("vtk"):
            continue

        if len(data.shape) > 1:
            # vector data
            for suffix, component in zip("xyz", data.T):
                ds[f"{name}_{suffix}"] = (coord_list, component.reshape(grid_shape))

        else:
            ds[name] = (coord_list, data.reshape(grid_shape))

    ds["lon"].attrs["units"] = "degrees_east"
    ds["lon"].attrs["axis"] = "X"
    ds["depth"].attrs["axis"] = "Z"
    ds["depth"].attrs["positive"] = "up"

    if not cylindrical:
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lat"].attrs["axis"] = "Y"

    return ds


if __name__ == "__main__":
    ds = interpolate(sys.argv[1])
    ds.to_netcdf("interp.nc")
