# import math

# # If we want to change this, will need to adapt data.get_vox_coords_from_view
# N_ROTS = 1

# def idx_to_x_z_rot(n: int, n_rots: int = N_ROTS):
#     # Parameterize square spiral as on https://math.stackexchange.com/a/3158068. Also take into account rotations at 
#     # each location.
#     rot = n % n_rots
#     n = n // n_rots
#     x, z = square_spiral(n)
#     rot = int(rot * 360 / n_rots)
#     return x, z, rot
    

# def square_spiral(n: int):
#     flr_sqrt_n = math.floor(math.sqrt(n))
#     n_hat = flr_sqrt_n if flr_sqrt_n % 2 == 0 else flr_sqrt_n - 1
#     n_hat_sqr = n_hat ** 2
#     if n_hat_sqr <= n <= n_hat_sqr + n_hat:
#         x = - n_hat / 2 + n - n_hat_sqr
#         z = n_hat / 2
#     elif n_hat_sqr + n_hat < n <= n_hat_sqr + 2 * n_hat + 1:
#         x = n_hat / 2
#         z = n_hat / 2 - n + n_hat_sqr + n_hat
#     elif n_hat_sqr + 2 * n_hat + 1 < n <= n_hat_sqr + 3 * n_hat + 2:
#         x = n_hat / 2 - n + n_hat_sqr + 2 * n_hat + 1
#         z = - n_hat / 2 - 1
#     elif n_hat_sqr + 3 * n_hat + 2 < n <= n_hat_sqr + 4 * n_hat + 3:
#         x = - n_hat / 2 - 1
#         z = - n_hat / 2 - 1 + n - n_hat_sqr - 3 * n_hat - 2
#     else: 
#         raise Exception
#     return int(x), int(z)


# def get_vox_xz_from_view(x, z, rot):
#     """Which chunk of voxels do we care about given the view in the screenshot?"""
#     # NOTE: Hardcoded: chunk is 10 tiles (L_inf distance) "in front" of camera
#     offset = 10

#     if rot == 0:
#         return x, z + offset
#     if rot == 45:
#         return x - offset, z + offset
#     if rot == 90:
#         return x - offset, z
#     if rot == 135:
#         return x - offset, z - offset
#     if rot == 180:
#         return x, z - offset
#     if rot == 225:
#         return x + offset, z - offset
#     if rot == 270:
#         return x + offset, z
#     if rot == 315:
#         return x + offset, z + offset
#     else:
#         raise ValueError(f"Invalid rotation: {rot}")