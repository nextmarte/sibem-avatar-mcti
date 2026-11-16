try:
    import pyximport

    pyximport.install()
    from .blend import blend_images_cy
except Exception:
    import numpy as np

    def blend_images_cy(mask_warped, frame_warped, frame_rgb, result):
        m = mask_warped[:, :, None]
        blended = m * frame_warped + (1.0 - m) * frame_rgb
        np.clip(blended, 0, 255, out=blended)
        result[:] = blended.astype(np.uint8)
