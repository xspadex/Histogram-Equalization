import cv2 as cv
import numpy as np

def CLAHE(src, clip, tiles):
    hist_size = 256
    tiles_x = tiles[0]
    tiles_y = tiles[1]
    tile_hist = np.zeros(hist_size, dtype=np.uint32)

    if src.shape[1] % tiles_x == 0 and src.shape[0] % tiles_y == 0:
        tile_size = (src.shape[1] // tiles_x, src.shape[0] // tiles_y)
        src_for_lut = src.copy()
    else:
        extra_x = tiles_x - (src.shape[1] % tiles_x)
        extra_y = tiles_y - (src.shape[0] % tiles_y)
        src_ext = cv.copyMakeBorder(src, 0, extra_y, 0, extra_x, cv.BORDER_REFLECT_101)
        tile_size = (src_ext.shape[1] // tiles_x, src_ext.shape[0] // tiles_y)
        src_for_lut = src_ext

    clip_limit = max(int(clip * tile_size[0] * tile_size[1] / hist_size), 1)
    lut = np.zeros((tiles_y * tiles_x, hist_size), dtype=np.uint32)
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_hist.fill(0)
            x1, y1 = tx * tile_size[0], ty * tile_size[1]
            tile = src_for_lut[y1:y1 + tile_size[1], x1:x1 + tile_size[0]].flatten()

            for val in tile:
                tile_hist[val] += 1

            if clip_limit > 0:
                clipped = 0
                for i in range(hist_size):
                    if tile_hist[i] > clip_limit:
                        clipped += tile_hist[i] - clip_limit
                        tile_hist[i] = clip_limit

                redist_batch = clipped // hist_size
                residual = clipped - redist_batch * hist_size
                tile_hist += redist_batch

                if residual > 0:
                    residual_step = max(hist_size // residual, 1)
                    for i in range(0, hist_size, residual_step):
                        if residual > 0:
                            tile_hist[i] += 1
                            residual -= 1

            lut_scale = (hist_size - 1) / tile.size
            lut[ty * tiles_x + tx] = (np.cumsum(tile_hist) * lut_scale).astype(np.uint8)


    dst = np.zeros_like(src)
    inv_tile_w = 1 / tile_size[0]
    inv_tile_h = 1 / tile_size[1]

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            pix = src[i, j]
            txf = j * inv_tile_w - 0.5
            tx1 = int(txf)
            tx2 = min(tx1 + 1, tiles_x - 1)

            px1 = txf - tx1
            px2 = 1.0 - px1

            tx1 = max(tx1, 0)

            tyf = i * inv_tile_h - 0.5
            ty1 = int(tyf)
            ty2 = min(ty1 + 1, tiles_y - 1)

            py1 = tyf - ty1
            py2 = 1.0 - py1

            ty1 = max(ty1, 0)

            tile_lut_y1x1 = lut[ty1 * tiles_x + tx1]
            tile_lut_y1x2 = lut[ty1 * tiles_x + tx2]
            tile_lut_y2x1 = lut[ty2 * tiles_x + tx1]
            tile_lut_y2x2 = lut[ty2 * tiles_x + tx2]

            interp_val = (
                (tile_lut_y1x1[pix] * px2 + tile_lut_y1x2[pix] * px1) * py2 +
                (tile_lut_y2x1[pix] * px2 + tile_lut_y2x2[pix] * px1) * py1
            )
            dst[i, j] = np.clip(interp_val, 0, 255).astype(src.dtype)

    return dst