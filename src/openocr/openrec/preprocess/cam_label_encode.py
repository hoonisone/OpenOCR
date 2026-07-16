import numpy as np
import cv2
from .ar_label_encode import ARLabelEncode


def crop_safe(arr, rect, bbs=[], pad=0):
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2 * pad
    v0 = [max(0, rect[0]), max(0, rect[1])]
    v1 = [
        min(arr.shape[0], rect[0] + rect[2]),
        min(arr.shape[1], rect[1] + rect[3])
    ]
    arr = arr[v0[0]:v1[0], v0[1]:v1[1], ...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= v0[0]
            bbs[i, 1] -= v0[1]
        return arr, bbs
    else:
        return arr


try:
    # pygame==2.5.2
    import pygame
    from pygame import freetype
except:
    pass


class CAMLabelEncode(ARLabelEncode):
    """Convert between text-label and text-index."""

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 font_path=None,
                 font_size=30,
                 font_strength=0.1,
                 image_shape=[32, 128],
                 **kwargs):
        super(CAMLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)
        if image_shape is None or len(image_shape) < 2:
            raise ValueError(
                f"CAMLabelEncode requires image_shape=[h, w], got: {image_shape}"
            )
        self.image_shape = [int(image_shape[0]), int(image_shape[1])]
        if self.image_shape[0] <= 0 or self.image_shape[1] <= 0:
            raise ValueError(
                f"CAMLabelEncode image_shape must be positive, got: {self.image_shape}"
            )

        if font_path is not None:
            if "freetype" not in globals():
                raise ImportError(
                    "pygame.freetype is not available. Install pygame first."
                )
            freetype.init()
            # init font
            self.font = freetype.Font(font_path)
            self.font.antialiased = True
            self.font.origin = True

            # choose font style
            self.font.size = font_size
            self.font.underline = False

            self.font.strong = True
            self.font.strength = font_strength
            self.font.oblique = False

    def render_normal(self, font, text):
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1

        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0 * line_bounds.width),
                 round(1.25 * line_spacing * len(lines)))
        src_alpha_flag = getattr(pygame, "SRCALPHA", None)
        if src_alpha_flag is None:
            # Compatibility fallback for environments where pygame.locals
            # is unavailable or not exposed.
            src_alpha_flag = 0
        surf = pygame.Surface(fsize, src_alpha_flag, 32)

        bbs = []
        space = font.get_rect('O')
        # space = font.get_rect(' ')
        x, y = 0, 0
        for l in lines:
            x = 2  # carriage-return
            y += line_spacing  # line-feed

            for ch in l:  # render each character
                if ch.isspace():  # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x, y), ch)
                    # ch_bounds.x = x + ch_bounds.x
                    # ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width + 5
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        # words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf),
                                  rect_union,
                                  bbs,
                                  pad=5)
        surf_arr = surf_arr.swapaxes(0, 1)

        # self.visualize_bb(surf_arr,bbs)
        return surf_arr, bbs

    def __call__(self, data):
        data = super().__call__(data=data)
        if data is None:
            return None
        target_h = max(1, self.image_shape[0] // 2)
        target_w = max(1, self.image_shape[1] // 2)
        mask_channels = max(1, len(self.character) - 3)

        word = []
        for c in data['label'][1:data['length'] + 1]:
            word.append(self.character[c])
        word = ''.join(word)
        if len(word) == 0:
            data['binary_mask'] = np.zeros(
                (target_h, target_w, mask_channels), dtype=np.float32)
            return data

        # binary mask
        binary_mask, bbs = self.render_normal(self.font, word)
        if binary_mask.size == 0 or binary_mask.shape[0] <= 0 or binary_mask.shape[
                1] <= 0:
            data['binary_mask'] = np.zeros(
                (target_h, target_w, mask_channels), dtype=np.float32)
            return data

        if binary_mask.size == 0 or binary_mask.shape[0] <= 0 or binary_mask.shape[
                1] <= 0:
            binary_mask = np.zeros((target_h, target_w, mask_channels),
                                   dtype=np.float32)
        else:
            src_h, src_w = binary_mask.shape[:2]
            resized_gray = cv2.resize(binary_mask, (target_w, target_h))
            cate_aware_surf = np.zeros((target_h, target_w, mask_channels),
                                       dtype=np.uint8)
            scale_x = float(target_w) / float(max(1, src_w))
            scale_y = float(target_h) / float(max(1, src_h))

            for id, bb in zip(data['label'][1:data['length'] + 1], bbs):
                char_id = id - 1
                if char_id < 0 or char_id >= mask_channels:
                    continue
                x0 = int(np.floor(bb[0] * scale_x))
                y0 = int(np.floor(bb[1] * scale_y))
                x1 = int(np.ceil((bb[0] + bb[2]) * scale_x))
                y1 = int(np.ceil((bb[1] + bb[3]) * scale_y))

                x0 = max(0, min(target_w, x0))
                y0 = max(0, min(target_h, y0))
                x1 = max(0, min(target_w, x1))
                y1 = max(0, min(target_h, y1))
                if x1 <= x0 or y1 <= y0:
                    continue

                cate_aware_surf[y0:y1, x0:x1, char_id] = resized_gray[y0:y1,
                                                                      x0:x1]
            binary_mask = cate_aware_surf
        if np.max(binary_mask) > 0:
            binary_mask = binary_mask / np.max(binary_mask)  # [0 ~ 1]
            binary_mask = binary_mask.astype(np.float32)
        data['binary_mask'] = binary_mask
        return data
