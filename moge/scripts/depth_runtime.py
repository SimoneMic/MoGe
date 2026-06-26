import faulthandler
faulthandler.enable()
import yarp
from yarp import BufferedPortImageRgb, BufferedPortImageMono, ImageRgb, ImageMono, ImageRgbCallback, Log
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from threading import Lock

log = Log()

class ImageCallback(ImageRgbCallback):
    def __init__(self, max_depth : float = 8.0):
        super().__init__()
        self.is_initialized = False
        self.depth_port = BufferedPortImageMono()
        self.depth_port.open("/moGe/depth:o")
        self.mutex = Lock()
        self.max_depth = max_depth

    def onRead(self, img: ImageRgb, reader=None):
        if not self.is_initialized:
            log.error("Module not initialized: call init() before enabling the port!")
            return
        with self.mutex:
            try:
                rgb_np = self._yarp_rgb_to_numpy(img)                 # (H, W, 3) uint8
                rgb = torch.from_numpy(rgb_np).to(self.device)
                rgb = rgb.permute(2, 0, 1).float() / 255.0            # (3, H, W) float32 in [0, 1]
                output = self.moge.infer(rgb, num_tokens=self.token_number)
                depth = output["depth"].cpu().numpy().astype(np.float32)  # (H, W) metric scale
                norm = np.clip(depth / self.max_depth, 0.0, 1.0)               # invalid (+inf) → 1.0
                depth_u8 = (norm * 255).astype(np.uint8)                    # (H, W)
                depth_img = self._numpy_to_yarp_mono(depth_u8)
                out = self.depth_port.prepare()
                out.copy(depth_img)
                self.depth_port.write()
            except Exception as ex:
                log.error(f"Got exception: {ex}")

    @staticmethod
    def _yarp_rgb_to_numpy(img: ImageRgb) -> np.ndarray:
        """Copy a yarp.ImageRgb's pixel buffer into a (H, W, 3) uint8 numpy array."""
        w, h = img.width(), img.height()
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        wrapper = ImageRgb()
        wrapper.resize(w, h)
        # Pass arr.data (a memoryview): this selects yarp's PyObject* setExternal
        # overload. Passing the bare ndarray matches the void* overload, which
        # treats the object pointer as a raw address and segfaults.
        wrapper.setExternal(arr.data, w, h)  # wrapper's pixel buffer now points at arr's memory
        wrapper.copy(img)                     # copies img's pixels into arr via the wrapper
        return arr

    @staticmethod
    def _numpy_to_yarp_mono(depth: np.ndarray) -> ImageMono:
        """Wrap a (H, W) uint8 numpy array as a yarp.ImageMono (no copy)."""
        h, w = depth.shape
        depth = np.ascontiguousarray(depth, dtype=np.uint8)
        img = ImageMono()
        img.resize(w, h)
        img.setExternal(depth.data, w, h)
        img._keepalive = depth  # pin numpy buffer to the image's lifetime (external = no copy)
        return img

    def init(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(self.device)
        # The model range is [1200, 3600]. The default resolution_level=9 uses the max (3600) number of tokens. More the tokens, more accurate and slower is the model
        self.token_number = 3000
        self.is_initialized = True

    def close(self):
        self.depth_port.close()

def main():
    yarp.Network.init()
    rgb_port = BufferedPortImageRgb()
    callback = ImageCallback()
    callback.init()
    rgb_port.useCallback(callback)
    rgb_port.open("/moGe/rgb:i")

    try:
        while True:
            yarp.delay(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        rgb_port.close()
        callback.close()
        yarp.Network.fini()

if __name__ == "__main__":
    main()
