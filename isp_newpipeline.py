import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

# -------------------- Base ISP Stage --------------------
class ISPStage:
    def __init__(self, name):
        self.name = name

    def process(self, image):
        start_time = time.time()
        result = self._process_implementation(image)
        end_time = time.time()
        print(f"{self.name} processed in {end_time - start_time:.4f} seconds")
        return result

    def _process_implementation(self, image):
        raise NotImplementedError

# -------------------- ISP Stages --------------------
class RawDataCapture(ISPStage):
    def __init__(self):
        super().__init__("Raw Data Capture")

    def _process_implementation(self, image_path):
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            return image
        return image_path

class Demosaicing(ISPStage):
    def __init__(self):
        super().__init__("Demosaicing")

    def _process_implementation(self, image):
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        return image

class WhiteBalance(ISPStage):
    def __init__(self, method='gray_world'):
        super().__init__("White Balance")
        self.method = method

    def _process_implementation(self, image):
        if self.method == 'gray_world':
            b, g, r = cv2.split(image.astype(np.float32))
            b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
            gray = (b_avg + g_avg + r_avg) / 3
            b, g, r = b * (gray / b_avg), g * (gray / g_avg), r * (gray / r_avg)
            return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)
        return image

class ColorCorrection(ISPStage):
    def __init__(self, strength=1.0):
        super().__init__("Color Correction")
        self.strength = strength

    def _process_implementation(self, image):
        correction_matrix = np.array([
            [1.1, -0.05, -0.05],
            [-0.05, 1.05, -0.05],
            [-0.05, -0.05, 1.1]
        ]) * self.strength + np.eye(3) * (1 - self.strength)
        img_float = image.astype(np.float32)
        result = np.zeros_like(img_float)
        for i in range(3):
            result[:,:,i] = sum(img_float[:,:,j] * correction_matrix[i,j] for j in range(3))
        return np.clip(result, 0, 255).astype(np.uint8)

class LensShading(ISPStage):
    def __init__(self):
        super().__init__("Lens Shading")

    def _process_implementation(self, image):
        h, w = image.shape[:2]
        y, x = np.indices((h, w))
        radius = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        correction = 1 + 0.3 * (radius / np.max(radius))**2
        result = image.astype(np.float32)
        for i in range(3):
            result[:,:,i] *= correction
        return np.clip(result, 0, 255).astype(np.uint8)

class GammaCorrection(ISPStage):
    def __init__(self, gamma=2.2):
        super().__init__("Gamma Correction")
        self.gamma = gamma

    def _process_implementation(self, image):
        img_float = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(img_float, 1.0 / self.gamma)
        return np.clip(gamma_corrected * 255.0, 0, 255).astype(np.uint8)

class ToneMapping(ISPStage):
    def __init__(self, method='reinhard'):
        super().__init__("Tone Mapping")
        self.method = method

    def _process_implementation(self, image):
        if self.method == 'reinhard':
            img_float = image.astype(np.float32) / 255.0
            L = 0.2126 * img_float[:,:,2] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,0]
            L_mapped = L / (1 + L)
            for i in range(3):
                img_float[:,:,i] *= L_mapped / (L + 1e-6)
            return np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
        return image

class ImageSharpening(ISPStage):
    def __init__(self, amount=0.5, radius=0.7):
        super().__init__("Image Sharpening")
        self.amount = amount
        self.radius = radius

    def _process_implementation(self, image):
        blur = cv2.GaussianBlur(image, (0, 0), self.radius)
        return cv2.addWeighted(image, 1.0 + self.amount, blur, -self.amount, 0)

class DistortionCorrection(ISPStage):
    def __init__(self):
        super().__init__("Distortion Correction")

    def _process_implementation(self, image):
        h, w = image.shape[:2]
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0, 0, 0])
        return cv2.undistort(image, K, dist)

class NoiseCorrection(ISPStage):
    def __init__(self):
        super().__init__("Noise Correction")

    def _process_implementation(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

class ExposureCompensation(ISPStage):
    def __init__(self, target_mean=128):
        super().__init__("Exposure Compensation")
        self.target_mean = target_mean

    def _process_implementation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_mean = np.mean(gray)
        factor = self.target_mean / max(current_mean, 1)
        return np.clip(image * factor, 0, 255).astype(np.uint8)

# -------------------- Pipeline Classes --------------------
class ISPPipeline:
    def __init__(self, name):
        self.name = name
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)
        return self

    def process(self, image):
        print(f"\nProcessing {self.name} Pipeline")
        for stage in self.stages:
            image = stage.process(image)
        return image

# -------------------- Pipelines --------------------
def create_hv_pipeline():
    pipeline = ISPPipeline("Human Vision")
    pipeline.add_stage(RawDataCapture())
    pipeline.add_stage(Demosaicing())
    pipeline.add_stage(WhiteBalance(method='adaptive'))
    pipeline.add_stage(ColorCorrection(strength=1.0))
    pipeline.add_stage(LensShading())
    pipeline.add_stage(ExposureCompensation(target_mean=128))
    pipeline.add_stage(GammaCorrection(gamma=2.2))
    pipeline.add_stage(ToneMapping(method='reinhard'))
    pipeline.add_stage(ImageSharpening(amount=0.5, radius=0.7))
    pipeline.add_stage(DistortionCorrection())
    return pipeline

def create_cv_pipeline():
    pipeline = ISPPipeline("Computer Vision")
    pipeline.add_stage(RawDataCapture())
    pipeline.add_stage(Demosaicing())
    pipeline.add_stage(WhiteBalance(method='gray_world'))
    pipeline.add_stage(ColorCorrection(strength=0.8))
    pipeline.add_stage(NoiseCorrection())
    pipeline.add_stage(ExposureCompensation(target_mean=128))
    pipeline.add_stage(GammaCorrection(gamma=1.8))
    pipeline.add_stage(ToneMapping(method='reinhard'))
    pipeline.add_stage(DistortionCorrection())
    return pipeline

# -------------------- Concurrent Processing --------------------
def process_with_concurrent_pipelines(image_path, pipelines):
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(p.process, image_path): name for name, p in pipelines.items()}
        results = {}
        for future in futures:
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"Error in {name} pipeline: {str(e)}")
                results[name] = None
        return results

# -------------------- Main Entry --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python isp_pipeline_clean.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    pipelines = {
        "HV": create_hv_pipeline(),
        "CV": create_cv_pipeline()
    }

    results = process_with_concurrent_pipelines(image_path, pipelines)
    for name, result in results.items():
        if result is not None:
            output_path = f"clean_output_{name}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved {name} processed image to {output_path}")
