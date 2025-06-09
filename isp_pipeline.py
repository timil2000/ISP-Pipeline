import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ISPStage:
    """Base class for all ISP pipeline stages"""
    def __init__(self, name):
        self.name = name
        
    def process(self, image):
        """Process the image and return the result"""
        start_time = time.time()
        result = self._process_implementation(image)
        end_time = time.time()
        logger.debug(f"{self.name} processing time: {end_time - start_time:.4f} seconds")
        return result
    
    def _process_implementation(self, image):
        """Implementation of the specific processing stage"""
        raise NotImplementedError("Each stage must implement this method")


class RawDataCapture(ISPStage):
    def __init__(self):
        super().__init__("Raw Data Capture")
        
    def _process_implementation(self, image_path):
        """Read raw image data"""
        # In a real implementation, this would handle actual raw sensor data
        # For simplicity, we'll just read an image file
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            return image
        return image_path  # If already an image array, return as is


class Demosaicing(ISPStage):
    def __init__(self):
        super().__init__("Demosaicing")
        
    def _process_implementation(self, image):
        """Convert Bayer pattern to RGB"""
        # Simulate demosaicing - in reality this would handle actual Bayer patterns
        # For simulation, if image has one channel, convert it to three
        if len(image.shape) == 2 or image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        return image


class ColorCorrection(ISPStage):
    def __init__(self, strength=1.0):
        super().__init__("Color Correction")
        self.strength = strength
        
    def _process_implementation(self, image):
        """Apply color correction matrix with enhanced saturation"""
        # Improved color correction matrix (better color reproduction)
        correction_matrix = np.array([
            [1.2, -0.1, -0.1],
            [-0.05, 1.1, -0.05],
            [-0.05, -0.1, 1.15]
        ]) * self.strength + np.eye(3) * (1 - self.strength)
        
        # Convert to float for processing
        img_float = image.astype(np.float32)
        result = np.zeros_like(img_float)
        
        # Apply matrix
        for i in range(3):
            result[:,:,i] = (img_float[:,:,0] * correction_matrix[i,0] + 
                           img_float[:,:,1] * correction_matrix[i,1] + 
                           img_float[:,:,2] * correction_matrix[i,2])
        
        return np.clip(result, 0, 255).astype(np.uint8)


class WhiteBalance(ISPStage):
    def __init__(self, method='gray_world'):
        super().__init__("White Balance")
        self.method = method
        
    def _process_implementation(self, image):
        """Apply white balance correction with multiple methods"""
        if self.method == 'gray_world':
            return self._gray_world_wb(image)
        elif self.method == 'retinex':
            return self._retinex_wb(image)
        else:
            return self._adaptive_wb(image)
            
    def _gray_world_wb(self, image):
        """White balance using gray world assumption"""
        # Similar to existing implementation but with improved handling
        b, g, r = cv2.split(image.astype(np.float32))
        
        # Calculate average values for each channel
        b_avg = np.mean(b)
        g_avg = np.mean(g)
        r_avg = np.mean(r)
        
        # Calculate scaling factors
        gray = (b_avg + g_avg + r_avg) / 3
        b_gain = gray / b_avg if b_avg > 0 else 1.0
        g_gain = gray / g_avg if g_avg > 0 else 1.0
        r_gain = gray / r_avg if r_avg > 0 else 1.0
        
        # Apply gains
        b = b * b_gain
        g = g * g_gain
        r = r * r_gain
        
        # Merge channels and clip to valid range
        balanced = cv2.merge([b, g, r])
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        return balanced
        
    def _retinex_wb(self, image):
        """White balance using Retinex theory"""
        # Single-scale Retinex implementation
        img_float = image.astype(np.float32)
        log_img = np.log1p(img_float)
        
        # Apply Gaussian blur
        blur_img = cv2.GaussianBlur(img_float, (0, 0), sigmaX=25)
        blur_img = np.maximum(blur_img, 1.0)  # Avoid log(0)
        log_blur = np.log1p(blur_img)
        
        # Retinex formula: log(image) - log(blurred image)
        retinex = log_img - log_blur
        
        # Scale the result
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        
        return retinex.astype(np.uint8)
        
    def _adaptive_wb(self, image):
        """Adaptive white balance based on percentile clipping"""
        img_float = image.astype(np.float32)
        result = np.zeros_like(img_float)
        
        # Process each channel separately with percentile clipping
        for i in range(3):
            channel = img_float[:,:,i]
            p_low, p_high = np.percentile(channel, [2, 98])
            
            # Scale the channel based on percentiles
            if p_high > p_low:
                channel = 255 * (channel - p_low) / (p_high - p_low)
            
            result[:,:,i] = channel
            
        return np.clip(result, 0, 255).astype(np.uint8)


class GammaCorrection(ISPStage):
    def __init__(self, gamma=2.2):
        super().__init__("Gamma Correction")
        self.gamma = gamma
        
    def _process_implementation(self, image):
        """Apply gamma correction"""
        # Convert to float for calculation
        image_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        gamma_corrected = np.power(image_float, 1.0/self.gamma)
        
        # Convert back to uint8
        return (gamma_corrected * 255.0).astype(np.uint8)


class NoiseCorrection(ISPStage):
    def __init__(self, method='nlm', strength=10):
        super().__init__("Noise Correction")
        self.method = method
        self.strength = strength
        
    def _process_implementation(self, image):
        """Apply noise reduction with various methods"""
        if self.method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif self.method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif self.method == 'nlm':  # Non-local means (higher quality)
            return cv2.fastNlMeansDenoisingColored(image, None, 
                                                 self.strength, self.strength, 7, 21)
        else:  # Default to median filtering
            return cv2.medianBlur(image, 5)


class PixelErrorDetection(ISPStage):
    def __init__(self, threshold=50):
        super().__init__("Pixel Error Detection")
        self.threshold = threshold
        
    def _process_implementation(self, image):
        """Detect and correct pixel errors"""
        # Simple method to detect outlier pixels
        result = image.copy()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect outliers compared to local neighborhood
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gray, kernel)
        eroded = cv2.erode(gray, kernel)
        
        # Find potential bad pixels
        diff_high = dilated - gray
        diff_low = gray - eroded
        
        # Locations where difference is above threshold
        bad_pixels_high = diff_high > self.threshold
        bad_pixels_low = diff_low > self.threshold
        
        # For detected bad pixels, replace with local median
        if np.any(bad_pixels_high) or np.any(bad_pixels_low):
            median_filtered = cv2.medianBlur(image, 3)
            
            if len(image.shape) == 3:
                for c in range(3):
                    result[:,:,c][bad_pixels_high] = median_filtered[:,:,c][bad_pixels_high]
                    result[:,:,c][bad_pixels_low] = median_filtered[:,:,c][bad_pixels_low]
            else:
                result[bad_pixels_high] = median_filtered[bad_pixels_high]
                result[bad_pixels_low] = median_filtered[bad_pixels_low]
                
        return result


class LensShading(ISPStage):
    def __init__(self):
        super().__init__("Lens Shading")
        
    def _process_implementation(self, image):
        """Correct lens shading effects"""
        # Create a radial gain function to correct vignetting
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create coordinate maps
        y, x = np.indices((height, width))
        
        # Calculate distance from center (normalized)
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        radius = radius / max_radius
        
        # Create a vignetting correction map (stronger correction at edges)
        # This simulates lens shading correction
        correction = 1 + 0.3 * radius**2
        
        # Apply correction
        result = image.copy().astype(np.float32)
        for i in range(image.shape[2] if len(image.shape) == 3 else 1):
            if len(image.shape) == 3:
                result[:,:,i] = image[:,:,i] * correction
            else:
                result = image * correction
                
        # Clip values to valid range and convert back to uint8
        return np.clip(result, 0, 255).astype(np.uint8)


class ToneMapping(ISPStage):
    def __init__(self, method='adaptive_reinhard'):
        super().__init__("Tone Mapping")
        self.method = method
        
    def _process_implementation(self, image):
        """Apply tone mapping to optimize dynamic range"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        if self.method == 'reinhard':
            # Standard Reinhard tone mapping
            L = 0.2126 * img_float[:,:,2] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,0]
            L_mean = np.mean(L)
            
            # Apply tone mapping formula
            L_mapped = L / (1 + L)
            scale = (1 + L_mean) / L_mean
            
            # Apply the same scaling to all channels
            result = img_float.copy()
            for i in range(3):
                result[:,:,i] = img_float[:,:,i] * scale * L_mapped / (L + 1e-6)
                
        elif self.method == 'adaptive_reinhard':
            # Adaptive Reinhard tone mapping with local adaptation
            L = 0.2126 * img_float[:,:,2] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,0]
            
            # Calculate local mean with Gaussian blur
            blur_size = max(3, min(image.shape[0], image.shape[1]) // 100)
            if blur_size % 2 == 0:
                blur_size += 1  # Ensure odd kernel size
                
            L_local = cv2.GaussianBlur(L, (blur_size, blur_size), 0)
            L_local = np.maximum(L_local, 0.001)  # Avoid division by zero
            
            # Adjust mapping based on local luminance
            L_ratio = L / L_local
            L_ratio = np.minimum(L_ratio, 2.0)  # Limit local contrast
            
            # Apply adaptive mapping
            result = img_float.copy()
            for i in range(3):
                result[:,:,i] = result[:,:,i] * L_ratio
            
        elif self.method == 'clahe':
            # CLAHE for local contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back and convert to BGR
            lab = cv2.merge((l, a, b))
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) / 255.0
        else:
            # HDR-like tone mapping
            # Apply gamma curve first
            gamma = 0.4
            img_gamma = np.power(img_float, gamma)
            
            # Apply S-curve for contrast
            result = 0.5 - np.sin(np.arcsin(1.0 - 2.0 * img_gamma) / 3.0)
            
        # Clip and convert back to uint8
        return np.clip(result * 255, 0, 255).astype(np.uint8)


class ImageSharpening(ISPStage):
    def __init__(self, method='usm', amount=1.0, radius=1.0, threshold=0):
        super().__init__("Image Sharpening")
        self.method = method
        self.amount = amount
        self.radius = radius
        self.threshold = threshold
        
    def _process_implementation(self, image):
        """Enhanced image sharpening with multiple methods"""
        if self.method == 'kernel':
            # Basic kernel sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * self.amount
            
            sharpened = cv2.filter2D(image, -1, kernel)
            
        elif self.method == 'usm':
            # Unsharp mask (higher quality)
            blur = cv2.GaussianBlur(image, (0, 0), self.radius)
            sharpened = cv2.addWeighted(image, 1.0 + self.amount, blur, -self.amount, 0)
            
            # Apply threshold if specified
            if self.threshold > 0:
                diff = cv2.absdiff(image, blur)
                mask = diff < self.threshold
                sharpened[mask] = image[mask]
                
        elif self.method == 'adaptive':
            # Adaptive sharpening based on local variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
            edge_map = cv2.absdiff(gray, blur)
            
            # Normalize edge map to 0-1 range
            edge_map = cv2.normalize(edge_map, None, 0, 1, cv2.NORM_MINMAX)
            
            # Create adaptive amount based on edge map
            adaptive_amount = self.amount * (1 - edge_map)
            
            # Apply unsharp mask with varying amount
            blur = cv2.GaussianBlur(image, (0, 0), self.radius)
            sharpened = image.copy().astype(np.float32)
            
            for i in range(3):
                adaptive_weights = 1.0 + np.expand_dims(adaptive_amount, axis=2)
                sharpened = image.astype(np.float32) * adaptive_weights + blur.astype(np.float32) * (1 - adaptive_weights)
        else:
            # Laplacian sharpening
            laplacian = cv2.Laplacian(image, cv2.CV_32F)
            sharpened = image.astype(np.float32) + laplacian * self.amount
            
        return np.clip(sharpened, 0, 255).astype(np.uint8)


class ColorQuantization(ISPStage):
    def __init__(self, levels=32):
        super().__init__("Color Quantization")
        self.levels = levels
        
    def _process_implementation(self, image):
        """Reduce color depth for compression"""
        # Quantize each channel to specified number of levels
        factor = 255 / (self.levels - 1)
        
        # Convert to float, quantize, then scale back
        img_float = image.astype(np.float32)
        quantized = np.round(img_float / factor) * factor
        
        return np.clip(quantized, 0, 255).astype(np.uint8)


class DownscalingFPS(ISPStage):
    def __init__(self, scale_factor=0.5):
        super().__init__("Downscaling FPS")
        self.scale_factor = scale_factor
        
    def _process_implementation(self, image):
        """Downscale image for faster processing"""
        # Calculate new dimensions
        height, width = image.shape[:2]
        new_height, new_width = int(height * self.scale_factor), int(width * self.scale_factor)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized


class DistortionCorrection(ISPStage):
    def __init__(self):
        super().__init__("Distortion Correction")
        
    def _process_implementation(self, image):
        """Correct lens distortion"""
        # Camera matrix (example values)
        height, width = image.shape[:2]
        focal_length = width
        camera_matrix = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])
        
        # Distortion coefficients (example values - slight barrel distortion)
        # [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([-0.1, 0.01, 0, 0, 0])
        
        # Undistort the image
        corrected = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        return corrected


class AdaptiveImageScaling(ISPStage):
    def __init__(self, target_size=None, method='content'):
        super().__init__("Adaptive Image Scaling")
        self.target_size = target_size  # (width, height) or None for adaptive
        self.method = method  # 'content', 'performance', or 'balanced'
        
    def _process_implementation(self, image):
        """Scale image based on content complexity or processing requirements"""
        if self.target_size:
            return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
        # If no target size specified, use adaptive scaling
        height, width = image.shape[:2]
        
        if self.method == 'content':
            # Analyze image complexity using Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            complexity = np.var(laplacian)
            
            # Scale based on complexity (higher complexity = less scaling)
            scale_factor = max(0.25, min(1.0, 1.0 - (complexity / 5000)))
            
        elif self.method == 'performance':
            # Scale based on image size for performance
            pixel_count = width * height
            scale_factor = min(1.0, 1000000 / pixel_count) if pixel_count > 1000000 else 1.0
            
        else:  # 'balanced'
            # Compromise between content and performance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            complexity = np.var(laplacian)
            
            pixel_count = width * height
            content_factor = max(0.5, min(1.0, 1.0 - (complexity / 10000)))
            performance_factor = min(1.0, 2000000 / pixel_count) if pixel_count > 2000000 else 1.0
            
            scale_factor = (content_factor + performance_factor) / 2
        
        # Apply scaling
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class ColorEnhancement(ISPStage):
    """New stage for color enhancement"""
    def __init__(self, saturation=1.2, vibrance=1.1):
        super().__init__("Color Enhancement")
        self.saturation = saturation
        self.vibrance = vibrance
        
    def _process_implementation(self, image):
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Apply global saturation adjustment
        s = s * self.saturation
        
        # Apply vibrance (boost lower saturated colors more)
        # Calculate how much to boost each pixel's saturation
        boost = (1 - s/255.0) * (self.vibrance - 1)
        s = s * (1 + boost)
        
        # Merge and convert back
        hsv = cv2.merge([h, np.clip(s, 0, 255), v])
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return enhanced


class DetailEnhancement(ISPStage):
    """New stage for detail enhancement"""
    def __init__(self, strength=0.5, bilateral_d=9, sigma_color=75, sigma_space=75):
        super().__init__("Detail Enhancement")
        self.strength = strength
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        
    def _process_implementation(self, image):
        # Use detail enhancement from OpenCV's photo module
        enhanced = cv2.detailEnhance(
            image, 
            sigma_s=self.sigma_space/100,
            sigma_r=self.sigma_color/100
        )
        
        # Blend with original based on strength
        result = cv2.addWeighted(
            image, 1 - self.strength,
            enhanced, self.strength,
            0
        )
        
        return result


class ISPPipeline:
    """Image Signal Processing Pipeline"""
    def __init__(self, name):
        self.name = name
        self.stages = []
        
    def add_stage(self, stage):
        """Add a processing stage to the pipeline"""
        self.stages.append(stage)
        return self
        
    def process(self, input_image):
        """Process an image through all stages in the pipeline"""
        logger.info(f"Starting {self.name} pipeline")
        start_time = time.time()
        
        current_image = input_image
        for stage in self.stages:
            current_image = stage.process(current_image)
            
        end_time = time.time()
        logger.info(f"{self.name} pipeline completed in {end_time - start_time:.4f} seconds")
        
        return current_image


def create_hv_pipeline():
    """Create enhanced Human Vision (HV) optimized pipeline"""
    pipeline = ISPPipeline("Human Vision")
    
    # Add improved stages for HV pipeline
    pipeline.add_stage(RawDataCapture())
    pipeline.add_stage(Demosaicing())
    pipeline.add_stage(ColorCorrection(strength=1.1))
    pipeline.add_stage(WhiteBalance(method='adaptive'))
    pipeline.add_stage(GammaCorrection(gamma=2.2))
    pipeline.add_stage(NoiseCorrection(method='nlm', strength=7))
    pipeline.add_stage(PixelErrorDetection(threshold=40))
    pipeline.add_stage(LensShading())
    pipeline.add_stage(ColorEnhancement(saturation=1.2, vibrance=1.15))
    pipeline.add_stage(DetailEnhancement(strength=0.4))
    pipeline.add_stage(ToneMapping(method='adaptive_reinhard'))
    pipeline.add_stage(ImageSharpening(method='usm', amount=0.8, radius=0.5))
    pipeline.add_stage(ColorQuantization(levels=64))  # Higher color depth
    pipeline.add_stage(DistortionCorrection())
    
    return pipeline


def create_cv_pipeline():
    """Create enhanced Computer Vision (CV) optimized pipeline"""
    pipeline = ISPPipeline("Computer Vision")
    
    # Add improved stages for CV pipeline
    pipeline.add_stage(RawDataCapture())
    pipeline.add_stage(Demosaicing())
    pipeline.add_stage(ColorCorrection(strength=0.9))  # Less color correction
    pipeline.add_stage(WhiteBalance(method='gray_world'))  # More neutral colors
    pipeline.add_stage(GammaCorrection(gamma=1.8))  # More linear response
    pipeline.add_stage(NoiseCorrection(method='bilateral'))  # Preserve edges better
    pipeline.add_stage(PixelErrorDetection(threshold=35))
    pipeline.add_stage(LensShading())
    # No color enhancement for CV - keep colors more accurate
    pipeline.add_stage(ToneMapping(method='clahe'))  # Better local contrast
    pipeline.add_stage(ImageSharpening(method='adaptive', amount=0.6))
    pipeline.add_stage(DownscalingFPS(scale_factor=0.6))
    
    return pipeline


def process_with_concurrent_pipelines(image_path, pipelines):
    """Process an image through multiple pipelines concurrently"""
    with ThreadPoolExecutor() as executor:
        # Submit all pipeline jobs
        futures = {}
        for name, pipeline in pipelines.items():
            futures[executor.submit(pipeline.process, image_path)] = name
            
        # Collect results
        results = {}
        for future in futures:
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logger.error(f"Error in {name} pipeline: {str(e)}")
                results[name] = None
                
        return results


def static_analysis_concurrent_pipelines(num_sensors=1):
    """Perform static analysis of concurrent pipelines for multiple sensors"""
    # Create pipelines
    hv_pipeline = create_hv_pipeline()
    cv_pipeline = create_cv_pipeline()
    
    # Analysis results
    results = {
        "pipelines": [],
        "total_stages": 0,
        "estimated_memory_usage": 0,
        "estimated_parallel_speedup": 0
    }
    
    # Analyze HV pipeline
    hv_analysis = {
        "name": "Human Vision",
        "stages": len(hv_pipeline.stages),
        "stages_detail": [stage.name for stage in hv_pipeline.stages]
    }
    results["pipelines"].append(hv_analysis)
    
    # Analyze CV pipeline
    cv_analysis = {
        "name": "Computer Vision",
        "stages": len(cv_pipeline.stages),
        "stages_detail": [stage.name for stage in cv_pipeline.stages]
    }
    results["pipelines"].append(cv_analysis)
    
    # Total stages across all pipelines and sensors
    results["total_stages"] = (hv_analysis["stages"] + cv_analysis["stages"]) * num_sensors
    
    # Estimate memory usage (very rough estimate)
    # Assuming 24MB per 8MP image per stage
    memory_per_image = 24  # MB
    results["estimated_memory_usage"] = memory_per_image * results["total_stages"]
    
    # Estimate parallel speedup (idealized)
    serial_time = results["total_stages"]
    parallel_time = max(hv_analysis["stages"], cv_analysis["stages"])
    results["estimated_parallel_speedup"] = serial_time / parallel_time
    
    # Add sensor-specific analysis
    results["sensors"] = {
        "count": num_sensors,
        "estimated_throughput": f"{1000/max(hv_analysis['stages'], cv_analysis['stages']):.2f} images/sec (theoretical)"
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python isp_pipeline.py <image_path> [num_sensors]")
        sys.exit(1)
        
    image_path = sys.argv[1]
    num_sensors = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    # Create pipelines
    hv_pipeline = create_hv_pipeline()
    cv_pipeline = create_cv_pipeline()
    
    # Process image through both pipelines
    pipelines = {
        "HV": hv_pipeline,
        "CV": cv_pipeline
    }
    
    # Perform static analysis
    analysis = static_analysis_concurrent_pipelines(num_sensors)
    print("\nStatic Analysis of Concurrent Pipelines:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Process the image
    results = process_with_concurrent_pipelines(image_path, pipelines)
    
    # Display results
    for name, result in results.items():
        if result is not None:
            # Save the processed image
            output_path = f"output_{name}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved {name} processed image to {output_path}") 