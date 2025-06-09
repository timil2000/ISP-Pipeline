import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from isp_pipeline import (
    create_hv_pipeline,
    create_cv_pipeline,
    process_with_concurrent_pipelines,
    static_analysis_concurrent_pipelines,
    AdaptiveImageScaling
)

def generate_test_image(width=800, height=600):
    """Generate a synthetic test image with a color gradient and some patterns"""
    # Create a base color gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # RGB channels with different patterns
    r = np.sin(xv * 10) * 0.5 + 0.5
    g = np.sin(yv * 10) * 0.5 + 0.5
    b = np.sin((xv + yv) * 10) * 0.5 + 0.5
    
    # Create an RGB image
    img = np.zeros((height, width, 3))
    img[:,:,0] = b * 255
    img[:,:,1] = g * 255
    img[:,:,2] = r * 255
    
    # Add some noise
    noise = np.random.normal(0, 15, (height, width, 3))
    img = img + noise
    
    # Add a few bad pixels
    for _ in range(20):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        img[y, x, :] = 255
    
    return np.clip(img, 0, 255).astype(np.uint8)

def plot_pipeline_results(original, hv_result, cv_result):
    """Display original and processed images"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(hv_result, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Human Vision (HV) Pipeline')
    axs[1].axis('off')
    
    axs[2].imshow(cv2.cvtColor(cv_result, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Computer Vision (CV) Pipeline')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pipeline_comparison.png')
    plt.close()

def test_adaptive_scaling():
    """Test the adaptive image scaling functionality"""
    # Generate test image
    test_img = generate_test_image(1200, 800)
    
    # Create different scaling instances
    content_scaler = AdaptiveImageScaling(method='content')
    performance_scaler = AdaptiveImageScaling(method='performance')
    balanced_scaler = AdaptiveImageScaling(method='balanced')
    
    # Apply scaling
    content_result = content_scaler.process(test_img)
    performance_result = performance_scaler.process(test_img)
    balanced_result = balanced_scaler.process(test_img)
    
    # Display results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    axs[0, 0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f'Original ({test_img.shape[1]}x{test_img.shape[0]})')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(cv2.cvtColor(content_result, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title(f'Content-based ({content_result.shape[1]}x{content_result.shape[0]})')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(cv2.cvtColor(performance_result, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title(f'Performance-based ({performance_result.shape[1]}x{performance_result.shape[0]})')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(cv2.cvtColor(balanced_result, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title(f'Balanced ({balanced_result.shape[1]}x{balanced_result.shape[0]})')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('adaptive_scaling_comparison.png')
    plt.close()
    
    return {
        'original_size': test_img.shape[:2],
        'content_size': content_result.shape[:2],
        'performance_size': performance_result.shape[:2],
        'balanced_size': balanced_result.shape[:2]
    }

def run_static_analysis():
    """Run and display static analysis for different numbers of sensors"""
    sensor_counts = [1, 2, 4, 8]
    results = []
    
    for count in sensor_counts:
        analysis = static_analysis_concurrent_pipelines(count)
        results.append({
            'sensor_count': count,
            'memory_usage': analysis['estimated_memory_usage'],
            'parallel_speedup': analysis['estimated_parallel_speedup'],
            'total_stages': analysis['total_stages']
        })
    
    # Create plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Memory usage
    counts = [r['sensor_count'] for r in results]
    memory = [r['memory_usage'] for r in results]
    axs[0].bar(counts, memory)
    axs[0].set_xlabel('Number of Sensors')
    axs[0].set_ylabel('Estimated Memory Usage (MB)')
    axs[0].set_title('Memory Usage by Sensor Count')
    
    # Parallel speedup vs. total stages
    speedup = [r['parallel_speedup'] for r in results]
    axs[1].plot(counts, speedup, 'o-', label='Parallel Speedup')
    axs[1].set_xlabel('Number of Sensors')
    axs[1].set_ylabel('Speedup Factor')
    axs[1].set_title('Parallel Processing Efficiency')
    
    plt.tight_layout()
    plt.savefig('static_analysis.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Generate a test image
    print("Generating test image...")
    test_image = generate_test_image()
    cv2.imwrite("test_image.jpg", test_image)
    
    # Create the pipelines
    hv_pipeline = create_hv_pipeline()
    cv_pipeline = create_cv_pipeline()
    
    # Process the image through both pipelines
    print("Processing image through both pipelines...")
    pipelines = {
        "HV": hv_pipeline,
        "CV": cv_pipeline
    }
    
    start_time = time.time()
    results = process_with_concurrent_pipelines(test_image, pipelines)
    end_time = time.time()
    
    print(f"Concurrent processing completed in {end_time - start_time:.4f} seconds")
    
    # Save results
    for name, result in results.items():
        if result is not None:
            output_path = f"output_{name}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved {name} processed image to {output_path}")
    
    # Plot results
    plot_pipeline_results(test_image, results["HV"], results["CV"])
    print("Pipeline comparison saved to pipeline_comparison.png")
    
    # Test adaptive scaling
    print("\nTesting adaptive image scaling...")
    scaling_results = test_adaptive_scaling()
    print("Adaptive scaling comparison saved to adaptive_scaling_comparison.png")
    print("Scaling results:", scaling_results)
    
    # Run static analysis
    print("\nRunning static analysis for multiple sensors...")
    analysis_results = run_static_analysis()
    print("Static analysis saved to static_analysis.png")
    
    for result in analysis_results:
        print(f"Sensors: {result['sensor_count']}, " +
              f"Memory: {result['memory_usage']} MB, " +
              f"Speedup: {result['parallel_speedup']:.2f}x") 