# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

class FrameSampler:
    """Computes adaptive sampling stride to meet performance constraints (e.g. CPU < 8s limit)."""
    
    def __init__(self, has_gpu: bool = False):
        self.has_gpu = has_gpu
        self.next_sample_time = 0.0

    def should_sample(self, timestamp: float) -> bool:
        """Determines if the frame at timestamp should be processed."""
        if timestamp >= self.next_sample_time - 1e-5:
            return True
        return False

    def update_stride(self, has_detections: bool, timestamp: float, duration: float = 0.0):
        """Computes next sample target timestamp based on current detections and performance mode."""
        if self.has_gpu:
            stride = 0.10 if has_detections else 0.34
        else:
            # CPU fallback: process less frames to optimize performance
            stride = 0.50 if has_detections else 1.5

        # Sample more frequently near the end of video (last 10%)
        if duration > 0.0 and timestamp > 0.0:
            remaining_ratio = 1.0 - (timestamp / duration)
            if remaining_ratio < 0.10:
                stride = min(stride, 0.50)
                
        self.next_sample_time = timestamp + stride
