# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

"""F6 Task: Cabin ROI Detector for Passengers and Laptops.

Uses pretrained COCO detections to identify 'bilgisayar' (laptop) and map
passengers to ROIs: on_koltuk, arka_koltuk_1, arka_koltuk_2.
"""

class CabinRoiDetector:
    """Detector for cabin ROIs (passenger seats) and laptops using COCO detections."""

    def __init__(self):
        # ROI ranges normalized [min_val, max_val]
        self.rois = {
            "on_koltuk": {
                "x": (0.0, 0.50),
                "y": (0.10, 0.70)
            },
            "arka_koltuk_1": {
                "x": (0.50, 1.00),
                "y": (0.10, 0.55)
            },
            "arka_koltuk_2": {
                "x": (0.50, 1.00),
                "y": (0.55, 1.00)
            }
        }

    def process_detections(self, coco_detections: list, width: int, height: int) -> list[dict]:
        """Process COCO detections to find laptop and passengers in ROIs.

        Args:
            coco_detections: List of detections where each is dict:
                             {"bbox": [x1, y1, x2, y2], "class_id": int, "confidence": float}
            width: Frame width
            height: Frame height

        Returns:
            List of detected events: [{"label": str, "confidence": float}]
        """
        events = []

        for det in coco_detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            conf = det["confidence"]

            x1, y1, x2, y2 = bbox
            
            if class_id == 73:  # laptop
                events.append({
                    "label": "bilgisayar",
                    "confidence": conf
                })
                
            elif class_id == 0:  # person
                # Calculate normalized center coordinates
                cx = (x1 + x2) / 2.0 / width
                cy = (y1 + y2) / 2.0 / height

                # Determine which ROI the person center belongs to
                for roi_name, limits in self.rois.items():
                    x_lim = limits["x"]
                    y_lim = limits["y"]

                    if x_lim[0] <= cx <= x_lim[1] and y_lim[0] <= cy <= y_lim[1]:
                        events.append({
                            "label": roi_name,
                            "confidence": conf
                        })
                        break  # Person can belong to only one ROI

        return events
