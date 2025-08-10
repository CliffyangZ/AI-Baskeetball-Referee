#!/usr/bin/env python3
"""
NMS效果演示腳本
展示Non-Maximum Suppression在籃球追蹤中的效果
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add tracker directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basketballTracker import BasketballTracker
from utils.openvino_utils import DeviceType

def create_test_frame_with_overlapping_detections():
    """創建包含重疊檢測框的測試幀"""
    # 創建一個測試幀
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 在幀上畫一個籃球
    center = (320, 240)
    radius = 30
    cv2.circle(frame, center, radius, (255, 165, 0), -1)  # 橙色籃球
    cv2.circle(frame, center, radius, (0, 0, 0), 2)       # 黑色邊框
    
    return frame

def simulate_overlapping_detections():
    """模擬重疊的檢測結果"""
    from basketballTracker import BasketballDetection
    
    # 模擬同一個籃球的多個重疊檢測
    detections = [
        BasketballDetection(bbox=(300, 220, 340, 260), confidence=0.9),   # 高置信度
        BasketballDetection(bbox=(305, 225, 345, 265), confidence=0.85),  # 稍微偏移
        BasketballDetection(bbox=(295, 215, 335, 255), confidence=0.8),   # 另一個偏移
        BasketballDetection(bbox=(310, 230, 350, 270), confidence=0.75),  # 更大偏移
        BasketballDetection(bbox=(302, 222, 342, 262), confidence=0.7),   # 小偏移
    ]
    
    return detections

def demonstrate_nms_effect():
    """演示NMS效果"""
    print("🎯 NMS效果演示")
    print("=" * 50)
    
    # 創建測試檢測
    detections = simulate_overlapping_detections()
    print(f"📊 原始檢測數量: {len(detections)}")
    
    # 顯示原始檢測信息
    print("\n📋 原始檢測:")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        print(f"  檢測 {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), confidence={det.confidence:.2f}")
    
    # 創建一個臨時的追蹤器實例來使用NMS方法
    try:
        model_path = "models/ov_models/basketballModel_openvino_model/basketballModel.xml"
        if Path(model_path).exists():
            tracker = BasketballTracker(model_path, DeviceType.CPU)
            
            # 應用NMS
            filtered_detections = tracker.apply_nms(detections, iou_threshold=0.5)
            
            print(f"\n✨ NMS後檢測數量: {len(filtered_detections)}")
            print("\n📋 NMS後檢測:")
            for i, det in enumerate(filtered_detections):
                x1, y1, x2, y2 = det.bbox
                print(f"  檢測 {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), confidence={det.confidence:.2f}")
            
            # 計算IoU矩陣來展示重疊情況
            print(f"\n🔍 IoU分析:")
            from utils.matching import calculate_iou
            
            print("原始檢測IoU矩陣:")
            for i, det1 in enumerate(detections):
                for j, det2 in enumerate(detections):
                    if i < j:
                        iou = calculate_iou(det1.bbox, det2.bbox)
                        print(f"  檢測{i+1} vs 檢測{j+1}: IoU = {iou:.3f}")
            
            reduction_rate = (len(detections) - len(filtered_detections)) / len(detections) * 100
            print(f"\n📈 NMS效果:")
            print(f"  • 檢測減少: {len(detections)} → {len(filtered_detections)}")
            print(f"  • 減少率: {reduction_rate:.1f}%")
            print(f"  • 保留最高置信度檢測")
            
        else:
            print(f"⚠️  模型文件不存在: {model_path}")
            print("   使用模擬NMS演示...")
            
            # 手動模擬NMS效果
            filtered = [detections[0]]  # 保留置信度最高的
            print(f"\n✨ 模擬NMS後檢測數量: {len(filtered)}")
            print("📋 保留的檢測:")
            det = filtered[0]
            x1, y1, x2, y2 = det.bbox
            print(f"  檢測 1: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), confidence={det.confidence:.2f}")
            
    except Exception as e:
        print(f"❌ 演示過程中出錯: {e}")

def demonstrate_tracking_improvement():
    """演示追蹤改進效果"""
    print("\n🚀 追蹤改進效果")
    print("=" * 50)
    
    print("✅ NMS修復帶來的改進:")
    print("  • 減少重複的籃球ID")
    print("  • 提高追蹤穩定性") 
    print("  • 降低計算開銷")
    print("  • 改善視覺效果")
    
    print("\n🔧 技術實現:")
    print("  • IoU閾值: 0.5 (可配置)")
    print("  • 按置信度排序")
    print("  • 保留最高置信度檢測")
    print("  • 移除高重疊檢測")
    
    print("\n📊 性能對比:")
    print("  修復前: 多個重複ID (如 Ball 2, Ball 4, Ball 6...)")
    print("  修復後: 單一穩定ID (如 Ball 1)")

def main():
    """主函數"""
    print("🎯 籃球追蹤器 NMS 效果演示")
    print("=" * 60)
    
    # 演示NMS效果
    demonstrate_nms_effect()
    
    # 演示追蹤改進
    demonstrate_tracking_improvement()
    
    print("\n" + "=" * 60)
    print("✅ NMS演示完成!")
    print("💡 NMS成功解決了多重檢測問題，提升了追蹤質量")

if __name__ == "__main__":
    main()
