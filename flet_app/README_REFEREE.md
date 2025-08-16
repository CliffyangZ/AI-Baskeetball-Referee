# AI Basketball Referee - Real-time Detection Interface

## 概述
這是一個基於 Flet 的實時籃球裁判系統前端介面，整合了 `referee.py` 的檢測功能，提供實時的籃球、籃框和姿態檢測，並顯示詳細的統計資訊。

## 功能特色

### 🏀 實時檢測
- **籃球追蹤**: 實時檢測和追蹤籃球位置
- **籃框檢測**: 識別籃框位置和狀態
- **姿態估計**: 檢測球員姿態和動作
- **投籃檢測**: 自動識別投籃動作和結果

### 📊 實時統計
- **投籃統計**: 投籃次數、命中率、成功率
- **運球計數**: 實時運球次數統計
- **步數統計**: 每位球員的步數追蹤
- **違規檢測**: 自動檢測籃球違規行為
- **性能監控**: FPS 和處理時間顯示

### 🎮 互動控制
- **攝像頭選擇**: 支援多攝像頭切換
- **檢測控制**: 即時開始/停止檢測
- **視覺化設定**: 可調整檢測顯示選項
- **信心度調整**: 可調整檢測敏感度

## 文件結構

```
flet_app/
├── main_referee.py                    # 主程式入口
├── backend/
│   └── referee_integration.py        # 後端整合模組
├── frontend/
│   ├── components/
│   │   └── statistics_panel.py       # 統計面板組件
│   └── pages/
│       ├── home_page.py              # 主頁面 (已更新)
│       └── referee_page.py           # 裁判檢測頁面
└── README_REFEREE.md                 # 本說明文件
```

## 安裝和運行

### 1. 環境要求
```bash
# 確保已安裝必要的依賴
pip install flet opencv-python numpy
```

### 2. 運行方式

#### 方法 1: 從主頁面啟動
```bash
cd flet_app
python main.py
```
然後點擊 "Launch Referee Detection" 按鈕

#### 方法 2: 直接啟動裁判介面
```bash
cd flet_app
python main_referee.py
```

### 3. 使用說明

1. **選擇攝像頭**: 從下拉選單選擇要使用的攝像頭
2. **開始檢測**: 點擊 "Start Detection" 或按空白鍵
3. **查看統計**: 右側面板顯示實時統計資訊
4. **調整設定**: 使用檢測控制面板調整顯示選項
5. **停止檢測**: 點擊 "Stop Detection" 或按空白鍵
6. **退出程式**: 按 ESC 鍵

## 核心組件說明

### RefereeIntegration (`backend/referee_integration.py`)
- 整合 `referee.py` 的檢測功能
- 提供多線程處理框架
- 管理檢測結果和統計資料

### StatisticsPanel (`frontend/components/statistics_panel.py`)
- 實時統計資訊顯示
- 包含投籃、運球、步數、違規統計
- 性能指標顯示 (FPS, 處理時間)

### DetectionVisualizationPanel
- 檢測視覺化控制
- 信心度調整滑桿
- 顯示選項開關

### RefereePage (`frontend/pages/referee_page.py`)
- 主要檢測介面
- 攝像頭串流處理
- 統計資料更新管理

## 技術特點

### 🔄 多線程架構
- 主 UI 線程: 處理使用者介面
- 攝像頭線程: 處理視訊串流
- 檢測線程: 執行 AI 檢測
- 統計更新線程: 更新統計顯示

### 📈 實時性能
- 目標 30 FPS 攝像頭串流
- 100ms 統計更新頻率
- 非阻塞式框架處理
- 智慧框架丟棄機制

### 🎯 檢測整合
- 完整整合 `referee.py` 功能
- 支援所有檢測類型 (籃球、籃框、姿態)
- 實時統計資料同步
- 視覺化結果顯示

## 快捷鍵

- **空白鍵**: 開始/停止檢測
- **ESC**: 退出程式

## 故障排除

### 攝像頭問題
- 確保攝像頭已連接且可用
- 檢查攝像頭權限設定
- 嘗試不同的攝像頭索引

### 模型載入問題
- 確保模型文件存在於正確路徑
- 檢查 `models/ov_models/` 目錄
- 確認 OpenVINO 環境設定

### 性能問題
- 降低攝像頭解析度
- 調整檢測信心度閾值
- 關閉不需要的視覺化選項

## 開發說明

### 擴展統計功能
在 `StatisticsPanel` 中添加新的統計項目:
```python
def update_statistics(self, stats: Dict[str, Any]):
    # 添加新的統計項目
    new_stat = stats.get('new_stat', 0)
    self.new_stat_text.value = f"New Stat: {new_stat}"
```

### 添加新的檢測控制
在 `DetectionVisualizationPanel` 中添加新的控制項:
```python
self.new_control = ft.Checkbox(
    label="New Control", 
    value=True, 
    on_change=self._on_toggle_change
)
```

### 自定義檢測參數
在 `RefereeIntegration` 中調整檢測參數:
```python
def initialize_referee(self, custom_param=None):
    # 添加自定義參數
    self.referee = BasketballReferee(
        custom_param=custom_param,
        # ... 其他參數
    )
```

## 版本資訊
- 版本: 1.0.0
- 基於: referee.py 檢測系統
- 框架: Flet UI
- 支援: Python 3.8+
