import cv2
import numpy as np

def detect_card(frame):
    """
    フレーム内で名刺を検出する。
    名刺が検出されると、その輪郭を返す。

    Args:
        frame (numpy.ndarray): カメラからの画像フレーム

    Returns:
        tuple: (boolean, numpy.ndarray)
               - boolean: 名刺が検出されたかどうか
               - contour: 名刺の輪郭（検出された場合）
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 輪郭を抽出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 輪郭の面積を確認（小さいノイズを除外）
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        # 輪郭が四角形に近いか判定
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4: # 四角形の場合
            return True, approx
        
    return False, None

def main():
    """
    カメラを使用して名刺をリアルタイムで検出するプログラム
    名刺が検出されると、その輪郭を描画して表示
    """
    # カメラを初期化
    cap = cv2.VideoCapture(0) # 0はデフォルトのカメラ

    if not cap.isOpened():
        print("カメラが見つかりませんでした。")
        return
    
    print("カメラを起動しました。'q'を押して終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        # 名刺検出
        detected, contour = detect_card(frame)

        if detected:
            # 名刺の輪郭を描画
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, "Card Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # フレームを表示
        cv2.imshow("Business Card Detector", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()