import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# 保存先ディレクトリ
SAVE_DIR = "detected_frames"

# 保存先ディレクトリを作成（存在しない場合）
os.makedirs(SAVE_DIR, exist_ok=True)

# YOLOモデルの読み込み
MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)

def save_frame(frame):
    """
    フレームを指定したディレクトリに保存する。

    Args:
        frame (numpy.ndarray): 保存対象のフレーム
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f") # タイムスタンプを作成
    filename = os.path.join(SAVE_DIR, f"detected_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"検出されたフレームを保存しました: {filename}")

def transform_card(frame, contour):
    """
    名刺領域を透視変換して正しい形に補正する。

    Args:
        frame (numpy.ndarray): 元のフレーム
        contour (numpy.ndarray): 名刺の四角形輪郭

    Returns:
        numpy.ndarray: 補正された名刺画像
    """
    # 頂点の順序を整列
    contour = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = contour.sum(axis=1)
    rect[0] = contour[np.argmin(s)] # 左上
    rect[2] = contour[np.argmax(s)] # 右下

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)] # 右上
    rect[3] = contour[np.argmax(diff)] # 左下

    # 幅と高さを計算
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    # 変換後の座標
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 透視変換行列を計算し、画像を補正
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))

    return warped

def detect_yolo(frame):
    """
    フレーム内でYOLOを使用して名刺を検出し、検出結果を保存する。

    Args:
        frame (numpy.ndarray): カメラまたは画像からのフレーム
    """
    # YOLO推論
    results = model(frame)
    detections = results[0].boxes.xyxy.numpy()  # 検出されたバウンディングボックス
    class_ids = results[0].boxes.cls.numpy()  # 検出されたクラスID
    confidences = results[0].boxes.conf.numpy()  # 信頼度スコア

    for i, box in enumerate(detections):
        # バウンディングボックスと信頼度を取得
        x1, y1, x2, y2 = map(int, box)
        class_id = int(class_ids[i])
        confidence = confidences[i]

        if class_id in [0]:  # クラスIDはモデルのラベルに応じて調整
            print(f"検出: ({x1}, {y1}, {x2}, {y2}), クラスID: {class_id}, 信頼度: {confidence:.2f}")
            card_image = frame[y1:y2, x1:x2]  # 名刺領域を切り出し
            save_frame(card_image)

        # バウンディングボックスを描画
        label = f"Class {class_id}, Conf: {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

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

    # med_val = np.median(blurred)
    # sigma = 0.33  # 0.33
    # min_val = int(max(0, (1.0 - sigma) * med_val))
    # max_val = int(max(255, (1.0 + sigma) * med_val))
    # edges = cv2.Canny(blurred, min_val, max_val)

    # edges = cv2.Canny(blurred, 30, 400)


    # Sobelフィルタを使用したエッジ検出
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向

    # エッジの大きさを計算
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(sobel_magnitude)  # データ型を8ビットに変換

    # エッジ画像の2値化（しきい値を適用）
    _, edges = cv2.threshold(sobel_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # cv2.imshow("Edges", edges)

    # 輪郭を抽出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 全て白の画像を作成
    # img_blank = np.ones_like(frame) * 255
    # 輪郭だけを描画（黒色で描画）
    # img_contour_only = cv2.drawContours(img_blank, contours, -1, (0,0,0), 3)
    # 描画
    # plt.imshow(cv2.cvtColor(img_contour_only, cv2.COLOR_BGR2RGB))
    # cv2.imshow("contour", img_contour_only)

    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 1)  # 青色で描画

        # 輪郭の面積を確認（小さいノイズを除外）
        area = cv2.contourArea(contour)
        if area < 50000:
            continue
        # 輪郭が四角形に近いか判定
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        # print(len(approx))
        if len(approx) == 4: # 四角形の場合
            print(area)
            return True, approx
        
    return False, None

def draw_guidelines(frame):
    """
    フレームにガイドラインを描画する。

    Args:
        frame (numpy.ndarray): カメラからの画像フレーム
    Returns:
        tuple: (tuple, tuple)
               - ガイドラインの左上と右下の座標
    """
    height, width, _ = frame.shape
    # 中身領域の座標
    top_left = (width // 4, height // 4)
    bottom_right = (width * 3 // 4, height * 3 // 4)
    # ガイドラインを描画
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    return top_left, bottom_right

def process_image(image_path):
    """
    画像ファイルを読み込み、名刺検出を実行する。

    Args:
        image_path (str): 画像ファイルのパス
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f'画像を読み込めませんでした: {image_path}')
        return

    print(f"画像を処理中: {image_path}")
    detected, contour = detect_card(frame)
    if detected:
        # 名刺領域を透視変換して保存
        card_image = transform_card(frame, contour)
        save_frame(card_image)
        print("名刺が検出されました")
    else:
        print("名刺が検出されませんでした")
    # frame = detect_yolo(frame)

    # cv2.imshow("Processed Image", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    """
    カメラを使用して名刺をリアルタイムで検出するプログラム
    名刺が検出されると、その輪郭を描画して表示
    """
    print("モードを選択してください: ")
    print("1: カメラモード")
    print("2: 画像ファイルモード")

    mode = input("モード番号を入力してください: ")

    if mode == "1":
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

            # ガイドラインを描画し、その範囲を取得
            top_left, bottom_right = draw_guidelines(frame)

            # ガイドライン内の領域を切り出し
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 名刺検出
            detected, contour = detect_card(roi)

            if detected:
                # 輪郭を元の座標系に変換
                contour += np.array([top_left[0], top_left[1]])
                # 名刺の輪郭を描画
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, "Card Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 名刺領域を透視変換して保存
                card_image = transform_card(frame, contour)
                save_frame(card_image)
            
            # フレームを表示
            cv2.imshow("Business Card Detector", frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # リソースを解放
        cap.release()
        cv2.destroyAllWindows()

    elif mode == "2":
        # 画像ファイルモード
        image_dir = input("画像ファイルパスを入力してください: ")
        for file_path in os.listdir(image_dir):
            base, ext = os.path.splitext(file_path)
            if ext == '.jpg' or ext == '.JPG':
                image_path = os.path.join(image_dir, file_path)
                process_image(image_path)
        
    else:
        print("無効なモード番号です。プログラムを終了します。")

if __name__ == "__main__":
    main()