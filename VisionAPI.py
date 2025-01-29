from pathlib import Path
from google.cloud import vision
import re
import os

CONFIG_FILE = 'config/ocr-visionapi-448100-e7af8d74edd3.json'

def classify_text(text):
    """テキストをルールベースで分類"""
    # 正規表現パターン
    patterns = {
        "Unclassified": r"([Ff][Aa][Xx]|[Tt][Ee][Ll]|[Ee]-[Mm][Aa][Ii][Ll])",
        "会社名": r"会社",
        "メールアドレス": r"[\w\.-]+@[\w\.-]+\.\w+",
        "住所": r"(都|道|府|県).*(市|区|町|村).*|.*(丁目|番地|号)|^.\d{1,3}-\d{1,4}$",
        "電話番号": r"(Tel|電話|Mobile|携帯)?.?(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        "FAX": r"(?i)\bfax\b.*\d+",
        "部署": r"(室|部|課|チーム|支店|ブロック|センター|支社|農場|班|グループ|局|ユニット|担当|Division|科|工場|拠点|委員会|G|係)",
        "役職": r"(技師|Assistant Professor|CFO|CTO|DX技術担当|Director|Managing Director|PMO|PRESIDENT|Ph.D.|Sales Manager|Section Manager|アソシエイト|インサイド&フィールドセールス|エンジニア|カスタマーサポート|コンサルタント|コンテンツクリエイター|サブマネージャー|シニアコンサルタント|スパイスアドバイザー|セールスエンジニア|ソリューション担当|ディレクター|トマト生産者|パートナー|プリンシパルエキスパート|プロデューサー|プロフェッショナル|ホーティカルチャースペシャリスト|マネジャー|マネージャー|ミニトマト農家|ロボットエンジニア|上席研究員|准教授|参事 (国内販路開拓担当)|専任職|専務･ジェネラルマネージャー|担当|教授|教育監|整理収納アドバイザー|日本政策金融金庫 農業経営アドバイザー試験合格者|栽培担当|栽培管理スタッフ|物流マネージャー|理事|生産統括|監査担当|研究員|社会教育指導員|福祉住環境コーディネーター|管理部担当|記者|販売管理マネージャー|部活動コーディネーター)",
    }
    
    # カテゴリごとにチェック
    for category, pattern in patterns.items():
        if re.search(pattern, text):
            return category
    
    # 未分類の場合
    return "氏名"

# テキストを行ごとに分類
def classify_lines(lines):
    classified_data = []
    for line in lines:
        category = classify_text(line)
        classified_data.append((line, category))
    return classified_data

def detect(file_path):
    client = vision.ImageAnnotatorClient.from_service_account_file(CONFIG_FILE)

    with open(file_path, 'rb') as image_file:
        content = image_file.read()
        image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    read_text = response.text_annotations[0].description

    # 行単位に分割
    lines = re.split(r'\s+', read_text.strip())

    # 分類実行
    classified = classify_lines(lines)

    # 分類結果を表示
    for line, category in classified:
        print(f"{category} : {line}")

    return

if __name__ == "__main__":
    dir_path = '/Users/yuuki/Desktop/hobby/Scan-with-OpenCV/detected_frames'

    for file in os.listdir(dir_path):
        base, ext = os.path.splitext(file)
        if ext == '.jpg':
            image_path = os.path.join(dir_path, file)
            detect(image_path)
            print('')
    